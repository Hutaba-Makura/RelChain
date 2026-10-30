from __future__ import annotations

import argparse
import hashlib
import json
import os
import inspect
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

from .artifacts import build_mode_relations, content_hash, freeze_common, load_common, materialize_common
from .config import MemoryConfig, RelationMode
from .evaluation import EvaluationRecord, all_evidence_retrieved, evidence_recall, token_f1
from .llm import LLMAnswerGenerator, LLMRelationTextGenerator, OpenAICompatibleTransport, ValidatedJSONCaller
from .locomo import load_locomo_sample_data
from .models import Answer, RetrievalResult
from .pipeline import MemorySystem
from .quick_eval import dataset_samples, load_set, save_set, select_set
from .tokens import ApproximateTokenCounter, TransformersTokenCounter
from .relations import RelationBuilder


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _code_hash(*names: str) -> str:
    directory = Path(__file__).parent
    digest = hashlib.sha256()
    for name in sorted(names):
        digest.update(name.encode())
        digest.update((directory / name).read_bytes())
    return digest.hexdigest()


def _identity(row: dict[str, Any]) -> str:
    return json.dumps([row["sample_id"], row["question_id"]], ensure_ascii=False, separators=(",", ":"))


def _sample_path(directory: Path, sample_id: str, suffix: str) -> Path:
    safe = hashlib.sha256(sample_id.encode()).hexdigest()[:12]
    return directory / f"sample-{safe}.{suffix}"


def _append_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise ValueError("journal has a malformed middle record")
            # Keep the interrupted tail for audit. The next append starts a new line.
    return rows


class StageJournal:
    def __init__(self, path: Path, *, stage: str, manifest: dict[str, Any]) -> None:
        self.path = path
        self.stage = stage
        self.manifest = manifest
        self.key = content_hash(manifest)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps({"record_type": "manifest", "stage": stage,
                    "key": self.key, "manifest": manifest}, ensure_ascii=False) + "\n")
        rows = self.rows()
        if not rows or rows[0].get("record_type") != "manifest" or rows[0].get("stage") != stage or rows[0].get("key") != self.key:
            raise ValueError("stage output belongs to different conditions; use a new output path")

    def rows(self) -> list[dict[str, Any]]:
        return _read_jsonl(self.path)

    def _write(self, row: dict[str, Any]) -> None:
        needs_separator = self.path.stat().st_size > 0
        if needs_separator:
            with self.path.open("rb") as stream:
                stream.seek(-1, 2)
                needs_separator = stream.read(1) != b"\n"
        with self.path.open("a", encoding="utf-8") as stream:
            if needs_separator:
                stream.write("\n")
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def completed(self) -> dict[str, dict[str, Any]]:
        rows = self.rows()
        for row in rows:
            if row.get("record_type") == "success" and row.get("payload_hash") != content_hash(row["payload"]):
                raise ValueError("stage result content hash mismatch")
        return {row["question_key"]: row for row in rows
                if row.get("record_type") == "success"}

    def counts(self, target: int) -> dict[str, int]:
        rows = self.rows()
        completed = self.completed()
        failed = {row["question_key"] for row in rows if row.get("record_type") == "failure"} - completed.keys()
        return {"successful_questions": len(completed), "failed_questions": len(failed),
                "unprocessed_questions": max(target - len(completed) - len(failed), 0)}

    def append(self, question_key: str, payload: dict[str, Any], *, seconds: float) -> None:
        self._write({"record_type": "success", "question_key": question_key,
            "seconds": seconds, "payload": payload, "payload_hash": content_hash(payload)})

    def failure(self, question_key: str, exc: Exception) -> None:
        self._write({"record_type": "failure", "question_key": question_key,
            "exception_type": type(exc).__name__})


def _open_journal(path: Path, stage: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    rows = _read_jsonl(path)
    if not rows or rows[0].get("record_type") != "manifest" or rows[0].get("stage") != stage:
        raise ValueError(f"expected a {stage} journal")
    manifest = rows[0]["manifest"]
    if content_hash(manifest) != rows[0].get("key"):
        raise ValueError("stage manifest hash mismatch")
    for row in rows:
        if row.get("record_type") == "success" and row.get("payload_hash") != content_hash(row["payload"]):
            raise ValueError("stage result content hash mismatch")
    completed = {row["question_key"]: row for row in rows if row.get("record_type") == "success"}
    return manifest, completed


def command_set(args: argparse.Namespace) -> None:
    start = perf_counter()
    body = select_set(args.dataset, profile=args.profile, seed=args.seed,
                      sample_indices=tuple(args.samples) if args.samples is not None else None)
    save_set(body, args.output)
    _print({"set": str(args.output), "set_hash": body["content_hash"],
        "profile": body["profile"], "actual_by_category": body["actual_by_category"],
        "actual_by_sample": body["actual_by_sample"],
        "shortage_by_category": body["shortage_by_category"],
        "redistributed_count": body["redistributed_count"],
        "selected_count": body["selected_count"], "selection_seconds": perf_counter() - start})


def _common_key(turns: list[Any]) -> str:
    config = MemoryConfig(relation_mode=RelationMode.FIXED)
    return content_hash({
        "schema": 1,
        "turns": [turn.model_dump(mode="json") for turn in turns],
        "build_config": config.model_dump(mode="json", exclude={
            "relation_mode", "reader_name", "context_token_budget", "full_input_token_budget",
            "max_hops", "max_episodes", "include_relation_context", "category_boost",
            "confidence_policy"}),
        "implementation": content_hash({
            "files": _code_hash("extraction.py", "episodes.py", "state.py",
                "embeddings.py", "models.py"),
            "pipeline": _code_hash("pipeline.py"),
            "snapshot": inspect.getsource(freeze_common),
            "candidate_generation": inspect.getsource(RelationBuilder.candidate_pairs),
        }),
    })


def command_build_common(args: argparse.Namespace) -> None:
    command_started = perf_counter()
    selected = load_set(args.set)
    dataset_start = perf_counter()
    samples, dataset_hash = dataset_samples(args.dataset)
    dataset_read_seconds = perf_counter() - dataset_start
    if dataset_hash != selected["dataset_hash"]:
        raise ValueError("evaluation set and dataset content differ")
    reports = []
    for index, sample_id in zip(selected["sample_indices"], selected["sample_ids"]):
        started = perf_counter()
        read_start = perf_counter()
        turns, _, loaded_id = load_locomo_sample_data(samples, index)
        read_seconds = perf_counter() - read_start
        if loaded_id != sample_id:
            raise ValueError("sample ID differs from evaluation set")
        expected_key = _common_key(turns)
        artifact_path = _sample_path(args.artifact_dir, sample_id, "common.json")
        if artifact_path.exists():
            if args.rebuild:
                raise ValueError("rebuild requires a new artifact directory; existing artifact is immutable")
            artifact = load_common(artifact_path)
            if artifact.get("build_key") != expected_key:
                raise ValueError("common artifact build inputs or implementation changed; use a new artifact directory")
            cache_hit = True
            metrics: dict[str, Any] = {"data_load_seconds": read_seconds,
                "candidate_graph_seconds": 0.0, "event_extraction_seconds": 0.0,
                "episode_construction_seconds": 0.0, "state_update_seconds": 0.0,
                "embedding_seconds": 0.0, "db_write_seconds": 0.0}
        else:
            database = _sample_path(args.artifact_dir, sample_id, "common.db")
            system = MemorySystem(database,
                                  MemoryConfig(relation_mode=RelationMode.FIXED))
            prior_key = system.store.get_metadata("common_build_complete_key")
            if prior_key is not None and prior_key != expected_key:
                raise ValueError("common DB belongs to a different build; use a new artifact directory")
            if prior_key is None and any(system.store.counts().values()):
                raise ValueError("common DB is incomplete; use a new artifact directory")
            if prior_key is None:
                report = system.ingest(turns, build_relations=False)
                if report.failures or system.store.counts()["turns"] != len(turns) or any(
                    not system.store.has_events_for_turn(turn.conversation_id, turn.turn_id) for turn in turns
                ):
                    raise ValueError("common memory construction is incomplete")
                system.store.set_metadata("common_build_complete_key", expected_key)
                metrics = {"data_load_seconds": read_seconds, **{
                    key + "_seconds": value for key, value in report.timings_seconds.items()}}
            else:
                metrics = {"data_load_seconds": read_seconds, "common_db_cache_hit": 1.0}
            artifact = freeze_common(system, artifact_path, build_key=expected_key, timings=metrics)
            cache_hit = False
        reports.append({"sample_id": sample_id, "artifact": str(artifact_path),
            "artifact_hash": artifact["content_hash"], "cache_hit": cache_hit,
            "reused_artifacts": ["common_memory", "candidate_graph"] if cache_hit else [],
            "built_artifacts": [] if cache_hit else ["common_memory", "candidate_graph"],
            "counts": {"turns": len(artifact["common"]["turns"]),
                "events": len(artifact["common"]["events"]),
                "episodes": len(artifact["common"]["episodes"]),
                "candidates": len(artifact["candidates"])},
            "timings_seconds": metrics, "total_seconds": perf_counter() - started})
    summary = {"stage": "build_common", "set_hash": selected["content_hash"],
            "dataset_read_seconds": dataset_read_seconds, "samples": reports,
            "initial_build_seconds": sum(row["total_seconds"] for row in reports if not row["cache_hit"]),
            "cached_reuse_seconds": sum(row["total_seconds"] for row in reports if row["cache_hit"]),
            "total_seconds": perf_counter() - command_started}
    _append_report(args.artifact_dir / "build-common-runs.jsonl", summary)
    _print(summary)


def _relation_signature(args: argparse.Namespace) -> str:
    llm_implementation = None
    if args.relation_generation == "llm":
        llm_implementation = content_hash({
            "relation_generator": inspect.getsource(LLMRelationTextGenerator),
            "caller": inspect.getsource(ValidatedJSONCaller),
            "transport": inspect.getsource(OpenAICompatibleTransport),
        })
    return content_hash({"schema": 1, "generation": args.relation_generation,
        "model": args.relation_model if args.relation_generation == "llm" else None,
        "endpoint_hash": hashlib.sha256(args.relation_base_url.encode()).hexdigest() if args.relation_generation == "llm" else None,
        "seed": args.seed,
        "prompt_version": LLMRelationTextGenerator.PROMPT_VERSION if args.relation_generation == "llm" else "deterministic-v1",
        "implementation": _code_hash("relations.py"),
        "llm_implementation": llm_implementation,
    })


def command_build_mode(args: argparse.Namespace) -> None:
    command_started = perf_counter()
    selected = load_set(args.set)
    if args.relation_mode == "fixed" and args.relation_generation == "llm":
        raise ValueError("fixed mode does not call the relation LLM")
    if args.relation_generation == "llm" and (not args.relation_model or not args.relation_base_url):
        raise ValueError("LLM relation generation requires --relation-model and --relation-base-url")
    signature = _relation_signature(args)
    reports = []
    for sample_id in selected["sample_ids"]:
        started = perf_counter()
        artifact = load_common(_sample_path(args.artifact_dir, sample_id, "common.json"))
        relation_generator = None
        transport = None
        if args.relation_generation == "llm":
            transport = OpenAICompatibleTransport(base_url=args.relation_base_url,
                model=args.relation_model, api_key=os.getenv(args.api_key_env) if args.api_key_env else None,
                seed=args.seed, timeout_seconds=args.timeout,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            relation_generator = LLMRelationTextGenerator(ValidatedJSONCaller(transport))
        database = _sample_path(args.mode_db_dir, sample_id, f"{args.relation_mode}.db")
        system = MemorySystem(database, MemoryConfig(relation_mode=RelationMode(args.relation_mode),
            confidence_policy=args.confidence_policy), relation_text_generator=relation_generator)
        metrics: dict[str, Any] = {}
        relation_hash = build_mode_relations(system, artifact, generation_signature=signature, metrics=metrics)
        reports.append({"sample_id": sample_id, "database": str(database),
            "common_artifact_hash": artifact["content_hash"], "relation_hash": relation_hash,
            "candidate_count": len(artifact["candidates"]),
            "edge_count": system.store.counts()["relations"],
            "llm_calls": max(transport.request_count,
                relation_generator.caller.call_count + relation_generator.caller.retry_count) if transport else 0,
            "llm_retries": relation_generator.caller.retry_count if relation_generator else 0,
            "llm_prompt_tokens": transport.prompt_tokens if transport else 0,
            "llm_completion_tokens": transport.completion_tokens if transport else 0,
            "metrics": metrics, "total_seconds": perf_counter() - started})
    summary = {"stage": "build_mode", "set_hash": selected["content_hash"],
               "relation_mode": args.relation_mode, "generation_signature": signature,
               "samples": reports,
               "initial_build_seconds": sum(row["total_seconds"] for row in reports if not row["metrics"]["cache_hit"]),
               "cached_reuse_seconds": sum(row["total_seconds"] for row in reports if row["metrics"]["cache_hit"]),
               "total_seconds": perf_counter() - command_started}
    _append_report(args.mode_db_dir / "build-mode-runs.jsonl", summary)
    _print(summary)


def _counter(path: str | None):
    return TransformersTokenCounter(path) if path else ApproximateTokenCounter()


def command_retrieve(args: argparse.Namespace) -> None:
    command_started = perf_counter()
    selected = load_set(args.set)
    counter = _counter(args.tokenizer)
    config = MemoryConfig(relation_mode=RelationMode(args.relation_mode),
        context_token_budget=args.context_tokens, max_hops=args.max_hops,
        max_episodes=args.max_episodes, episode_candidates=args.episode_candidates,
        state_candidates=args.state_candidates, neighbors_per_node=args.neighbors_per_node,
        include_relation_context=not args.no_relation_context,
        category_boost=args.category_boost)
    artifacts = {}
    for sample_id in selected["sample_ids"]:
        artifact = load_common(_sample_path(args.artifact_dir, sample_id, "common.json"))
        database = _sample_path(args.mode_db_dir, sample_id, f"{args.relation_mode}.db")
        if not database.exists():
            raise ValueError("KG database is missing; run locomo-build-mode first")
        system = MemorySystem(database, config, token_counter=counter)
        if system.store.get_metadata("common_artifact_hash") != artifact["content_hash"]:
            raise ValueError("KG common artifact differs from evaluation set")
        materialize_common(system, artifact)  # validation only when the hash exists
        relation_hash = system.store.get_metadata("relation_artifact_hash")
        if relation_hash is None:
            raise ValueError("KG relation artifact is incomplete; finish build-mode first")
        relations = system.store.list_relations()
        if len(relations) != len(artifact["candidates"]) or content_hash([item.model_dump(mode="json") for item in relations]) != relation_hash:
            raise ValueError("KG relation artifact failed integrity check")
        artifacts[sample_id] = {"system": system, "common_hash": artifact["content_hash"],
            "relation_hash": relation_hash}
    manifest = {"schema_version": 1, "set_hash": selected["content_hash"],
        "kg": {sample: {k: value for k, value in item.items() if k != "system"}
               for sample, item in artifacts.items()},
        "search_config": config.model_dump(mode="json"), "token_counter": counter.name,
        "retrieval_implementation": _code_hash("retrieval.py", "tokens.py", "models.py")}
    journal = StageJournal(args.output, stage="retrieval", manifest=manifest)
    completed = journal.completed()
    hits = 0
    new = 0
    seconds = 0.0
    for row in selected["questions"]:
        key = _identity(row)
        if key in completed:
            hits += 1
            continue
        start = perf_counter()
        try:
            retrieval = artifacts[row["sample_id"]]["system"].retriever.retrieve(row["question"])
            trace = retrieval.trace
            required = tuple(row["evidence"])
            metrics = {
                "retrieved_evidence_recall": evidence_recall(trace.evidence_turn_ids, required),
                "context_evidence_recall": evidence_recall(trace.context_turn_ids, required),
                "retrieved_all_required_evidence": all_evidence_retrieved(trace.evidence_turn_ids, required),
                "context_all_required_evidence": all_evidence_retrieved(trace.context_turn_ids, required),
            }
            elapsed = perf_counter() - start
            journal.append(key, {"sample_id": row["sample_id"], "question_id": row["question_id"],
                "category": row["category"], "retrieval": retrieval.model_dump(mode="json"),
                "metrics": metrics}, seconds=elapsed)
            new += 1
            seconds += elapsed
        except Exception as exc:
            journal.failure(key, exc)
    finished = list(journal.completed().values())
    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {"count": len(items),
            "input_drop_count": sum(set(item["payload"]["retrieval"]["trace"]["evidence_turn_ids"])
                                    != set(item["payload"]["retrieval"]["trace"]["context_turn_ids"])
                                    for item in items),
            "search_seconds": sum(item["payload"]["retrieval"]["trace"].get("search_seconds") or 0 for item in items),
            "context_build_seconds": sum(item["payload"]["retrieval"]["trace"].get("context_build_seconds") or 0 for item in items)}
        for prefix in ("retrieved", "context"):
            values = [item["payload"]["metrics"][f"{prefix}_evidence_recall"] for item in items
                      if item["payload"]["metrics"][f"{prefix}_evidence_recall"] is not None]
            all_values = [item["payload"]["metrics"][f"{prefix}_all_required_evidence"] for item in items
                          if item["payload"]["metrics"][f"{prefix}_all_required_evidence"] is not None]
            result[f"{prefix}_evidence_recall"] = sum(values) / len(values) if values else None
            result[f"{prefix}_all_required_rate"] = sum(all_values) / len(all_values) if all_values else None
            result[f"{prefix}_evaluable_count"] = len(values)
        return result
    _print({"stage": "retrieval", "key": journal.key, "output": str(args.output),
        "selected_questions": len(selected["questions"]), "cache_hits": hits,
        "new_results": new, "retrieval_seconds": seconds,
        "total_seconds": perf_counter() - command_started,
        "answer_llm_calls": 0, "kg_build_calls": 0,
        **journal.counts(len(selected["questions"])),
        "summary": {"overall": summarize(finished),
            "by_category": {str(cat): summarize([item for item in finished if item["payload"]["category"] == cat])
                            for cat in sorted({item["payload"]["category"] for item in finished})},
            "by_conversation": {sample: summarize([item for item in finished if item["payload"]["sample_id"] == sample])
                                for sample in sorted({item["payload"]["sample_id"] for item in finished})}}})


def command_answer(args: argparse.Namespace) -> None:
    command_started = perf_counter()
    selected = load_set(args.set)
    retrieval_manifest, retrievals = _open_journal(args.retrieval, "retrieval")
    if retrieval_manifest["set_hash"] != selected["content_hash"]:
        raise ValueError("retrieval and evaluation set differ")
    counter = _counter(args.tokenizer)
    if counter.name != retrieval_manifest["token_counter"]:
        raise ValueError("Reader tokenizer differs from retrieval context; rerun retrieval with the new tokenizer")
    if args.full_input_tokens is not None and not args.tokenizer:
        raise ValueError("full input limit requires a local tokenizer")
    reader_implementation = content_hash({
        "answer_generator": inspect.getsource(LLMAnswerGenerator),
        "caller": inspect.getsource(ValidatedJSONCaller),
        "transport": inspect.getsource(OpenAICompatibleTransport),
        "token_counter": _code_hash("tokens.py"),
    })
    reader_config = {"model": args.model, "endpoint_hash": hashlib.sha256(args.base_url.encode()).hexdigest(),
        "seed": args.seed, "temperature": 0, "prompt_version": LLMAnswerGenerator.PROMPT_VERSION,
        "implementation": reader_implementation,
        "tokenizer": counter.name, "full_input_tokens": args.full_input_tokens}
    manifest = {"schema_version": 1, "set_hash": selected["content_hash"],
        "retrieval_key": content_hash(retrieval_manifest),
        "retrieval_manifest": retrieval_manifest,
        "reader": reader_config}
    journal = StageJournal(args.output, stage="answer", manifest=manifest)
    completed = journal.completed()
    transport = OpenAICompatibleTransport(base_url=args.base_url, model=args.model,
        api_key=os.getenv(args.api_key_env) if args.api_key_env else None,
        timeout_seconds=args.timeout, seed=args.seed,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    generator = LLMAnswerGenerator(ValidatedJSONCaller(transport), token_counter=counter,
        full_input_token_budget=args.full_input_tokens)
    hits = new = 0
    seconds = 0.0
    for row in selected["questions"]:
        key = _identity(row)
        if key in completed:
            hits += 1
            continue
        source = retrievals.get(key)
        if source is None:
            continue
        start = perf_counter()
        try:
            retrieval = RetrievalResult.model_validate(source["payload"]["retrieval"])
            answer = generator.generate(retrieval)
            elapsed = perf_counter() - start
            journal.append(key, {"sample_id": row["sample_id"], "question_id": row["question_id"],
                "answer": answer.model_dump(mode="json")}, seconds=elapsed)
            new += 1
            seconds += elapsed
        except Exception as exc:
            journal.failure(key, exc)
    _print({"stage": "answer", "key": journal.key, "output": str(args.output),
        "selected_questions": len(selected["questions"]), "cache_hits": hits,
        "new_results": new, "unprocessed_retrievals": len(selected["questions"]) - len(retrievals),
        "answer_seconds": seconds, "llm_calls": max(transport.request_count,
            generator.call_count + generator.retry_count),
        "total_seconds": perf_counter() - command_started,
        "llm_retries": generator.retry_count,
        "llm_prompt_tokens": transport.prompt_tokens,
        "llm_completion_tokens": transport.completion_tokens,
        "kg_build_calls": 0, **journal.counts(len(selected["questions"]))})


def _score(row: dict[str, Any], answer: Answer, run_id: str) -> EvaluationRecord:
    trace = answer.retrieval.trace
    required = tuple(row["evidence"])
    has_required = bool(required)
    has_answer = row["answer"] is not None
    cited = answer.evidence_turn_ids if answer.citation_status == "valid" else ()
    cited_events = answer.evidence_event_ids if answer.citation_status == "valid" else ()
    evaluable_citation = answer.citation_status != "not_applicable"
    return EvaluationRecord(
        run_id=run_id, relation_mode="from-retrieval", question_id=row["question_id"],
        category=str(row["category"]), question=row["question"],
        expected_answer=str(row["answer"]) if has_answer else "",
        predicted_answer=answer.text,
        f1=token_f1(answer.text, str(row["answer"])) if has_answer else None,
        answer_annotation_available=has_answer,
        required_turn_ids=required,
        conversation_id=f"locomo-{row['sample_id']}",
        retrieved_evidence_turn_ids=trace.evidence_turn_ids,
        retrieved_evidence_event_ids=trace.evidence_event_ids,
        context_evidence_turn_ids=trace.context_turn_ids,
        context_evidence_event_ids=trace.context_event_ids,
        cited_evidence_turn_ids=cited, cited_evidence_event_ids=cited_events,
        retrieved_turn_refs=trace.retrieved_turn_refs,
        context_turn_refs=trace.context_turn_refs,
        cited_turn_refs=tuple(f"{item.conversation_id}:{item.turn_id}" for item in answer.retrieval.evidence
                              if set(item.event_ids) & set(cited_events)),
        retrieved_evidence_recall=evidence_recall(trace.evidence_turn_ids, required) if has_required else None,
        context_evidence_recall=evidence_recall(trace.context_turn_ids, required) if has_required else None,
        citation_evidence_recall=evidence_recall(cited, required) if has_required and evaluable_citation else None,
        retrieved_all_required_evidence=all_evidence_retrieved(trace.evidence_turn_ids, required) if has_required else None,
        context_all_required_evidence=all_evidence_retrieved(trace.context_turn_ids, required) if has_required else None,
        citation_all_required_evidence=all_evidence_retrieved(cited, required) if has_required and evaluable_citation else None,
        retrieved_episode_ids=trace.retrieved_episode_ids,
        traversed_edge_ids=trace.traversed_edge_ids,
        inspected_edge_ids=trace.inspected_edge_ids,
        context_state_ids=trace.context_state_ids,
        context_episode_ids=trace.context_episode_ids,
        context_relation_ids=trace.context_relation_ids,
        excluded_turn_ids=trace.excluded_turn_ids,
        excluded_state_ids=trace.excluded_state_ids,
        excluded_episode_ids=trace.excluded_episode_ids,
        excluded_relation_ids=trace.excluded_relation_ids,
        model_cited_event_ids=answer.model_cited_event_ids,
        invalid_cited_event_ids=answer.invalid_cited_event_ids,
        citation_status=answer.citation_status, response_status=answer.response_status,
        raw_response=answer.raw_response,
        memory_token_count=trace.memory_token_count,
        memory_token_method=trace.token_count_method,
        prompt_token_count=answer.prompt_token_count,
        prompt_token_method=answer.prompt_token_method,
        information_not_found=answer.response_status == "information_not_found",
        adversarial_abstention_correct=(answer.citation_status == "valid_abstention")
        if row["category"] == 5 else None,
    )


def _summary(records: list[dict[str, Any]], total: int) -> dict[str, Any]:
    def group(rows: list[dict[str, Any]]) -> dict[str, Any]:
        f1 = [row["f1"] for row in rows if row["f1"] is not None]
        result: dict[str, Any] = {"count": len(rows),
            "average_token_f1_success_only": sum(f1) / len(f1) if f1 else None,
            "token_f1_evaluable_count": len(f1)}
        abstentions = [row["adversarial_abstention_correct"] for row in rows
                       if row["adversarial_abstention_correct"] is not None]
        result["adversarial_abstention_rate"] = sum(abstentions) / len(abstentions) if abstentions else None
        result["adversarial_evaluable_count"] = len(abstentions)
        for prefix in ("retrieved", "context", "citation"):
            values = [row[f"{prefix}_evidence_recall"] for row in rows
                      if row[f"{prefix}_evidence_recall"] is not None]
            all_values = [row[f"{prefix}_all_required_evidence"] for row in rows
                          if row[f"{prefix}_all_required_evidence"] is not None]
            result[f"{prefix}_evidence_recall"] = sum(values) / len(values) if values else None
            result[f"{prefix}_all_required_rate"] = sum(all_values) / len(all_values) if all_values else None
            result[f"{prefix}_evaluable_count"] = len(values)
        return result
    by_category = {category: group([row for row in records if row["category"] == category])
                   for category in sorted({row["category"] for row in records})}
    by_conversation = {sample: group([row for row in records if row["conversation_id"] == sample])
                       for sample in sorted({row["conversation_id"] for row in records})}
    return {"target_questions": total, "successful_questions": len(records),
        "unprocessed_or_failed_questions": total - len(records),
        "overall": group(records), "by_category": by_category,
        "by_conversation": by_conversation}


def command_score(args: argparse.Namespace) -> None:
    command_started = perf_counter()
    selected = load_set(args.set)
    answer_manifest, answers = _open_journal(args.answers, "answer")
    if answer_manifest["set_hash"] != selected["content_hash"]:
        raise ValueError("answers and evaluation set differ")
    manifest = {"schema_version": 1, "set_hash": selected["content_hash"],
        "answer_key": content_hash(answer_manifest),
        "answer_manifest": answer_manifest,
        "evaluator": "relchain-token-f1-v1",
        "implementation": content_hash({
            "evaluation": _code_hash("evaluation.py"),
            "score": inspect.getsource(_score),
            "summary": inspect.getsource(_summary),
        })}
    journal = StageJournal(args.output, stage="score", manifest=manifest)
    completed = journal.completed()
    new = 0
    elapsed = 0.0
    for row in selected["questions"]:
        key = _identity(row)
        if key in completed or key not in answers:
            continue
        start = perf_counter()
        try:
            answer = Answer.model_validate(answers[key]["payload"]["answer"])
            record = _score(row, answer, journal.key)
            duration = perf_counter() - start
            journal.append(key, asdict(record), seconds=duration)
            new += 1
            elapsed += duration
        except Exception as exc:
            journal.failure(key, exc)
    all_records = [item["payload"] for item in journal.completed().values()]
    _print({"stage": "score", "key": journal.key, "output": str(args.output),
        "new_results": new, "cache_hits": len(completed), "scoring_seconds": elapsed,
        "total_seconds": perf_counter() - command_started,
        "answer_llm_calls": 0, "kg_build_calls": 0,
        **journal.counts(len(selected["questions"])),
        "summary": _summary(all_records, len(selected["questions"]))})


def command_compare(args: argparse.Namespace) -> None:
    left_manifest, left = _open_journal(args.left, "score")
    right_manifest, right = _open_journal(args.right, "score")
    if left_manifest["set_hash"] != right_manifest["set_hash"]:
        raise ValueError("cannot compare different question sets")
    paired = sorted(set(left) & set(right))
    changes = []
    for key in paired:
        a = left[key]["payload"]["f1"]
        b = right[key]["payload"]["f1"]
        if a is None or b is None:
            continue
        changes.append({"question_key": key,
            "category": left[key]["payload"]["category"],
            "conversation_id": left[key]["payload"]["conversation_id"],
            "delta_token_f1": b - a})
    def counts(items: list[dict[str, Any]]) -> dict[str, int]:
        return {"improved": sum(item["delta_token_f1"] > 0 for item in items),
            "worsened": sum(item["delta_token_f1"] < 0 for item in items),
            "tied": sum(item["delta_token_f1"] == 0 for item in items)}
    left_answer = left_manifest["answer_manifest"]
    right_answer = right_manifest["answer_manifest"]
    _print({"paired_questions": len(changes),
        "conditions": {
            "same_kg": left_answer["retrieval_manifest"]["kg"] == right_answer["retrieval_manifest"]["kg"],
            "same_search": left_answer["retrieval_manifest"]["search_config"] == right_answer["retrieval_manifest"]["search_config"],
            "same_reader": left_answer["reader"] == right_answer["reader"],
            "same_evaluator": left_manifest["evaluator"] == right_manifest["evaluator"],
        },
        "overall": counts(changes),
        "by_category": {cat: counts([item for item in changes if item["category"] == cat])
                        for cat in sorted({item["category"] for item in changes})},
        "by_conversation": {conv: counts([item for item in changes if item["conversation_id"] == conv])
                            for conv in sorted({item["conversation_id"] for item in changes})}})


def register_quick_commands(subparsers: Any) -> None:
    select = subparsers.add_parser("locomo-set", help="freeze a stratified short evaluation set")
    select.add_argument("--dataset", type=Path, required=True)
    select.add_argument("--profile", choices=("smoke", "dev60", "confirm120", "final"), required=True)
    select.add_argument("--samples", type=int, nargs="+")
    select.add_argument("--seed", type=int, default=0)
    select.add_argument("--output", type=Path, required=True)
    select.set_defaults(handler=command_set)

    common = subparsers.add_parser("locomo-build-common", help="build or reuse frozen full-history memory")
    common.add_argument("--dataset", type=Path, required=True)
    common.add_argument("--set", type=Path, required=True)
    common.add_argument("--artifact-dir", type=Path, required=True)
    common.add_argument("--rebuild", action="store_true")
    common.set_defaults(handler=command_build_common)

    mode = subparsers.add_parser("locomo-build-mode", help="build or resume one relation mode from frozen candidates")
    mode.add_argument("--set", type=Path, required=True)
    mode.add_argument("--artifact-dir", type=Path, required=True)
    mode.add_argument("--mode-db-dir", type=Path, required=True)
    mode.add_argument("--relation-mode", choices=[item.value for item in RelationMode], required=True)
    mode.add_argument("--relation-generation", choices=("deterministic", "llm"), default="deterministic")
    mode.add_argument("--relation-model")
    mode.add_argument("--relation-base-url")
    mode.add_argument("--api-key-env", default="RELCHAIN_LLM_API_KEY")
    mode.add_argument("--seed", type=int, default=0)
    mode.add_argument("--confidence-policy", choices=("common", "generation"), default="common")
    mode.add_argument("--timeout", type=float, default=180.0)
    mode.set_defaults(handler=command_build_mode)

    retrieve = subparsers.add_parser("locomo-retrieve", help="evaluate retrieval without an answer LLM")
    retrieve.add_argument("--set", type=Path, required=True)
    retrieve.add_argument("--artifact-dir", type=Path, required=True)
    retrieve.add_argument("--mode-db-dir", type=Path, required=True)
    retrieve.add_argument("--relation-mode", choices=[item.value for item in RelationMode], required=True)
    retrieve.add_argument("--output", type=Path, required=True)
    retrieve.add_argument("--tokenizer")
    retrieve.add_argument("--context-tokens", type=int, default=4096)
    retrieve.add_argument("--max-hops", type=int, default=2)
    retrieve.add_argument("--max-episodes", type=int, default=30)
    retrieve.add_argument("--episode-candidates", type=int, default=10)
    retrieve.add_argument("--state-candidates", type=int, default=5)
    retrieve.add_argument("--neighbors-per-node", type=int, default=5)
    retrieve.add_argument("--category-boost", type=float, default=1.0)
    retrieve.add_argument("--no-relation-context", action="store_true")
    retrieve.set_defaults(handler=command_retrieve)

    answer = subparsers.add_parser("locomo-answer", help="answer saved retrievals using a Reader")
    answer.add_argument("--set", type=Path, required=True)
    answer.add_argument("--retrieval", type=Path, required=True)
    answer.add_argument("--output", type=Path, required=True)
    answer.add_argument("--base-url", required=True)
    answer.add_argument("--model", required=True)
    answer.add_argument("--api-key-env", default="RELCHAIN_LLM_API_KEY")
    answer.add_argument("--tokenizer")
    answer.add_argument("--full-input-tokens", type=int)
    answer.add_argument("--seed", type=int, default=0)
    answer.add_argument("--timeout", type=float, default=180.0)
    answer.set_defaults(handler=command_answer)

    score = subparsers.add_parser("locomo-score", help="score saved answers without a model")
    score.add_argument("--set", type=Path, required=True)
    score.add_argument("--answers", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.set_defaults(handler=command_score)

    compare = subparsers.add_parser("locomo-compare", help="pair two scored runs on the same question set")
    compare.add_argument("--left", type=Path, required=True)
    compare.add_argument("--right", type=Path, required=True)
    compare.set_defaults(handler=command_compare)
