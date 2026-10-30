from __future__ import annotations

import argparse
import json
import os
import subprocess
import hashlib
from pathlib import Path
from typing import Any

from .config import MemoryConfig, RelationMode
from .artifacts import build_mode_relations, content_hash, freeze_common, load_common
from .evaluation import relation_statistics
from .llm import LLMAnswerGenerator, LLMRelationTextGenerator, OpenAICompatibleTransport, ValidatedJSONCaller
from .locomo import LoCoMoRunner, load_locomo_sample
from .models import ConversationTurn
from .pipeline import MemorySystem
from .storage import SQLiteMemoryStore
from .tokens import ApproximateTokenCounter, TransformersTokenCounter


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _mode_for_database(path: Path, requested: str | None = None) -> RelationMode:
    if requested:
        return RelationMode(requested)
    mode = SQLiteMemoryStore(path).get_metadata("relation_mode")
    return RelationMode(mode or RelationMode.HYBRID.value)


def _system(args: argparse.Namespace) -> MemorySystem:
    mode = _mode_for_database(args.db, getattr(args, "relation_mode", None))
    return MemorySystem(args.db, MemoryConfig(relation_mode=mode))


def command_init(args: argparse.Namespace) -> None:
    system = _system(args)
    _print(
        {
            "database": str(system.store.path),
            "relation_mode": system.config.relation_mode,
        }
    )


def command_ingest(args: argparse.Namespace) -> None:
    system = _system(args)
    turns: list[ConversationTurn] = []
    with args.input.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                turns.append(ConversationTurn.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
    report = system.ingest(turns)
    _print(report.model_dump(mode="json"))


def command_query(args: argparse.Namespace) -> None:
    answer = _system(args).query(args.question)
    payload = (
        answer.model_dump(mode="json")
        if args.trace
        else {
            "answer": answer.text,
            "evidence_turn_ids": answer.evidence_turn_ids,
            "evidence_event_ids": answer.evidence_event_ids,
            "uncertainty": answer.uncertainty,
        }
    )
    _print(payload)


def command_inspect(args: argparse.Namespace) -> None:
    store = SQLiteMemoryStore(args.db)
    _print(
        {
            "database": str(store.path),
            "counts": store.counts(),
            "metadata": {
                key: store.get_metadata(key)
                for key in (
                    "relation_mode",
                    "writer_name",
                    "reader_name",
                    "embedding_name",
                )
            },
        }
    )


def command_export(args: argparse.Namespace) -> None:
    store = SQLiteMemoryStore(args.db)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(store.export_json(), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    _print({"output": str(args.output)})


def command_relation_stats(args: argparse.Namespace) -> None:
    store = SQLiteMemoryStore(args.db)
    _print(relation_statistics(store.list_relations()))


def command_locomo(args: argparse.Namespace) -> None:
    if args.full_input_tokens is not None and not args.tokenizer:
        raise ValueError("--full-input-tokens requires --tokenizer for an exact local chat template count")
    if args.relation_mode == "fixed" and args.relation_generation == "llm":
        raise ValueError("fixed mode does not generate free relation names; select deterministic")
    counter = TransformersTokenCounter(args.tokenizer) if args.tokenizer else ApproximateTokenCounter()
    api_key = os.getenv(args.api_key_env) if args.api_key_env else None
    transport = OpenAICompatibleTransport(
        base_url=args.base_url,
        model=args.model,
        api_key=api_key,
        timeout_seconds=args.timeout,
        seed=args.seed,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    relation_generator = None
    if args.relation_generation == "llm" and args.relation_mode != "fixed":
        relation_transport = OpenAICompatibleTransport(
            base_url=args.relation_base_url or args.base_url,
            model=args.relation_model or args.model,
            api_key=os.getenv(args.relation_api_key_env) if args.relation_api_key_env else api_key,
            timeout_seconds=args.timeout,
            seed=args.seed,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        relation_generator = LLMRelationTextGenerator(ValidatedJSONCaller(relation_transport))
    runner = LoCoMoRunner(
        database=args.db,
        relation_mode=RelationMode(args.relation_mode),
        reader_name=args.model,
        answer_generator=LLMAnswerGenerator(
            ValidatedJSONCaller(transport), token_counter=counter,
            full_input_token_budget=args.full_input_tokens,
        ),
        context_token_budget=args.context_tokens,
        relation_text_generator=relation_generator,
        token_counter=counter,
        full_input_token_budget=args.full_input_tokens,
        max_hops=args.max_hops,
        include_relation_context=not args.no_relation_context,
        category_boost=args.category_boost,
        confidence_policy=args.confidence_policy,
    )
    turns, questions, sample_id = load_locomo_sample(args.dataset, args.sample)
    if args.common_artifact:
        if not args.common_artifact.exists():
            common_db = args.common_artifact.with_suffix(".common.db")
            common = LoCoMoRunner(
                database=common_db, relation_mode=RelationMode.FIXED,
                reader_name="common-builder", answer_generator=runner.system.answer_generator,
            )
            common_report = common.prepare(turns)
            if common_report.failures:
                raise ValueError(f"common memory construction failed for {len(common_report.failures)} items")
            freeze_common(common.system, args.common_artifact)
        artifact = load_common(args.common_artifact)
        expected_turn_hash = content_hash(sorted(
            [item.model_dump(mode="json") for item in turns],
            key=lambda row: (row["conversation_id"], row["turn_id"]),
        ))
        artifact_turn_hash = content_hash(artifact["common"]["turns"])
        if expected_turn_hash != artifact_turn_hash:
            raise ValueError("dataset turns differ from the frozen common artifact")
        generation_signature = content_hash({
            "method": args.relation_generation,
            "model": args.relation_model or args.model if args.relation_generation == "llm" else None,
            "endpoint_hash": hashlib.sha256((args.relation_base_url or args.base_url).encode()).hexdigest() if args.relation_generation == "llm" else None,
            "seed": args.seed,
            "prompt_version": LLMRelationTextGenerator.PROMPT_VERSION if args.relation_generation == "llm" else "deterministic-v1",
        })
        relation_hash = build_mode_relations(runner.system, artifact, generation_signature=generation_signature)
        report = None
    else:
        report = runner.prepare(turns)
        if report.failures:
            raise ValueError(f"memory construction failed for {len(report.failures)} items")
        relation_hash = content_hash([item.model_dump(mode="json") for item in runner.system.store.list_relations()])
    if args.prepare_only:
        _print(
            {
                "sample_id": sample_id,
                "turns": len(turns),
                "questions": len(questions),
                "ingest": {
                    "new_turns": len(report.turn_ids) if report else 0,
                    "new_events": len(report.event_ids) if report else 0,
                    "affected_episodes": len(report.episode_ids) if report else 0,
                    "relations": runner.system.store.counts()["relations"],
                    "state_updates": len(report.transitions) if report else 0,
                    "failures": report.failures if report else (),
                },
            }
        )
        return
    def git_bytes(*command: str) -> bytes | None:
        result = subprocess.run(["git", *command], capture_output=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    def git_value(*command: str) -> str:
        raw = git_bytes(*command)
        return raw.decode("utf-8", errors="replace") if raw is not None else "unknown"

    git_status = git_value("status", "--porcelain", "--untracked-files=no")
    git_diff = git_bytes("diff", "HEAD", "--binary")
    manifest = {
        "schema_version": 2,
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty": bool(git_status),
        "git_diff_hash": hashlib.sha256(git_diff).hexdigest() if git_status and git_diff is not None else None,
        "dataset_hash": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "sample_id": sample_id,
        "question_set_hash": content_hash(questions),
        "common_artifact_hash": artifact["content_hash"] if args.common_artifact else "unfrozen",
        "node_hash": artifact["node_hash"] if args.common_artifact else content_hash(runner.system.store.export_json()["episodes"]),
        "candidate_hash": artifact["candidate_hash"] if args.common_artifact else "unfrozen",
        "relation_hash": relation_hash,
        "relation_mode": args.relation_mode,
        "relation_generation": args.relation_generation,
        "reader_endpoint_hash": hashlib.sha256(args.base_url.encode()).hexdigest(),
        "relation_endpoint_hash": hashlib.sha256((args.relation_base_url or args.base_url).encode()).hexdigest() if args.relation_generation == "llm" else None,
        "relation_model": args.relation_model or args.model if args.relation_generation == "llm" else "deterministic-v1",
        "reader_model": args.model,
        "model_revision": "unknown", "quantization": "unknown",
        "temperature": 0, "seed": args.seed,
        "generation_parameters": {"max_tokens": 512, "response_format": "json_object"},
        "answer_prompt_version": LLMAnswerGenerator.PROMPT_VERSION,
        "relation_prompt_version": LLMRelationTextGenerator.PROMPT_VERSION if args.relation_generation == "llm" else "deterministic-v1",
        "search": runner.system.config.model_dump(mode="json"),
        "token_counter": counter.name,
        "token_count_exact": counter.exact,
        "event_extractor": type(runner.system.event_extractor).__name__,
        "episode_annotator": type(runner.system.episode_builder.annotator).__name__,
        "state_extractor": type(runner.system.state_extractor).__name__,
        "relation_generator": type(runner.system.relation_builder.text_generator).__name__,
        "embedding": runner.system.embedder.name,
        "reader": type(runner.system.answer_generator).__name__,
        "evaluator": "relchain-token-f1-v1",
    }
    run_id = f"locomo-{sample_id}-{args.relation_mode}-{content_hash(manifest)[:16]}"
    result = runner.answer_questions(
        questions=questions,
        output_path=args.output,
        run_id=run_id,
        max_questions=args.max_questions,
        manifest=manifest,
    )
    result["sample_id"] = sample_id
    result["turn_count"] = len(turns)
    result["question_count"] = len(questions)
    _print(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="relchain")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="initialize a mode-isolated database")
    init.add_argument("--db", type=Path, required=True)
    init.add_argument(
        "--relation-mode",
        choices=[item.value for item in RelationMode],
        default="hybrid",
    )
    init.set_defaults(handler=command_init)

    ingest = subparsers.add_parser("ingest", help="ingest a JSONL conversation")
    ingest.add_argument("--db", type=Path, required=True)
    ingest.add_argument("--input", type=Path, required=True)
    ingest.add_argument(
        "--relation-mode", choices=[item.value for item in RelationMode]
    )
    ingest.set_defaults(handler=command_ingest)

    query = subparsers.add_parser(
        "query", help="retrieve evidence and answer a question"
    )
    query.add_argument("--db", type=Path, required=True)
    query.add_argument("--question", required=True)
    query.add_argument("--trace", action="store_true")
    query.set_defaults(handler=command_query)

    inspect = subparsers.add_parser("inspect", help="show database counts and versions")
    inspect.add_argument("--db", type=Path, required=True)
    inspect.set_defaults(handler=command_inspect)

    export = subparsers.add_parser(
        "export", help="export latest graph and state records"
    )
    export.add_argument("--db", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.set_defaults(handler=command_export)

    stats = subparsers.add_parser(
        "relation-stats", help="summarize relation vocabulary"
    )
    stats.add_argument("--db", type=Path, required=True)
    stats.set_defaults(handler=command_relation_stats)

    locomo = subparsers.add_parser(
        "locomo",
        help="run one LoCoMo conversation with an OpenAI-compatible Qwen reader",
    )
    locomo.add_argument("--dataset", type=Path, required=True)
    locomo.add_argument("--sample", type=int, default=0)
    locomo.add_argument("--db", type=Path, required=True)
    locomo.add_argument("--output", type=Path, required=True)
    locomo.add_argument(
        "--relation-mode",
        choices=[item.value for item in RelationMode],
        default="hybrid",
    )
    locomo.add_argument("--base-url", required=True)
    locomo.add_argument("--model", default="Qwen/Qwen3-8B")
    locomo.add_argument("--api-key-env", default="RELCHAIN_LLM_API_KEY")
    locomo.add_argument("--max-questions", type=int)
    locomo.add_argument("--context-tokens", type=int, default=4096)
    locomo.add_argument("--timeout", type=float, default=180.0)
    locomo.add_argument("--seed", type=int, default=0)
    locomo.add_argument("--relation-generation", choices=("deterministic", "llm"), default="deterministic")
    locomo.add_argument("--relation-model")
    locomo.add_argument("--relation-base-url")
    locomo.add_argument("--relation-api-key-env")
    locomo.add_argument("--common-artifact", type=Path)
    locomo.add_argument("--tokenizer", help="local Transformers tokenizer path for exact counting")
    locomo.add_argument("--full-input-tokens", type=int)
    locomo.add_argument("--max-hops", type=int, default=2)
    locomo.add_argument("--category-boost", type=float, default=1.0)
    locomo.add_argument("--confidence-policy", choices=("common", "generation"), default="common")
    locomo.add_argument("--no-relation-context", action="store_true")
    locomo.add_argument("--prepare-only", action="store_true")
    locomo.set_defaults(handler=command_locomo)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
