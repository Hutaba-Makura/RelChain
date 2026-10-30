from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .config import MemoryConfig, RelationMode
from .evaluation import (
    EvaluationLogger,
    EvaluationRecord,
    all_evidence_retrieved,
    evidence_recall,
    token_f1,
)
from .models import ConversationTurn, IngestReport
from .pipeline import MemorySystem
from .ports import RelationTextGenerator
from .tokens import TokenCounter


def _session_datetime(value: str | None, fallback_index: int) -> datetime:
    if value:
        normalized = value.strip()
        for pattern in ("%I:%M %p on %d %B, %Y", "%I:%M%p on %d %B, %Y"):
            try:
                return datetime.strptime(normalized, pattern).replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                continue
    return datetime(2000, 1, 1, tzinfo=timezone.utc) + timedelta(days=fallback_index)


def load_locomo_sample(
    dataset_path: str | Path, sample_index: int
) -> tuple[list[ConversationTurn], list[dict[str, Any]], str]:
    payload = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        payload = payload.get("samples", [])
    if sample_index < 0 or sample_index >= len(payload):
        raise IndexError(
            f"LoCoMo sample index {sample_index} is outside 0..{len(payload)-1}"
        )
    sample = payload[sample_index]
    sample_id = str(sample.get("sample_id", sample_index))
    conversation = sample["conversation"]
    turns: list[ConversationTurn] = []
    session_keys = sorted(
        (
            key
            for key in conversation
            if key.startswith("session_") and not key.endswith("_date_time")
        ),
        key=lambda key: int(key.split("_")[1]),
    )
    for session_position, session_key in enumerate(session_keys, start=1):
        observed_at = _session_datetime(
            conversation.get(f"{session_key}_date_time"), session_position
        )
        for turn_position, raw in enumerate(conversation[session_key]):
            text = str(raw.get("text", "")).strip()
            caption = str(raw.get("blip_caption", "")).strip()
            if caption:
                text = f"{text}\n[Image description: {caption}]"
            if not text:
                continue
            turn_id = str(
                raw.get("dia_id") or f"S{session_position}:{turn_position + 1}"
            )
            turns.append(
                ConversationTurn(
                    conversation_id=f"locomo-{sample_id}",
                    turn_id=turn_id,
                    speaker_id=str(raw.get("speaker", "unknown")),
                    text=text,
                    observed_at=observed_at + timedelta(seconds=turn_position),
                    metadata={
                        "dataset": "locomo",
                        "sample_id": sample_id,
                        "session": session_key,
                    },
                )
            )
    questions = [dict(item) for item in sample.get("qa", sample.get("questions", []))]
    return turns, questions, sample_id


class LoCoMoRunner:
    def __init__(
        self,
        *,
        database: str | Path,
        relation_mode: RelationMode,
        reader_name: str,
        answer_generator: Any,
        context_token_budget: int = 4096,
        relation_text_generator: RelationTextGenerator | None = None,
        token_counter: TokenCounter | None = None,
        full_input_token_budget: int | None = None,
        max_hops: int = 2,
        include_relation_context: bool = True,
        category_boost: float = 1.0,
        confidence_policy: str = "common",
    ) -> None:
        config = MemoryConfig(
            relation_mode=relation_mode,
            reader_name=reader_name,
            context_token_budget=context_token_budget,
            full_input_token_budget=full_input_token_budget,
            max_hops=max_hops,
            include_relation_context=include_relation_context,
            category_boost=category_boost,
            confidence_policy=confidence_policy,
        )
        self.system = MemorySystem(
            database,
            config,
            answer_generator=answer_generator,
            relation_text_generator=relation_text_generator,
            token_counter=token_counter,
        )

    def prepare(self, turns: list[ConversationTurn]) -> IngestReport:
        return self.system.ingest(turns)

    def answer_questions(
        self,
        *,
        questions: list[dict[str, Any]],
        output_path: str | Path,
        run_id: str,
        max_questions: int | None = None,
        manifest: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        logger = EvaluationLogger(output_path)
        if manifest is not None:
            logger.ensure_manifest(run_id, manifest)
        conversations = {row["conversation_id"] for row in self.system.store._raw_models("turns")}
        conversation_id = next(iter(conversations)) if len(conversations) == 1 else ""
        completed = {
            record.question_id for record in logger.load() if record.run_id == run_id
        }
        selected = questions if max_questions is None else questions[:max_questions]
        failures: list[dict[str, str]] = []
        added = 0
        for index, item in enumerate(selected):
            question_id = str(item.get("question_id", index))
            if question_id in completed:
                continue
            question = str(item["question"])
            has_answer = "answer" in item and item["answer"] is not None
            expected = str(item["answer"]) if has_answer else ""
            required = tuple(str(value) for value in item.get("evidence", ()))
            has_required = "evidence" in item and bool(required)
            stage = "retrieval"
            try:
                retrieval = self.system.retriever.retrieve(question)
                stage = "answer_generation"
                answer = self.system.answer_generator.generate(retrieval)
                stage = "evaluation"
                trace = answer.retrieval.trace
                retrieved = trace.evidence_turn_ids
                context = trace.context_turn_ids
                citation_evaluable = answer.citation_status != "not_applicable"
                cited = answer.evidence_turn_ids if answer.citation_status == "valid" else ()
                cited_events = answer.evidence_event_ids if answer.citation_status == "valid" else ()
                record = EvaluationRecord(
                    run_id=run_id,
                    relation_mode=self.system.config.relation_mode.value,
                    question_id=question_id,
                    category=str(item.get("category", "unknown")),
                    question=question,
                    expected_answer=expected,
                    predicted_answer=answer.text,
                    f1=token_f1(answer.text, expected) if has_answer else None,
                    answer_annotation_available=has_answer,
                    required_turn_ids=required,
                    conversation_id=conversation_id,
                    retrieved_evidence_turn_ids=retrieved,
                    retrieved_evidence_event_ids=trace.evidence_event_ids,
                    context_evidence_turn_ids=context,
                    context_evidence_event_ids=trace.context_event_ids,
                    cited_evidence_turn_ids=cited,
                    retrieved_turn_refs=trace.retrieved_turn_refs,
                    context_turn_refs=trace.context_turn_refs,
                    cited_turn_refs=tuple(f"{e.conversation_id}:{e.turn_id}" for e in answer.retrieval.evidence if set(e.event_ids) & set(cited_events)),
                    cited_evidence_event_ids=cited_events,
                    retrieved_evidence_recall=evidence_recall(retrieved, required) if has_required else None,
                    context_evidence_recall=evidence_recall(context, required) if has_required else None,
                    citation_evidence_recall=evidence_recall(cited, required) if has_required and citation_evaluable else None,
                    retrieved_all_required_evidence=all_evidence_retrieved(retrieved, required) if has_required else None,
                    context_all_required_evidence=all_evidence_retrieved(context, required) if has_required else None,
                    citation_all_required_evidence=all_evidence_retrieved(cited, required) if has_required and citation_evaluable else None,
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
                    citation_status=answer.citation_status,
                    response_status=answer.response_status,
                    raw_response=answer.raw_response,
                    memory_token_count=trace.memory_token_count,
                    memory_token_method=trace.token_count_method,
                    prompt_token_count=answer.prompt_token_count,
                    prompt_token_method=answer.prompt_token_method,
                    information_not_found=answer.response_status == "information_not_found" or answer.uncertainty == "information_not_found",
                )
                logger.append(record)
                added += 1
            except Exception as exc:
                logger.append_failure(run_id=run_id, question_id=question_id, stage=stage, exception_type=type(exc).__name__)
                failures.append({"question_id": question_id, "stage": stage, "exception_type": type(exc).__name__})
        return {
            "run_id": run_id,
            "selected_questions": len(selected),
            "new_results": added,
            "failures": failures,
            "metrics": logger.aggregate(run_id, total_questions=len(questions)),
            "output": str(Path(output_path)),
        }
