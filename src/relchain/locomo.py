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
    ) -> None:
        config = MemoryConfig(
            relation_mode=relation_mode,
            reader_name=reader_name,
            context_token_budget=context_token_budget,
        )
        self.system = MemorySystem(
            database,
            config,
            answer_generator=answer_generator,
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
    ) -> dict[str, Any]:
        logger = EvaluationLogger(output_path)
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
            expected = str(item.get("answer", ""))
            required = tuple(str(value) for value in item.get("evidence", ()))
            try:
                answer = self.system.query(question)
                retrieved = answer.evidence_turn_ids
                record = EvaluationRecord(
                    run_id=run_id,
                    relation_mode=self.system.config.relation_mode.value,
                    question_id=question_id,
                    category=str(item.get("category", "unknown")),
                    question=question,
                    expected_answer=expected,
                    predicted_answer=answer.text,
                    f1=token_f1(answer.text, expected),
                    required_turn_ids=required,
                    retrieved_turn_ids=retrieved,
                    evidence_recall=evidence_recall(retrieved, required),
                    all_evidence_retrieved=all_evidence_retrieved(retrieved, required),
                    retrieved_episode_ids=answer.retrieval.trace.retrieved_episode_ids,
                    traversed_edge_ids=answer.retrieval.trace.traversed_edge_ids,
                    information_not_found=answer.uncertainty == "information_not_found",
                )
                logger.append(record)
                added += 1
            except Exception as exc:
                failures.append({"question_id": question_id, "error": str(exc)})
        return {
            "run_id": run_id,
            "selected_questions": len(selected),
            "new_results": added,
            "failures": failures,
            "metrics": logger.aggregate(),
            "output": str(Path(output_path)),
        }
