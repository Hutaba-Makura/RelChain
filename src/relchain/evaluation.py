from __future__ import annotations

import json
import re
import hashlib
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .models import Relation


def _answer_tokens(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    return re.findall(r"[a-z0-9_]+|[\u3040-\u30ff\u3400-\u9fff]", normalized)


def token_f1(predicted: str, expected: str) -> float:
    prediction = Counter(_answer_tokens(predicted))
    reference = Counter(_answer_tokens(expected))
    overlap = sum((prediction & reference).values())
    if not prediction or not reference:
        return float(prediction == reference)
    if overlap == 0:
        return 0.0
    precision = overlap / sum(prediction.values())
    recall = overlap / sum(reference.values())
    return 2 * precision * recall / (precision + recall)


def evidence_recall(
    retrieved_turn_ids: Sequence[str], required_turn_ids: Sequence[str]
) -> float | None:
    required = set(required_turn_ids)
    if not required:
        return None
    return len(required & set(retrieved_turn_ids)) / len(required)


def all_evidence_retrieved(
    retrieved_turn_ids: Sequence[str], required_turn_ids: Sequence[str]
) -> bool | None:
    required = set(required_turn_ids)
    if not required:
        return None
    return required.issubset(retrieved_turn_ids)


def relation_statistics(relations: Iterable[Relation]) -> dict[str, object]:
    items = list(relations)
    modes = Counter(item.mode for item in items)
    categories = Counter(
        item.category.value if item.category else "free" for item in items
    )
    relation_names = Counter(item.relation_text for item in items)
    return {
        "total_edges": len(items),
        "by_mode": dict(sorted(modes.items())),
        "by_category": dict(sorted(categories.items())),
        "unique_relation_texts": len(relation_names),
        "single_use_relation_texts": sum(
            count == 1 for count in relation_names.values()
        ),
        "relation_frequencies": dict(relation_names.most_common()),
    }


@dataclass(frozen=True)
class EvaluationRecord:
    run_id: str
    relation_mode: str
    question_id: str
    category: str
    question: str
    expected_answer: str
    predicted_answer: str
    f1: float | None
    answer_annotation_available: bool = True
    required_turn_ids: tuple[str, ...] = ()
    retrieved_turn_ids: tuple[str, ...] = ()  # legacy field: old files stored citations here
    evidence_recall: float | None = None  # legacy metric
    all_evidence_retrieved: bool | None = None  # legacy metric
    retrieved_episode_ids: tuple[str, ...] = ()
    traversed_edge_ids: tuple[str, ...] = ()
    information_not_found: bool = False
    schema_version: int = 2
    conversation_id: str = ""
    retrieved_evidence_turn_ids: tuple[str, ...] = ()
    retrieved_evidence_event_ids: tuple[str, ...] = ()
    context_evidence_turn_ids: tuple[str, ...] = ()
    context_evidence_event_ids: tuple[str, ...] = ()
    cited_evidence_turn_ids: tuple[str, ...] = ()
    cited_evidence_event_ids: tuple[str, ...] = ()
    retrieved_evidence_recall: float | None = None
    context_evidence_recall: float | None = None
    citation_evidence_recall: float | None = None
    retrieved_all_required_evidence: bool | None = None
    context_all_required_evidence: bool | None = None
    citation_all_required_evidence: bool | None = None
    model_cited_event_ids: tuple[str, ...] = ()
    invalid_cited_event_ids: tuple[str, ...] = ()
    citation_status: str = "unknown"
    response_status: str = "unknown"
    raw_response: str | None = None
    memory_token_count: int | None = None
    memory_token_method: str | None = None
    prompt_token_count: int | None = None
    prompt_token_method: str | None = None
    adversarial_abstention_correct: bool | None = None
    context_state_ids: tuple[str, ...] = ()
    context_episode_ids: tuple[str, ...] = ()
    context_relation_ids: tuple[str, ...] = ()
    excluded_turn_ids: tuple[str, ...] = ()
    excluded_state_ids: tuple[str, ...] = ()
    excluded_episode_ids: tuple[str, ...] = ()
    excluded_relation_ids: tuple[str, ...] = ()
    inspected_edge_ids: tuple[str, ...] = ()
    retrieved_turn_refs: tuple[str, ...] = ()
    context_turn_refs: tuple[str, ...] = ()
    cited_turn_refs: tuple[str, ...] = ()


class EvaluationLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: EvaluationRecord) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def ensure_manifest(self, run_id: str, manifest: dict[str, object]) -> None:
        serialized = json.dumps(manifest, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        existing = [row for row in self._rows() if row.get("run_id") == run_id]
        prior = [row for row in existing if row.get("record_type") == "manifest"]
        if prior:
            if any(row.get("manifest_hash") != digest for row in prior):
                raise ValueError("run manifest mismatch; use a new run ID")
            return
        if existing:
            raise ValueError("run has legacy results without a manifest; use a new run ID")
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "record_type": "manifest", "schema_version": 2,
                "run_id": run_id, "manifest_hash": digest, "manifest": manifest,
            }, ensure_ascii=False) + "\n")

    def append_failure(self, *, run_id: str, question_id: str, stage: str, exception_type: str) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "schema_version": 2, "record_type": "failure", "run_id": run_id,
                "question_id": question_id, "stage": stage,
                "exception_type": exception_type,
            }, ensure_ascii=False) + "\n")

    def _rows(self) -> list[dict[str, object]]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]

    def load(self) -> list[EvaluationRecord]:
        records: list[EvaluationRecord] = []
        allowed = {item.name for item in fields(EvaluationRecord)}
        for row in self._rows():
            if row.get("record_type") in {"failure", "manifest"}:
                continue
            data = {key: value for key, value in row.items() if key in allowed}
            if "schema_version" not in data:
                data["schema_version"] = 1
                # The old retrieved_turn_ids and evidence_recall are citation based.
                # No new retrieval/context metrics can be recovered from them.
            for key, value in list(data.items()):
                if key.endswith("_ids") and isinstance(value, list):
                    data[key] = tuple(value)
            records.append(EvaluationRecord(**data))
        return records

    def aggregate(self, run_id: str, *, total_questions: int | None = None) -> dict[str, object]:
        records = {record.question_id: record for record in self.load() if record.run_id == run_id}
        rows = [row for row in self._rows() if row.get("run_id") == run_id]
        failures = {str(row["question_id"]) for row in rows if row.get("record_type") == "failure"} - records.keys()
        items = list(records.values())
        total = total_questions if total_questions is not None else len(set(records) | failures)

        def summarize(group: Sequence[EvaluationRecord]) -> dict[str, object]:
            f1_values = [item.f1 for item in group if item.f1 is not None]
            result: dict[str, object] = {
                "count": len(group),
                "average_token_f1_success_only": sum(f1_values) / len(f1_values) if f1_values else None,
                "token_f1_evaluable_count": len(f1_values),
                "information_not_found_rate": sum(item.information_not_found for item in group) / len(group) if group else None,
            }
            abstentions = [item.adversarial_abstention_correct for item in group
                           if item.adversarial_abstention_correct is not None]
            result["adversarial_abstention_rate"] = sum(abstentions) / len(abstentions) if abstentions else None
            result["adversarial_evaluable_count"] = len(abstentions)
            for prefix in ("retrieved", "context", "citation"):
                values = [getattr(item, f"{prefix}_evidence_recall") for item in group]
                valid = [value for value in values if value is not None]
                all_values = [getattr(item, f"{prefix}_all_required_evidence") for item in group]
                all_valid = [value for value in all_values if value is not None]
                result[f"{prefix}_evidence_recall"] = sum(valid) / len(valid) if valid else None
                result[f"{prefix}_all_required_rate"] = sum(all_valid) / len(all_valid) if all_valid else None
                result[f"{prefix}_evaluable_count"] = len(valid)
            return result

        categories = sorted({item.category for item in items})
        return {
            "run_id": run_id,
            "total_questions": total,
            "successful_questions": len(items),
            "failed_questions": len(failures),
            "unprocessed_questions": max(total - len(items) - len(failures), 0),
            "overall": summarize(items),
            "by_category": {
                category: summarize(
                    [item for item in items if item.category == category]
                )
                for category in categories
            },
        }
