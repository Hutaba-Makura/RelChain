from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
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
    f1: float
    required_turn_ids: tuple[str, ...]
    retrieved_turn_ids: tuple[str, ...]
    evidence_recall: float | None
    all_evidence_retrieved: bool | None
    retrieved_episode_ids: tuple[str, ...]
    traversed_edge_ids: tuple[str, ...]
    information_not_found: bool


class EvaluationLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: EvaluationRecord) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def load(self) -> list[EvaluationRecord]:
        if not self.path.exists():
            return []
        records: list[EvaluationRecord] = []
        with self.path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                data = json.loads(line)
                data["required_turn_ids"] = tuple(data["required_turn_ids"])
                data["retrieved_turn_ids"] = tuple(data["retrieved_turn_ids"])
                data["retrieved_episode_ids"] = tuple(data["retrieved_episode_ids"])
                data["traversed_edge_ids"] = tuple(data["traversed_edge_ids"])
                records.append(EvaluationRecord(**data))
        return records

    def aggregate(self) -> dict[str, object]:
        records = self.load()
        if not records:
            return {"count": 0}

        def summarize(items: Sequence[EvaluationRecord]) -> dict[str, float | int]:
            recalls = [
                item.evidence_recall
                for item in items
                if item.evidence_recall is not None
            ]
            return {
                "count": len(items),
                "average_f1": sum(item.f1 for item in items) / len(items),
                "average_evidence_recall": (
                    sum(recalls) / len(recalls) if recalls else 0.0
                ),
                "all_evidence_rate": sum(
                    item.all_evidence_retrieved is True for item in items
                )
                / len(items),
                "information_not_found_rate": sum(
                    item.information_not_found for item in items
                )
                / len(items),
            }

        categories = sorted({item.category for item in records})
        return {
            "overall": summarize(records),
            "by_category": {
                category: summarize(
                    [item for item in records if item.category == category]
                )
                for category in categories
            },
        }
