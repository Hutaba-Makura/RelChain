from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .artifacts import content_hash
from .text import stable_id


# Released locomo10.json IDs, not the numbered prose list in the paper.
CATEGORIES = {1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop", 5: "adversarial"}
QUOTAS = {
    "smoke": {4: 3, 1: 3, 2: 2, 5: 2},
    "dev60": {4: 15, 1: 20, 2: 15, 5: 10},
    "confirm120": {4: 30, 1: 40, 2: 30, 5: 20},
}
DEFAULT_SAMPLES = {"smoke": (0,), "dev60": (0, 1), "confirm120": (2, 3, 4)}
SELECTION_VERSION = "stratified-balanced-v1"


def dataset_samples(path: str | Path) -> tuple[list[dict[str, Any]], str]:
    raw = Path(path).read_bytes()
    parsed = json.loads(raw)
    samples = parsed if isinstance(parsed, list) else parsed["samples"]
    return samples, hashlib.sha256(raw).hexdigest()


def _rank(seed: int, dataset_hash: str, sample_id: str, question_id: str) -> str:
    return content_hash((SELECTION_VERSION, seed, dataset_hash, sample_id, question_id))


def _question_rows(samples: list[dict[str, Any]], sample_indices: tuple[int, ...],
                   dataset_hash: str, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_index in sample_indices:
        if sample_index < 0 or sample_index >= len(samples):
            raise IndexError(f"sample index outside dataset: {sample_index}")
        sample = samples[sample_index]
        sample_id = str(sample.get("sample_id", sample_index))
        turn_session = {
            str(turn.get("dia_id")): session_key
            for session_key, turns in sample["conversation"].items()
            if session_key.startswith("session_") and not session_key.endswith("_date_time")
            for turn in turns
        }
        for original_index, raw in enumerate(sample.get("qa", sample.get("questions", []))):
            category = int(raw.get("category", -1))
            if category not in CATEGORIES:
                continue
            source_question_id = raw.get("question_id")
            question_id = str(source_question_id) if source_question_id is not None else stable_id(
                "locomo-q", sample_id, str(original_index)
            )
            source_evidence = raw.get("evidence", ()) or ()
            if isinstance(source_evidence, str):
                source_evidence = (source_evidence,)
            evidence = tuple(str(item) for item in source_evidence)
            sessions = {turn_session[item] for item in evidence if item in turn_session}
            rows.append({
                "sample_index": sample_index, "sample_id": sample_id,
                "original_question_index": original_index, "question_id": question_id,
                "question": str(raw["question"]), "answer": raw.get("answer"),
                "evidence": evidence, "evidence_annotated": bool(evidence),
                "evidence_session_count": len(sessions),
                "category": category, "category_name": CATEGORIES[category],
                "selection_rank": _rank(seed, dataset_hash, sample_id, question_id),
            })
    keys = [(row["sample_id"], row["question_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate source question IDs")
    return rows


def _balanced_take(rows: list[dict[str, Any]], count: int,
                   sample_indices: tuple[int, ...]) -> list[dict[str, Any]]:
    buckets: dict[int, list[dict[str, Any]]] = {}
    for sample_index in sample_indices:
        candidates = [row for row in rows if row["sample_index"] == sample_index]
        buckets[sample_index] = sorted(
            candidates,
            key=lambda row: (
                -int(row["category"] == 1 and row["evidence_session_count"] > 1),
                row["selection_rank"], row["original_question_index"],
            ),
        )
    selected: list[dict[str, Any]] = []
    while len(selected) < count and any(buckets.values()):
        for sample_index in sample_indices:
            if buckets[sample_index] and len(selected) < count:
                selected.append(buckets[sample_index].pop(0))
    return selected


def select_set(dataset: str | Path, *, profile: str, seed: int = 0,
               sample_indices: tuple[int, ...] | None = None) -> dict[str, Any]:
    samples, dataset_hash = dataset_samples(dataset)
    if profile not in (*QUOTAS, "final"):
        raise ValueError(f"unsupported profile: {profile}")
    if profile == "final" and sample_indices is None:
        raise ValueError("final profile requires explicit held-out sample indices")
    selected_samples = sample_indices if sample_indices is not None else DEFAULT_SAMPLES[profile]
    if profile == "final" and set(selected_samples) & {index for group in DEFAULT_SAMPLES.values() for index in group}:
        raise ValueError("final samples must be outside default development and confirmation samples")
    if not selected_samples or len(set(selected_samples)) != len(selected_samples):
        raise ValueError("sample indices must be nonempty and unique")
    rows = _question_rows(samples, selected_samples, dataset_hash, seed)
    if profile == "final":
        chosen = sorted(rows, key=lambda row: (row["sample_index"], row["original_question_index"]))
        shortage: dict[str, int] = {}
        redistributed = 0
    else:
        quotas = QUOTAS[profile]
        chosen = []
        shortage = {}
        for category, requested in quotas.items():
            taken = _balanced_take([row for row in rows if row["category"] == category],
                                  requested, selected_samples)
            chosen.extend(taken)
            shortage[CATEGORIES[category]] = requested - len(taken)
        target = sum(quotas.values())
        selected_keys = {(row["sample_id"], row["question_id"]) for row in chosen}
        remaining = [row for row in rows if row["category"] in quotas
                     and (row["sample_id"], row["question_id"]) not in selected_keys]
        supplement = _balanced_take(remaining, target - len(chosen), selected_samples)
        chosen.extend(supplement)
        redistributed = len(supplement)
    chosen.sort(key=lambda row: (row["sample_index"], row["original_question_index"]))
    counts = Counter(row["category_name"] for row in chosen)
    by_sample = {
        str(index): dict(Counter(row["category_name"] for row in chosen if row["sample_index"] == index))
        for index in selected_samples
    }
    body = {
        "schema_version": 1, "profile": profile, "selection_version": SELECTION_VERSION,
        "seed": seed, "dataset_hash": dataset_hash,
        "sample_indices": selected_samples,
        "sample_ids": [str(samples[index].get("sample_id", index)) for index in selected_samples],
        "target_by_category": {CATEGORIES[key]: value for key, value in QUOTAS.get(profile, {}).items()},
        "shortage_by_category": shortage, "redistributed_count": redistributed,
        "actual_by_category": dict(counts), "actual_by_sample": by_sample,
        "selected_count": len(chosen),
        "cross_session_multi_hop_count": sum(
            row["category"] == 1 and row["evidence_session_count"] > 1 for row in chosen
        ),
        "questions": chosen,
    }
    body["content_hash"] = content_hash(body)
    return body


def save_set(body: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    if target.exists():
        existing = load_set(target)
        if existing["content_hash"] != body["content_hash"]:
            raise ValueError("evaluation set path contains a different selection; use a new path")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(body, stream, ensure_ascii=False, indent=2)


def load_set(path: str | Path, dataset: str | Path | None = None) -> dict[str, Any]:
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    digest = body.get("content_hash")
    if body.get("schema_version") != 1 or content_hash({k: v for k, v in body.items() if k != "content_hash"}) != digest:
        raise ValueError("evaluation set is incomplete or modified")
    if dataset is not None and dataset_samples(dataset)[1] != body["dataset_hash"]:
        raise ValueError("evaluation set and dataset content differ")
    return body
