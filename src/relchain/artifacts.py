from __future__ import annotations

import hashlib
import json
from time import perf_counter
from pathlib import Path
from typing import Any

from .models import ConversationTurn, Episode, Event, RelationCategory, StateTransition, UserState
from .pipeline import MemorySystem
from .relations import RelationCandidate


def content_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def freeze_common(system: MemorySystem, path: str | Path, *, build_key: str | None = None,
                  timings: dict[str, float] | None = None) -> dict[str, Any]:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"common artifact already exists: {target}")
    events = {item.event_id: item for item in system.store.list_events()}
    episodes = system.store.list_episodes()
    start = perf_counter()
    candidates = system.relation_builder.candidate_pairs(episodes, events)
    if timings is not None:
        timings["candidate_graph_seconds"] = perf_counter() - start
    candidate_rows = [
        {"candidate_id": item.candidate_id, "source_episode_id": item.source.episode_id,
         "target_episode_id": item.target.episode_id, "category": item.category.value,
         "evidence_event_ids": item.evidence_event_ids, "confidence": item.confidence,
         "symmetric": item.category == RelationCategory.SEMANTIC}
        for item in candidates
    ]
    common = {
        "turns": sorted(system.store._raw_models("turns"), key=lambda row: (row["conversation_id"], row["turn_id"])),
        "events": sorted(system.store._raw_models("events"), key=lambda row: (row["event_id"], row["revision"])),
        "episodes": sorted(system.store._raw_models("episodes"), key=lambda row: (row["episode_id"], row["revision"])),
        "user_states": sorted(system.store._raw_models("user_states"), key=lambda row: (row["state_id"], row["revision"])),
        "state_transitions": sorted(system.store._raw_models("state_transitions"), key=lambda row: row["transition_id"]),
    }
    body: dict[str, Any] = {
        "schema_version": 1,
        "build_config": system.config.model_dump(mode="json", exclude={"relation_mode", "reader_name", "context_token_budget", "full_input_token_budget", "max_hops", "max_episodes", "include_relation_context", "category_boost", "confidence_policy"}),
        "common": common,
        "candidates": candidate_rows,
        "node_hash": content_hash(common),
        "candidate_hash": content_hash(candidate_rows),
        "build_key": build_key,
        "complete": True,
    }
    body["content_hash"] = content_hash(body)
    target.parent.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    target.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    if timings is not None:
        timings["artifact_write_seconds"] = perf_counter() - start
    return body


def load_common(path: str | Path) -> dict[str, Any]:
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    if body.get("schema_version") != 1:
        raise ValueError("unsupported common artifact schema")
    expected = body.get("content_hash")
    if content_hash({key: value for key, value in body.items() if key != "content_hash"}) != expected:
        raise ValueError("common artifact content hash mismatch")
    if body.get("complete") is False:
        raise ValueError("common artifact is incomplete")
    return body


def materialize_common(system: MemorySystem, body: dict[str, Any]) -> None:
    store = system.store
    existing = store.get_metadata("common_artifact_hash")
    if existing is not None:
        if existing != body["content_hash"]:
            raise ValueError("database belongs to a different common artifact")
        for table in ("turns", "events", "episodes", "user_states", "state_transitions"):
            expected = {content_hash(row) for row in body["common"][table]}
            actual = {content_hash(row) for row in store._raw_models(table)}
            if expected != actual or len(expected) != len(body["common"][table]):
                raise ValueError("materialized common memory failed integrity check")
        return
    common = body["common"]
    for table, artifact_key, id_field in (
        ("turns", "turns", ("conversation_id", "turn_id")),
        ("events", "events", ("event_id", "revision")),
        ("episodes", "episodes", ("episode_id", "revision")),
        ("user_states", "user_states", ("state_id", "revision")),
        ("state_transitions", "state_transitions", ("transition_id",)),
    ):
        expected = {tuple(row[field] for field in id_field): row for row in common[artifact_key]}
        for existing_row in store._raw_models(table):
            key = tuple(existing_row[field] for field in id_field)
            if expected.get(key) != existing_row:
                raise ValueError("database contains records outside the common artifact")
    if store.counts()["relations"]:
        raise ValueError("database has relations without a common artifact hash")
    for raw in common["turns"]:
        store.save_turn(ConversationTurn.model_validate(raw))
    store.save_events(Event.model_validate(raw) for raw in common["events"])
    store.save_episodes(Episode.model_validate(raw) for raw in common["episodes"])
    for raw in common["user_states"]:
        store.save_state(UserState.model_validate(raw))
    for raw in common["state_transitions"]:
        store.save_transition(StateTransition.model_validate(raw))
    store.set_metadata("common_artifact_hash", body["content_hash"])


def build_mode_relations(system: MemorySystem, body: dict[str, Any], *, generation_signature: str = "deterministic-v1",
                         metrics: dict[str, Any] | None = None) -> str:
    import_start = perf_counter()
    materialize_common(system, body)
    if metrics is not None:
        metrics["common_import_seconds"] = perf_counter() - import_start
    config_hash = content_hash({
        "common": body["content_hash"], "mode": system.config.relation_mode.value,
        "generator": generation_signature, "confidence_policy": system.config.confidence_policy,
    })
    prior_config = system.store.get_metadata("relation_config_hash")
    if prior_config is not None and prior_config != config_hash:
        raise ValueError("relation generation conditions differ from existing database")
    prior = system.store.get_metadata("relation_artifact_hash")
    if prior is not None:
        current = system.store.list_relations()
        if len(current) != len(body["candidates"]) or {item.candidate_id for item in current} != {row["candidate_id"] for row in body["candidates"]}:
            raise ValueError("completed relation artifact has missing or unexpected edges")
        if content_hash([item.model_dump(mode="json") for item in current]) != prior:
            raise ValueError("completed relation artifact content changed")
        if metrics is not None:
            metrics.update({"cache_hit": True, "generated_edges": 0,
                            "reused_edges": len(current), "relation_generation_seconds": 0.0,
                            "db_write_seconds": 0.0})
        return prior
    system.store.set_metadata("relation_config_hash", config_hash)
    episodes = {item.episode_id: item for item in system.store.list_episodes()}
    candidates = [RelationCandidate(
        source=episodes[row["source_episode_id"]],
        target=episodes[row["target_episode_id"]],
        category=RelationCategory(row["category"]),
        evidence_event_ids=tuple(row["evidence_event_ids"]),
        confidence=row["confidence"],
    ) for row in body["candidates"]]
    if any(item.candidate_id != row["candidate_id"] for item, row in zip(candidates, body["candidates"])):
        raise ValueError("candidate ID mismatch")
    saved = {item.candidate_id: item for item in system.store.list_relations()}
    reused_edges = len(saved)
    generated_edges = 0
    generation_seconds = 0.0
    write_seconds = 0.0
    expected_ids = {candidate.candidate_id for candidate in candidates}
    if any(candidate_id not in expected_ids for candidate_id in saved):
        raise ValueError("incomplete relation artifact has unexpected edges")
    for candidate in candidates:
        if candidate.candidate_id in saved:
            continue
        started = perf_counter()
        relation = system.relation_builder.build(list(episodes.values()), {}, (candidate,))[0]
        generation_seconds += perf_counter() - started
        started = perf_counter()
        system.store.save_relations((relation,))
        write_seconds += perf_counter() - started
        saved[candidate.candidate_id] = relation
        generated_edges += 1
    relations = system.store.list_relations()
    if len(relations) != len(candidates):
        raise ValueError("relation artifact is incomplete")
    digest = content_hash([item.model_dump(mode="json") for item in relations])
    system.store.set_metadata("relation_artifact_hash", digest)
    if metrics is not None:
        metrics.update({"cache_hit": False, "generated_edges": generated_edges,
                        "reused_edges": reused_edges,
                        "relation_generation_seconds": generation_seconds,
                        "db_write_seconds": write_seconds})
    return digest
