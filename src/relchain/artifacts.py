from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .models import ConversationTurn, Episode, Event, RelationCategory, StateTransition, UserState
from .pipeline import MemorySystem
from .relations import RelationCandidate


def content_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def freeze_common(system: MemorySystem, path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"common artifact already exists: {target}")
    events = {item.event_id: item for item in system.store.list_events()}
    episodes = system.store.list_episodes()
    candidates = system.relation_builder.candidate_pairs(episodes, events)
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
    }
    body["content_hash"] = content_hash(body)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    return body


def load_common(path: str | Path) -> dict[str, Any]:
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    if body.get("schema_version") != 1:
        raise ValueError("unsupported common artifact schema")
    expected = body.get("content_hash")
    if content_hash({key: value for key, value in body.items() if key != "content_hash"}) != expected:
        raise ValueError("common artifact content hash mismatch")
    return body


def materialize_common(system: MemorySystem, body: dict[str, Any]) -> None:
    store = system.store
    existing = store.get_metadata("common_artifact_hash")
    if existing is not None:
        if existing != body["content_hash"]:
            raise ValueError("database belongs to a different common artifact")
        return
    if any(store.counts().values()):
        raise ValueError("database contains existing memory without a common artifact hash")
    common = body["common"]
    for raw in common["turns"]:
        store.save_turn(ConversationTurn.model_validate(raw))
    store.save_events(Event.model_validate(raw) for raw in common["events"])
    store.save_episodes(Episode.model_validate(raw) for raw in common["episodes"])
    for raw in common["user_states"]:
        store.save_state(UserState.model_validate(raw))
    for raw in common["state_transitions"]:
        store.save_transition(StateTransition.model_validate(raw))
    store.set_metadata("common_artifact_hash", body["content_hash"])


def build_mode_relations(system: MemorySystem, body: dict[str, Any], *, generation_signature: str = "deterministic-v1") -> str:
    materialize_common(system, body)
    config_hash = content_hash({
        "common": body["content_hash"], "mode": system.config.relation_mode.value,
        "generator": generation_signature, "confidence_policy": system.config.confidence_policy,
    })
    prior = system.store.get_metadata("relation_artifact_hash")
    if prior is not None:
        if system.store.get_metadata("relation_config_hash") != config_hash:
            raise ValueError("relation generation conditions differ from existing database")
        return prior
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
    relations = system.relation_builder.build(list(episodes.values()), {}, candidates)
    system.store.replace_relations(system.config.relation_mode.value, relations)
    digest = content_hash([item.model_dump(mode="json") for item in relations])
    system.store.set_metadata("relation_artifact_hash", digest)
    system.store.set_metadata("relation_config_hash", config_hash)
    return digest
