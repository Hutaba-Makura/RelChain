from datetime import datetime, timedelta, timezone

import pytest

from relchain import MemoryConfig, MemorySystem, RelationMode
from relchain.models import ConversationTurn, StateStatus
from relchain.storage import SQLiteMemoryStore


def make_turn(turn_id: str, text: str, day: int) -> ConversationTurn:
    return ConversationTurn(
        conversation_id="demo",
        turn_id=turn_id,
        speaker_id="user",
        text=text,
        observed_at=datetime(2026, 10, 20, 10, 0, tzinfo=timezone.utc)
        + timedelta(days=day),
    )


def test_end_to_end_grounded_answer_and_idempotency(tmp_path) -> None:
    database = tmp_path / "hybrid.db"
    system = MemorySystem(database, MemoryConfig(relation_mode=RelationMode.HYBRID))
    turns = [
        make_turn("t1", "旅行では駅近のホテルが好きです。", 0),
        make_turn("t2", "昨日、駅の近くのホテルに泊まりました。", 1),
        make_turn(
            "t3",
            "以前は旅行では駅近を重視していたが、今は旅行では静かな立地を重視する。",
            2,
        ),
    ]
    report = system.ingest(turns)
    assert not report.failures
    assert report.event_ids and report.episode_ids and report.relation_ids
    assert system.store.counts()["turns"] == 3

    active = system.store.list_states(statuses=(StateStatus.ACTIVE,))
    assert any("静かな立地" in item.value for item in active)
    answer = system.query("今の旅行の好みは？")
    assert "静かな立地" in answer.text
    assert answer.evidence_turn_ids
    for event_id in answer.evidence_event_ids:
        event = system.store.get_event(event_id)
        assert event is not None
        assert system.store.get_turn(event.conversation_id, event.turn_id) is not None

    before = system.store.counts()
    second = system.ingest(turns)
    assert not second.turn_ids
    assert system.store.counts() == before


def test_append_only_turns_and_mode_isolation(tmp_path) -> None:
    database = tmp_path / "fixed.db"
    system = MemorySystem(database, MemoryConfig(relation_mode=RelationMode.FIXED))
    system.ingest([make_turn("t1", "ロックが好きです。", 0)])
    with pytest.raises(ValueError, match="append-only"):
        system.ingest([make_turn("t1", "ジャズが好きです。", 0)])
    with pytest.raises(ValueError, match="relation mode mismatch"):
        MemorySystem(database, MemoryConfig(relation_mode=RelationMode.FREE))


def test_export_contains_all_layers(tmp_path) -> None:
    system = MemorySystem(tmp_path / "memory.db")
    system.ingest([make_turn("t1", "旅行では駅近のホテルが好きです。", 0)])
    exported = SQLiteMemoryStore(tmp_path / "memory.db").export_json()
    assert exported["turns"]
    assert exported["events"]
    assert exported["episodes"]
    assert exported["user_states"]
