from datetime import datetime, timezone

from relchain.config import MemoryConfig
from relchain.models import AssertionType, StateCandidate, StateOperation, StateStatus
from relchain.state import StateUpdater
from relchain.storage import SQLiteMemoryStore

NOW = datetime(2026, 10, 23, tzinfo=timezone.utc)


def candidate(
    value: str,
    event: str,
    episode: str,
    *,
    assertion: AssertionType = AssertionType.EXPLICIT,
    change: bool = False,
    retraction: bool = False,
    multi: bool = True,
) -> StateCandidate:
    return StateCandidate(
        subject_id="user",
        attribute="preference",
        value=value,
        scope="travel",
        recorded_at=NOW,
        assertion_type=assertion,
        evidence_episode_ids=(episode,),
        evidence_event_ids=(event,),
        change_cue=change,
        retraction_cue=retraction,
        multi_valued=multi,
    )


def test_inferred_state_requires_independent_episode_support(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    updater = StateUpdater(store, MemoryConfig(inferred_state_min_episodes=2))
    states, transitions = updater.apply(
        candidate(
            "near_station", "e1", "p1", assertion=AssertionType.INFERRED, multi=False
        )
    )
    assert states[0].status == StateStatus.UNCERTAIN
    assert transitions[0].operation == StateOperation.ADD

    states, transitions = updater.apply(
        candidate(
            "near_station", "e2", "p2", assertion=AssertionType.INFERRED, multi=False
        )
    )
    assert states[0].status == StateStatus.ACTIVE
    assert transitions[0].operation == StateOperation.SUPPORT


def test_state_operations_and_history(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    updater = StateUpdater(store, MemoryConfig())
    updater.apply(candidate("station", "e1", "p1", multi=False))
    states, transitions = updater.apply(
        candidate("quiet", "e2", "p2", change=True, multi=False)
    )
    assert {item.status for item in states} == {
        StateStatus.SUPERSEDED,
        StateStatus.ACTIVE,
    }
    assert all(item.operation == StateOperation.SUPERSEDE for item in transitions)

    _, transitions = updater.apply(candidate("quiet", "e3", "p3", retraction=True))
    assert transitions[0].operation == StateOperation.RETRACT
    assert store.list_transitions()


def test_duplicate_evidence_is_idempotent(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    updater = StateUpdater(store, MemoryConfig())
    item = candidate("station", "e1", "p1")
    updater.apply(item)
    states, transitions = updater.apply(item)
    assert states == []
    assert transitions == []
