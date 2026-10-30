from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, TypeVar

from pydantic import BaseModel

from .models import (
    ConversationTurn,
    Episode,
    Event,
    Relation,
    RetrievalTrace,
    StateStatus,
    StateTransition,
    UserState,
)

T = TypeVar("T", bound=BaseModel)


class SQLiteMemoryStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS turns (
                    conversation_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    PRIMARY KEY (conversation_id, turn_id)
                );
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    conversation_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    PRIMARY KEY (event_id, revision),
                    FOREIGN KEY (conversation_id, turn_id)
                        REFERENCES turns(conversation_id, turn_id)
                );
                CREATE INDEX IF NOT EXISTS ix_events_conversation
                    ON events(conversation_id, observed_at);
                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    conversation_id TEXT NOT NULL,
                    observed_start TEXT NOT NULL,
                    observed_end TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    PRIMARY KEY (episode_id, revision)
                );
                CREATE INDEX IF NOT EXISTS ix_episodes_conversation
                    ON episodes(conversation_id, observed_start);
                CREATE TABLE IF NOT EXISTS relations (
                    edge_id TEXT PRIMARY KEY,
                    source_episode_id TEXT NOT NULL,
                    target_episode_id TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    category TEXT,
                    status TEXT NOT NULL,
                    data_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ix_relations_source
                    ON relations(source_episode_id, status);
                CREATE INDEX IF NOT EXISTS ix_relations_target
                    ON relations(target_episode_id, status);
                CREATE TABLE IF NOT EXISTS user_states (
                    state_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    subject_id TEXT NOT NULL,
                    attribute TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    status TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    PRIMARY KEY (state_id, revision)
                );
                CREATE INDEX IF NOT EXISTS ix_states_lookup
                    ON user_states(subject_id, attribute, scope, status);
                CREATE TABLE IF NOT EXISTS state_transitions (
                    transition_id TEXT PRIMARY KEY,
                    state_id TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    data_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS failures (
                    failure_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stage TEXT NOT NULL,
                    reference_id TEXT,
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS retrieval_logs (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    data_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _json(model: BaseModel) -> str:
        return model.model_dump_json()

    @staticmethod
    def _parse(model_type: type[T], raw: str) -> T:
        return model_type.model_validate_json(raw)

    def set_metadata(self, key: str, value: str) -> None:
        with self.connect() as con:
            con.execute(
                "INSERT INTO metadata(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def get_metadata(self, key: str) -> str | None:
        with self.connect() as con:
            row = con.execute(
                "SELECT value FROM metadata WHERE key = ?", (key,)
            ).fetchone()
        return None if row is None else str(row["value"])

    def save_turn(self, turn: ConversationTurn) -> None:
        with self.connect() as con:
            row = con.execute(
                "SELECT data_json FROM turns WHERE conversation_id=? AND turn_id=?",
                (turn.conversation_id, turn.turn_id),
            ).fetchone()
            if row is not None:
                if self._parse(ConversationTurn, row["data_json"]) != turn:
                    raise ValueError(
                        f"turn is append-only and already exists: {turn.turn_id}"
                    )
                return
            con.execute(
                "INSERT INTO turns VALUES (?, ?, ?, ?)",
                (
                    turn.conversation_id,
                    turn.turn_id,
                    turn.observed_at.isoformat(),
                    self._json(turn),
                ),
            )

    def save_events(self, events: Iterable[Event]) -> None:
        with self.connect() as con:
            con.executemany(
                "INSERT OR IGNORE INTO events VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        event.event_id,
                        event.revision,
                        event.conversation_id,
                        event.turn_id,
                        event.observed_at.isoformat(),
                        self._json(event),
                    )
                    for event in events
                ],
            )

    def save_episodes(self, episodes: Iterable[Episode]) -> None:
        with self.connect() as con:
            con.executemany(
                "INSERT OR IGNORE INTO episodes VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        item.episode_id,
                        item.revision,
                        item.conversation_id,
                        item.observed_start.isoformat(),
                        item.observed_end.isoformat(),
                        self._json(item),
                    )
                    for item in episodes
                ],
            )

    def save_relations(self, relations: Iterable[Relation]) -> None:
        with self.connect() as con:
            con.executemany(
                "INSERT OR REPLACE INTO relations VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        item.edge_id,
                        item.source_episode_id,
                        item.target_episode_id,
                        item.mode,
                        item.category.value if item.category else None,
                        item.status,
                        self._json(item),
                    )
                    for item in relations
                ],
            )

    def replace_relations(self, mode: str, relations: Iterable[Relation]) -> None:
        items = list(relations)
        with self.connect() as con:
            con.execute("DELETE FROM relations WHERE mode=?", (mode,))
            con.executemany(
                "INSERT INTO relations VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        item.edge_id,
                        item.source_episode_id,
                        item.target_episode_id,
                        item.mode,
                        item.category.value if item.category else None,
                        item.status,
                        self._json(item),
                    )
                    for item in items
                ],
            )

    def save_state(self, state: UserState) -> None:
        with self.connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO user_states VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    state.state_id,
                    state.revision,
                    state.subject_id,
                    state.attribute,
                    state.scope,
                    state.status.value,
                    state.recorded_at.isoformat(),
                    self._json(state),
                ),
            )

    def save_transition(self, transition: StateTransition) -> None:
        with self.connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO state_transitions VALUES (?, ?, ?, ?, ?)",
                (
                    transition.transition_id,
                    transition.state_id,
                    transition.operation.value,
                    transition.recorded_at.isoformat(),
                    self._json(transition),
                ),
            )

    def record_failure(
        self, stage: str, reference_id: str | None, message: str
    ) -> None:
        with self.connect() as con:
            con.execute(
                "INSERT INTO failures(stage, reference_id, message, created_at) "
                "VALUES (?, ?, ?, ?)",
                (stage, reference_id, message, datetime.now(timezone.utc).isoformat()),
            )

    def log_retrieval(self, trace: RetrievalTrace) -> None:
        with self.connect() as con:
            con.execute(
                "INSERT INTO retrieval_logs(question, created_at, data_json) VALUES (?, ?, ?)",
                (
                    trace.question,
                    datetime.now(timezone.utc).isoformat(),
                    self._json(trace),
                ),
            )

    def get_turn(self, conversation_id: str, turn_id: str) -> ConversationTurn | None:
        with self.connect() as con:
            row = con.execute(
                "SELECT data_json FROM turns WHERE conversation_id=? AND turn_id=?",
                (conversation_id, turn_id),
            ).fetchone()
        return None if row is None else self._parse(ConversationTurn, row["data_json"])

    def has_events_for_turn(self, conversation_id: str, turn_id: str) -> bool:
        with self.connect() as con:
            row = con.execute(
                "SELECT 1 FROM events WHERE conversation_id=? AND turn_id=? LIMIT 1",
                (conversation_id, turn_id),
            ).fetchone()
        return row is not None

    def get_event(self, event_id: str) -> Event | None:
        with self.connect() as con:
            row = con.execute(
                "SELECT data_json FROM events WHERE event_id=? ORDER BY revision DESC LIMIT 1",
                (event_id,),
            ).fetchone()
        return None if row is None else self._parse(Event, row["data_json"])

    def get_episode(self, episode_id: str) -> Episode | None:
        with self.connect() as con:
            row = con.execute(
                "SELECT data_json FROM episodes WHERE episode_id=? ORDER BY revision DESC LIMIT 1",
                (episode_id,),
            ).fetchone()
        return None if row is None else self._parse(Episode, row["data_json"])

    def _all_latest(self, table: str, id_column: str, model_type: type[T]) -> list[T]:
        if table not in {"events", "episodes", "user_states"}:
            raise ValueError("unsupported table")
        query = (
            f"SELECT item.data_json FROM {table} item JOIN "
            f"(SELECT {id_column}, MAX(revision) revision FROM {table} "
            f"GROUP BY {id_column}) latest "
            f"ON item.{id_column}=latest.{id_column} AND item.revision=latest.revision"
        )
        with self.connect() as con:
            rows = con.execute(query).fetchall()
        return [self._parse(model_type, row["data_json"]) for row in rows]

    def list_events(self, conversation_id: str | None = None) -> list[Event]:
        items = self._all_latest("events", "event_id", Event)
        if conversation_id is not None:
            items = [item for item in items if item.conversation_id == conversation_id]
        return sorted(items, key=lambda item: (item.observed_at, item.event_id))

    def list_episodes(self, conversation_id: str | None = None) -> list[Episode]:
        items = self._all_latest("episodes", "episode_id", Episode)
        if conversation_id is not None:
            items = [item for item in items if item.conversation_id == conversation_id]
        return sorted(items, key=lambda item: (item.observed_start, item.episode_id))

    def list_relations(self, status: str = "active") -> list[Relation]:
        with self.connect() as con:
            rows = con.execute(
                "SELECT data_json FROM relations WHERE status=? ORDER BY edge_id",
                (status,),
            ).fetchall()
        return [self._parse(Relation, row["data_json"]) for row in rows]

    def list_states(
        self,
        subject_id: str | None = None,
        attribute: str | None = None,
        statuses: Sequence[StateStatus] | None = None,
    ) -> list[UserState]:
        items = self._all_latest("user_states", "state_id", UserState)
        if subject_id is not None:
            items = [item for item in items if item.subject_id == subject_id]
        if attribute is not None:
            items = [item for item in items if item.attribute == attribute]
        if statuses is not None:
            accepted = set(statuses)
            items = [item for item in items if item.status in accepted]
        return sorted(
            items, key=lambda item: (item.recorded_at, item.state_id), reverse=True
        )

    def list_transitions(self, state_id: str | None = None) -> list[StateTransition]:
        with self.connect() as con:
            if state_id is None:
                rows = con.execute(
                    "SELECT data_json FROM state_transitions ORDER BY recorded_at"
                ).fetchall()
            else:
                rows = con.execute(
                    "SELECT data_json FROM state_transitions WHERE state_id=? ORDER BY recorded_at",
                    (state_id,),
                ).fetchall()
        return [self._parse(StateTransition, row["data_json"]) for row in rows]

    def counts(self) -> dict[str, int]:
        tables = (
            "turns",
            "events",
            "episodes",
            "relations",
            "user_states",
            "state_transitions",
        )
        result: dict[str, int] = {}
        with self.connect() as con:
            for table in tables:
                result[table] = int(
                    con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                )
        return result

    def export_json(self) -> dict[str, Any]:
        return {
            "turns": self._raw_models("turns"),
            "events": [item.model_dump(mode="json") for item in self.list_events()],
            "episodes": [item.model_dump(mode="json") for item in self.list_episodes()],
            "relations": [
                item.model_dump(mode="json") for item in self.list_relations()
            ],
            "user_states": [
                item.model_dump(mode="json") for item in self.list_states()
            ],
            "state_transitions": [
                item.model_dump(mode="json") for item in self.list_transitions()
            ],
        }

    def _raw_models(self, table: str) -> list[dict[str, Any]]:
        with self.connect() as con:
            rows = con.execute(f"SELECT data_json FROM {table}").fetchall()
        return [json.loads(row["data_json"]) for row in rows]
