from __future__ import annotations

from typing import Protocol, Sequence

from .models import (
    Answer,
    ConversationTurn,
    Episode,
    Event,
    RelationCategory,
    RetrievalResult,
    StateCandidate,
)


class Embedder(Protocol):
    name: str

    def embed(self, text: str) -> tuple[float, ...]: ...

    def similarity(self, left: Sequence[float], right: Sequence[float]) -> float: ...


class EventExtractor(Protocol):
    def extract(self, turn: ConversationTurn) -> list[Event]: ...


class StateExtractor(Protocol):
    def extract(
        self, episode: Episode, events: Sequence[Event]
    ) -> list[StateCandidate]: ...


class EpisodeAnnotator(Protocol):
    def decide_boundary(self, previous: Event, current: Event) -> tuple[bool, str]: ...

    def summarize(self, events: Sequence[Event]) -> tuple[str, str, str]: ...


class RelationTextGenerator(Protocol):
    def free_relation(
        self, source: Episode, target: Episode, category_hint: RelationCategory
    ) -> tuple[str, str, float]: ...

    def hybrid_relation(
        self, source: Episode, target: Episode, category: RelationCategory
    ) -> tuple[str, str, float]: ...


class AnswerGenerator(Protocol):
    def generate(self, retrieval: RetrievalResult) -> Answer: ...
