from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, use_enum_values=False)


class Modality(StrEnum):
    ACTUAL = "actual"
    PLANNED = "planned"
    DESIRED = "desired"
    HYPOTHETICAL = "hypothetical"


class Polarity(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class AssertionType(StrEnum):
    EXPLICIT = "explicit"
    INFERRED = "inferred"


class StateStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"
    UNCERTAIN = "uncertain"


class StateOperation(StrEnum):
    ADD = "add"
    SUPPORT = "support"
    COEXIST = "coexist"
    SUPERSEDE = "supersede"
    RETRACT = "retract"
    HOLD = "hold"


class RelationCategory(StrEnum):
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"
    CAUSAL = "causal"


class ConversationTurn(FrozenModel):
    conversation_id: str
    turn_id: str
    speaker_id: str
    text: str = Field(min_length=1)
    observed_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceSpan(FrozenModel):
    turn_id: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_range(self) -> "SourceSpan":
        if self.end < self.start:
            raise ValueError("source span end must be >= start")
        return self


class Event(FrozenModel):
    event_id: str
    revision: int = Field(default=1, ge=1)
    conversation_id: str
    turn_id: str
    speaker_id: str
    subject_id: str
    event_text: str
    source_span: SourceSpan
    entity_ids: tuple[str, ...] = ()
    observed_at: datetime
    event_time_start: datetime | None = None
    event_time_end: datetime | None = None
    time_precision: str = "unknown"
    modality: Modality = Modality.ACTUAL
    polarity: Polarity = Polarity.POSITIVE


class Episode(FrozenModel):
    episode_id: str
    revision: int = Field(default=1, ge=1)
    conversation_id: str
    title: str
    summary: str
    event_ids: tuple[str, ...]
    source_turn_ids: tuple[str, ...]
    entity_ids: tuple[str, ...] = ()
    topic: str = "general"
    observed_start: datetime
    observed_end: datetime
    event_start: datetime | None = None
    event_end: datetime | None = None
    boundary_reason: str
    embedding: tuple[float, ...] = ()


class Relation(FrozenModel):
    edge_id: str
    source_episode_id: str
    target_episode_id: str
    mode: str
    category: RelationCategory | None = None
    relation_text: str
    description: str = ""
    evidence_event_ids: tuple[str, ...] = ()
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    candidate_id: str = ""
    candidate_confidence: float | None = None
    generation_confidence: float | None = None
    symmetric: bool = False
    status: str = "active"

    @model_validator(mode="after")
    def no_self_loop(self) -> "Relation":
        if self.source_episode_id == self.target_episode_id:
            raise ValueError("relation self loops are not allowed")
        return self


class StateCandidate(FrozenModel):
    subject_id: str
    attribute: str
    value: str
    scope: str = "general"
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    recorded_at: datetime
    assertion_type: AssertionType
    evidence_episode_ids: tuple[str, ...]
    evidence_event_ids: tuple[str, ...]
    counter_evidence_ids: tuple[str, ...] = ()
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    change_cue: bool = False
    retraction_cue: bool = False
    multi_valued: bool = True


class UserState(FrozenModel):
    state_id: str
    subject_id: str
    attribute: str
    value: str
    scope: str = "general"
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    recorded_at: datetime
    revision: int = Field(default=1, ge=1)
    status: StateStatus
    assertion_type: AssertionType
    evidence_episode_ids: tuple[str, ...]
    evidence_event_ids: tuple[str, ...]
    counter_evidence_ids: tuple[str, ...] = ()
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class StateTransition(FrozenModel):
    transition_id: str
    operation: StateOperation
    state_id: str
    previous_revision: int | None = None
    new_revision: int
    recorded_at: datetime
    evidence_event_ids: tuple[str, ...]
    reason: str


class QueryIntent(FrozenModel):
    subject_id: str = "user"
    attribute: str | None = None
    reference_time: str = "current"
    preferred_categories: tuple[RelationCategory, ...] = ()
    tokens: tuple[str, ...] = ()


class EvidenceItem(FrozenModel):
    conversation_id: str = ""
    turn_id: str
    event_id: str
    event_ids: tuple[str, ...] = ()
    episode_id: str
    speaker_id: str
    observed_at: datetime
    raw_text: str
    source_start: int
    source_end: int


class RetrievalTrace(FrozenModel):
    question: str
    intent: QueryIntent
    seed_episode_ids: tuple[str, ...]
    state_ids: tuple[str, ...]
    traversed_edge_ids: tuple[str, ...]
    retrieved_episode_ids: tuple[str, ...]
    evidence_event_ids: tuple[str, ...]
    evidence_turn_ids: tuple[str, ...]
    relation_descriptions: tuple[str, ...]
    context_event_ids: tuple[str, ...] = ()
    context_turn_ids: tuple[str, ...] = ()
    context_state_ids: tuple[str, ...] = ()
    context_episode_ids: tuple[str, ...] = ()
    context_relation_ids: tuple[str, ...] = ()
    excluded_state_ids: tuple[str, ...] = ()
    excluded_episode_ids: tuple[str, ...] = ()
    excluded_relation_ids: tuple[str, ...] = ()
    memory_token_count: int = 0
    token_count_method: str = "approximate"
    inspected_edge_ids: tuple[str, ...] = ()
    retrieved_turn_refs: tuple[str, ...] = ()
    context_turn_refs: tuple[str, ...] = ()
    excluded_turn_ids: tuple[str, ...] = ()
    search_seconds: float | None = None
    context_build_seconds: float | None = None


class RetrievalResult(FrozenModel):
    question: str
    states: tuple[UserState, ...]
    episodes: tuple[Episode, ...]
    evidence: tuple[EvidenceItem, ...]
    trace: RetrievalTrace
    context: str


class Answer(FrozenModel):
    text: str
    evidence_turn_ids: tuple[str, ...]
    evidence_event_ids: tuple[str, ...]
    uncertainty: str | None = None
    retrieval: RetrievalResult
    raw_response: str | None = None
    model_cited_event_ids: tuple[str, ...] = ()
    invalid_cited_event_ids: tuple[str, ...] = ()
    citation_status: str = "not_applicable"
    response_status: str = "answered"
    prompt_token_count: int | None = None
    prompt_token_method: str | None = None


class IngestReport(FrozenModel):
    turn_ids: tuple[str, ...]
    event_ids: tuple[str, ...]
    episode_ids: tuple[str, ...]
    relation_ids: tuple[str, ...]
    state_ids: tuple[str, ...]
    transitions: tuple[StateTransition, ...]
    failures: tuple[str, ...] = ()
    timings_seconds: dict[str, float] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)
