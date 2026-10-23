from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from typing import Any, Protocol, TypeVar

import httpx
from pydantic import BaseModel, Field

from .models import (
    Answer,
    AssertionType,
    ConversationTurn,
    Episode,
    Event,
    Modality,
    Polarity,
    RelationCategory,
    RetrievalResult,
    SourceSpan,
    StateCandidate,
)
from .text import jaccard, stable_id, tokenize

T = TypeVar("T")


class JSONTransport(Protocol):
    def complete(self, prompt: str) -> str: ...


class OpenAICompatibleTransport:
    """Synchronous transport for OpenAI-compatible chat-completions servers."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
        seed: int = 0,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.seed = seed
        self.extra_body = extra_body or {}

    def complete(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return only valid JSON matching the requested schema. "
                        "Do not add facts absent from the supplied evidence."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "seed": self.seed,
            "max_tokens": 512,
            "response_format": {"type": "json_object"},
        }
        payload.update(self.extra_body)
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(self.url, headers=headers, json=payload)
            response.raise_for_status()
        body = response.json()
        try:
            return str(body["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                "LLM response did not contain choices[0].message.content"
            ) from exc


class ValidatedJSONCaller:
    """Parse, validate, and retry once; invalid output never reaches memory."""

    def __init__(self, transport: JSONTransport) -> None:
        self.transport = transport

    def call(self, prompt: str, validator: Callable[[Any], T]) -> T:
        error = ""
        for attempt in range(2):
            current = prompt
            if attempt:
                current += (
                    "\nYour previous output was invalid. Correct it without changing "
                    f"the evidence. Validation error: {error}"
                )
            raw = self.transport.complete(current)
            try:
                data = json.loads(self._strip_fence(raw))
                return validator(data)
            except Exception as exc:
                error = str(exc)
        raise ValueError(f"LLM JSON validation failed after one retry: {error}")

    @staticmethod
    def _strip_fence(raw: str) -> str:
        text = raw.strip()
        match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.S | re.I)
        return match.group(1) if match else text


class _EventItem(BaseModel):
    event_text: str
    source_start: int = Field(ge=0)
    source_end: int = Field(ge=0)
    subject_id: str
    entity_ids: tuple[str, ...] = ()
    event_time_start: datetime | None = None
    event_time_end: datetime | None = None
    time_precision: str = "unknown"
    modality: Modality = Modality.ACTUAL
    polarity: Polarity = Polarity.POSITIVE


class _EventResponse(BaseModel):
    events: list[_EventItem]


class LLMEventExtractor:
    def __init__(self, caller: ValidatedJSONCaller) -> None:
        self.caller = caller

    def extract(self, turn: ConversationTurn) -> list[Event]:
        prompt = f"""Extract atomic grounded events from this turn.
Keep speaker_id and subject_id distinct. Do not turn desires or plans into completed facts.
Every source range must be a valid character range in raw_text.
Schema: {{"events":[{{"event_text":"...","source_start":0,"source_end":1,
"subject_id":"...","entity_ids":[],"event_time_start":null,"event_time_end":null,
"time_precision":"unknown|day|month|exact","modality":"actual|planned|desired|hypothetical",
"polarity":"positive|negative"}}]}}
conversation_id={turn.conversation_id}
turn_id={turn.turn_id}
speaker_id={turn.speaker_id}
observed_at={turn.observed_at.isoformat()}
raw_text={json.dumps(turn.text, ensure_ascii=False)}"""

        def validate(data: Any) -> list[Event]:
            response = _EventResponse.model_validate(data)
            if not response.events:
                raise ValueError("at least one event is required")
            result: list[Event] = []
            for item in response.events:
                if (
                    item.source_end > len(turn.text)
                    or item.source_end <= item.source_start
                ):
                    raise ValueError("source range is outside raw_text")
                if not turn.text[item.source_start : item.source_end].strip():
                    raise ValueError("source range is empty")
                result.append(
                    Event(
                        event_id=stable_id(
                            turn.conversation_id + ":event",
                            turn.turn_id,
                            item.source_start,
                            item.event_text,
                        ),
                        conversation_id=turn.conversation_id,
                        turn_id=turn.turn_id,
                        speaker_id=turn.speaker_id,
                        subject_id=item.subject_id,
                        event_text=item.event_text,
                        source_span=SourceSpan(
                            turn_id=turn.turn_id,
                            start=item.source_start,
                            end=item.source_end,
                        ),
                        entity_ids=item.entity_ids,
                        observed_at=turn.observed_at,
                        event_time_start=item.event_time_start,
                        event_time_end=item.event_time_end,
                        time_precision=item.time_precision,
                        modality=item.modality,
                        polarity=item.polarity,
                    )
                )
            return result

        return self.caller.call(prompt, validate)


class _BoundaryResponse(BaseModel):
    split: bool
    reason: str


class _SummaryResponse(BaseModel):
    title: str
    summary: str
    topic: str


class LLMEpisodeAnnotator:
    def __init__(self, caller: ValidatedJSONCaller) -> None:
        self.caller = caller

    def decide_boundary(self, previous: Event, current: Event) -> tuple[bool, str]:
        prompt = f"""Decide whether the second grounded event starts a new semantic episode.
Use topic/activity change; do not invent context.
Schema: {{"split":true,"reason":"topic_shift|activity_change|same_episode"}}
previous={previous.model_dump_json()}
current={current.model_dump_json()}"""
        response = self.caller.call(prompt, _BoundaryResponse.model_validate)
        return response.split, response.reason

    def summarize(self, events: Sequence[Event]) -> tuple[str, str, str]:
        serialized = [item.model_dump(mode="json") for item in events]
        prompt = f"""Summarize these grounded events as one episode.
Preserve who did what, time, result, and unfinished items. Do not infer causes or preferences.
Schema: {{"title":"...","summary":"...","topic":"..."}}
events={json.dumps(serialized, ensure_ascii=False)}"""
        response = self.caller.call(prompt, _SummaryResponse.model_validate)
        return response.title, response.summary, response.topic


class _StateItem(BaseModel):
    subject_id: str
    attribute: str
    value: str
    scope: str = "general"
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    assertion_type: AssertionType
    evidence_event_ids: tuple[str, ...]
    counter_evidence_ids: tuple[str, ...] = ()
    confidence: float = Field(ge=0.0, le=1.0)
    change_cue: bool = False
    retraction_cue: bool = False
    multi_valued: bool = True


class _StateResponse(BaseModel):
    states: list[_StateItem]


class LLMStateExtractor:
    def __init__(self, caller: ValidatedJSONCaller) -> None:
        self.caller = caller

    def extract(
        self, episode: Episode, events: Sequence[Event]
    ) -> list[StateCandidate]:
        allowed_ids = set(episode.event_ids)
        serialized = [item.model_dump(mode="json") for item in events]
        prompt = f"""Extract user-state claims only: preferences, interests, goals,
ongoing activities, and current situations. One experience is not a stable preference.
Mark explicit vs inferred, change, correction, scope, and evidence IDs.
Schema: {{"states":[{{"subject_id":"user","attribute":"...","value":"...",
"scope":"general","valid_from":null,"valid_to":null,"assertion_type":"explicit|inferred",
"evidence_event_ids":[],"counter_evidence_ids":[],"confidence":1.0,
"change_cue":false,"retraction_cue":false,"multi_valued":true}}]}}
episode={episode.model_dump_json()}
events={json.dumps(serialized, ensure_ascii=False)}"""

        def validate(data: Any) -> list[StateCandidate]:
            response = _StateResponse.model_validate(data)
            result: list[StateCandidate] = []
            for item in response.states:
                if not item.evidence_event_ids or not set(
                    item.evidence_event_ids
                ).issubset(allowed_ids):
                    raise ValueError(
                        "state evidence must reference events in the episode"
                    )
                evidence_events = [
                    event
                    for event in events
                    if event.event_id in item.evidence_event_ids
                ]
                recorded_at = max(event.observed_at for event in evidence_events)
                result.append(
                    StateCandidate(
                        **item.model_dump(),
                        recorded_at=recorded_at,
                        evidence_episode_ids=(episode.episode_id,),
                    )
                )
            return result

        return self.caller.call(prompt, validate)


class _RelationResponse(BaseModel):
    relation: str
    description: str
    confidence: float = Field(ge=0.0, le=1.0)


class LLMRelationTextGenerator:
    def __init__(self, caller: ValidatedJSONCaller) -> None:
        self.caller = caller

    def _generate(
        self, source: Episode, target: Episode, category: RelationCategory, hybrid: bool
    ) -> tuple[str, str, float]:
        category_rule = (
            f"The fixed upper category is {category.value}; do not change it."
            if hybrid
            else "Do not choose from a predefined relation vocabulary."
        )
        prompt = f"""Generate a short, specific snake_case relation from source to target.
Avoid generic related_to/associated_with/connected_to. {category_rule}
Schema: {{"relation":"...","description":"...","confidence":0.0}}
source={source.model_dump_json()}
target={target.model_dump_json()}"""

        def validate(data: Any) -> _RelationResponse:
            response = _RelationResponse.model_validate(data)
            if not re.fullmatch(r"[a-z][a-z0-9_]{2,80}", response.relation):
                raise ValueError("relation must be concise snake_case")
            if response.relation in {"related_to", "associated_with", "connected_to"}:
                raise ValueError("generic relation is forbidden")
            return response

        response = self.caller.call(prompt, validate)
        return response.relation, response.description, response.confidence

    def free_relation(
        self, source: Episode, target: Episode, category_hint: RelationCategory
    ) -> tuple[str, str, float]:
        return self._generate(source, target, category_hint, hybrid=False)

    def hybrid_relation(
        self, source: Episode, target: Episode, category: RelationCategory
    ) -> tuple[str, str, float]:
        return self._generate(source, target, category, hybrid=True)


class _AnswerResponse(BaseModel):
    answer: str
    evidence_event_ids: tuple[str, ...]
    uncertainty: str | None = None


class LLMAnswerGenerator:
    def __init__(self, caller: ValidatedJSONCaller) -> None:
        self.caller = caller

    def generate(self, retrieval: RetrievalResult) -> Answer:
        allowed_events = set(retrieval.trace.evidence_event_ids)
        allowed_event_list = sorted(allowed_events)
        prompt = f"""Answer the question using only the supplied memory context.
Prefer raw turns over summaries. Distinguish current, past, explicit, inferred, and uncertain.
Give only the concise answer, normally a date, name, or short phrase; do not restate the
question or add merely plausible options. Resolve relative dates such as yesterday and
last year against the ISO timestamp on the TURN line, returning an absolute date or year
when the question asks when. Cite exactly one best EVENT identifier verbatim from
allowed_evidence_event_ids whenever the answer is not empty.
Schema: {{"answer":"...","evidence_event_ids":[],"uncertainty":null}}
question={json.dumps(retrieval.question, ensure_ascii=False)}
allowed_evidence_event_ids={json.dumps(allowed_event_list)}
context={retrieval.context}"""

        def validate(data: Any) -> _AnswerResponse:
            response = _AnswerResponse.model_validate(data)
            valid_ids = tuple(
                event_id
                for event_id in response.evidence_event_ids
                if event_id in allowed_events
            )
            if valid_ids != response.evidence_event_ids:
                response = response.model_copy(update={"evidence_event_ids": valid_ids})
            if response.answer and not valid_ids and allowed_events:
                # Small local readers occasionally omit the citation array even when
                # their answer is grounded in the supplied context.  Recover the most
                # lexically compatible source instead of discarding the whole answer.
                answer_tokens = tokenize(response.answer)
                best = max(
                    retrieval.evidence,
                    key=lambda item: jaccard(answer_tokens, tokenize(item.raw_text)),
                )
                response = response.model_copy(
                    update={"evidence_event_ids": (best.event_id,)}
                )
            return response

        response = self.caller.call(prompt, validate)
        answer_text = self._normalize_relative_time(
            retrieval, response.evidence_event_ids, response.answer
        )
        turn_ids = tuple(
            dict.fromkeys(
                item.turn_id
                for item in retrieval.evidence
                if item.event_id in response.evidence_event_ids
            )
        )
        return Answer(
            text=answer_text,
            evidence_turn_ids=turn_ids,
            evidence_event_ids=response.evidence_event_ids,
            uncertainty=response.uncertainty,
            retrieval=retrieval,
        )

    @staticmethod
    def _normalize_relative_time(
        retrieval: RetrievalResult,
        evidence_event_ids: tuple[str, ...],
        answer: str,
    ) -> str:
        if not retrieval.question.lower().strip().startswith("when"):
            return answer
        cited = [
            item for item in retrieval.evidence if item.event_id in evidence_event_ids
        ]
        for item in cited:
            raw = item.raw_text.lower()
            resolved: datetime | None = None
            if "yesterday" in raw:
                resolved = item.observed_at - timedelta(days=1)
            else:
                match = re.search(r"\b(\d+)\s+days?\s+ago\b", raw)
                if match:
                    resolved = item.observed_at - timedelta(days=int(match.group(1)))
            if resolved is not None:
                return f"{resolved.day} {resolved.strftime('%B %Y')}"
            if "last year" in raw or "a year ago" in raw:
                return str(item.observed_at.year - 1)
        return answer
