from __future__ import annotations

import calendar
import re
from datetime import datetime, timedelta
from typing import TypedDict

from .models import (
    AssertionType,
    ConversationTurn,
    Episode,
    Event,
    Modality,
    Polarity,
    SourceSpan,
    StateCandidate,
)
from .text import sentence_spans, stable_id, tokenize

_NEGATIVE = re.compile(
    r"(?:ではない|じゃない|しない|なかった|嫌い|not\b|never\b|no longer)", re.I
)
_DESIRED = re.compile(r"(?:したい|欲しい|希望|want to|would like|hope to)", re.I)
_PLANNED = re.compile(
    r"(?:予定|つもり|計画|will\b|plan(?:ning|ned)? to|going to)", re.I
)
_HYPOTHETICAL = re.compile(r"(?:もし|ならば|だったら|\bif\b|would have)", re.I)
_OTHER_SUBJECT = re.compile(
    r"(?P<name>[A-Z][a-z]{1,20}|友人|母|父|彼|彼女|同僚)(?:は|が|の|\s)"
)
_ENTITY_STOP = {
    "です",
    "ます",
    "した",
    "して",
    "いる",
    "ある",
    "これ",
    "それ",
    "ため",
    "the",
    "and",
    "that",
    "with",
    "from",
}


def _shift_month(value: datetime, delta: int) -> datetime:
    month_index = value.year * 12 + value.month - 1 + delta
    year, month_zero = divmod(month_index, 12)
    month = month_zero + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)


def _resolve_event_time(
    text: str, observed_at: datetime
) -> tuple[datetime | None, str]:
    lowered = text.lower()
    if "一昨日" in text or "day before yesterday" in lowered:
        return observed_at - timedelta(days=2), "day"
    if "昨日" in text or "yesterday" in lowered:
        return observed_at - timedelta(days=1), "day"
    if "今日" in text or "today" in lowered:
        return observed_at, "day"
    if "明日" in text or "tomorrow" in lowered:
        return observed_at + timedelta(days=1), "day"
    if "先月" in text or "last month" in lowered:
        return _shift_month(observed_at, -1), "month"
    if "来月" in text or "next month" in lowered:
        return _shift_month(observed_at, 1), "month"
    return None, "unknown"


class RuleBasedEventExtractor:
    """Offline baseline. A production writer LLM can replace this port."""

    def extract(self, turn: ConversationTurn) -> list[Event]:
        result: list[Event] = []
        for sentence, start, end in sentence_spans(turn.text):
            other = _OTHER_SUBJECT.search(sentence)
            subject = turn.speaker_id
            if other and turn.speaker_id == "user":
                subject = f"entity:{other.group('name').lower()}"
            modality = Modality.ACTUAL
            if _HYPOTHETICAL.search(sentence):
                modality = Modality.HYPOTHETICAL
            elif _DESIRED.search(sentence):
                modality = Modality.DESIRED
            elif _PLANNED.search(sentence):
                modality = Modality.PLANNED
            event_time, precision = _resolve_event_time(sentence, turn.observed_at)
            entities = tuple(
                f"entity:{token.lower()}"
                for token in tokenize(sentence)
                if len(token) >= 2 and token.lower() not in _ENTITY_STOP
            )[:12]
            event_id = stable_id(
                turn.conversation_id + ":event", turn.turn_id, start, sentence
            )
            result.append(
                Event(
                    event_id=event_id,
                    conversation_id=turn.conversation_id,
                    turn_id=turn.turn_id,
                    speaker_id=turn.speaker_id,
                    subject_id=subject,
                    event_text=sentence,
                    source_span=SourceSpan(turn_id=turn.turn_id, start=start, end=end),
                    entity_ids=entities,
                    observed_at=turn.observed_at,
                    event_time_start=event_time,
                    event_time_end=event_time,
                    time_precision=precision,
                    modality=modality,
                    polarity=(
                        Polarity.NEGATIVE
                        if _NEGATIVE.search(sentence)
                        else Polarity.POSITIVE
                    ),
                )
            )
        return result


_JA_CHANGE = re.compile(
    r"以前は(?P<old>.+?)(?:が|を)?(?:好き|重視)[^。]*?(?:今は|現在は)(?P<new>.+?)(?:が|を)?(?:好き|重視)"
)
_JA_LIKE = re.compile(r"(?P<value>[^、。！？]{1,50}?)(?:が|を)?(?:好き|好む|重視する)")
_EN_LIKE = re.compile(
    r"\bI\s+(?:really\s+)?(?:like|love|prefer)\s+(?P<value>[^.!?]{1,80})", re.I
)
_JA_GOAL = re.compile(r"(?P<value>[^、。！？]{1,60}?)(?:したい|を目指している|が目標)")
_EN_GOAL = re.compile(r"\bI\s+(?:want|hope|plan)\s+to\s+(?P<value>[^.!?]{1,80})", re.I)
_JA_RETRACT = re.compile(
    r"(?P<value>[^、。！？]{1,50}?)(?:という話|と言ったの)?は(?:勘違い|間違い|誤り)"
)
_EN_RETRACT = re.compile(
    r"(?:I was wrong about|It was wrong that I liked)\s+(?P<value>[^.!?]{1,80})", re.I
)


class _CandidateBase(TypedDict):
    subject_id: str
    recorded_at: datetime
    evidence_episode_ids: tuple[str, ...]
    evidence_event_ids: tuple[str, ...]


def _scope_and_value(raw: str) -> tuple[str, str]:
    value = raw.strip(" 、,は")
    for marker in ("では", "のときは", "の時は", "when "):
        if marker in value:
            left, right = value.split(marker, 1)
            if left and right:
                return left.strip(), right.strip()
    return "general", value


class RuleBasedStateExtractor:
    """Conservative explicit-state extractor with a small inferred preference baseline."""

    def extract(
        self, episode: Episode, events: list[Event] | tuple[Event, ...]
    ) -> list[StateCandidate]:
        candidates: list[StateCandidate] = []
        for event in events:
            if event.subject_id != "user" or event.modality == Modality.HYPOTHETICAL:
                continue
            text = event.event_text
            base: _CandidateBase = dict(
                subject_id="user",
                recorded_at=event.observed_at,
                evidence_episode_ids=(episode.episode_id,),
                evidence_event_ids=(event.event_id,),
            )
            change = _JA_CHANGE.search(text)
            if change:
                scope, value = _scope_and_value(change.group("new"))
                candidates.append(
                    StateCandidate(
                        **base,
                        attribute="preference",
                        value=value,
                        scope=scope,
                        assertion_type=AssertionType.EXPLICIT,
                        change_cue=True,
                        multi_valued=False,
                    )
                )
                continue
            retract = _JA_RETRACT.search(text) or _EN_RETRACT.search(text)
            if retract:
                scope, value = _scope_and_value(retract.group("value"))
                candidates.append(
                    StateCandidate(
                        **base,
                        attribute="preference",
                        value=value,
                        scope=scope,
                        assertion_type=AssertionType.EXPLICIT,
                        retraction_cue=True,
                    )
                )
                continue
            match = _JA_LIKE.search(text) or _EN_LIKE.search(text)
            if match:
                scope, value = _scope_and_value(match.group("value"))
                candidates.append(
                    StateCandidate(
                        **base,
                        attribute="preference",
                        value=value,
                        scope=scope,
                        assertion_type=AssertionType.EXPLICIT,
                    )
                )
                continue
            goal = _JA_GOAL.search(text) or _EN_GOAL.search(text)
            if goal:
                scope, value = _scope_and_value(goal.group("value"))
                candidates.append(
                    StateCandidate(
                        **base,
                        attribute="goal",
                        value=value,
                        scope=scope,
                        assertion_type=AssertionType.EXPLICIT,
                        multi_valued=True,
                    )
                )
                continue
            lowered = text.lower()
            if ("駅近" in text or "near the station" in lowered) and (
                "泊ま" in text or "stayed" in lowered
            ):
                candidates.append(
                    StateCandidate(
                        **base,
                        attribute="hotel_location_preference",
                        value="near_station",
                        scope="travel",
                        assertion_type=AssertionType.INFERRED,
                        confidence=0.55,
                        multi_valued=False,
                    )
                )
        return candidates
