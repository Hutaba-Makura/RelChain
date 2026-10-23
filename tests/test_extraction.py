from datetime import datetime, timedelta, timezone

from relchain.extraction import RuleBasedEventExtractor, RuleBasedStateExtractor
from relchain.models import ConversationTurn, Episode, Modality, Polarity


def turn(text: str) -> ConversationTurn:
    return ConversationTurn(
        conversation_id="c1",
        turn_id="t1",
        speaker_id="user",
        text=text,
        observed_at=datetime(2026, 10, 23, 9, 0, tzinfo=timezone.utc),
    )


def test_event_extractor_preserves_grounding_and_semantics() -> None:
    item = turn("昨日、ホテルに泊まった。来月引っ越したい。友人はジャズが嫌いです。")
    events = RuleBasedEventExtractor().extract(item)

    assert len(events) == 3
    assert events[0].source_span.start == 0
    assert events[0].event_time_start == item.observed_at - timedelta(days=1)
    assert events[1].modality == Modality.DESIRED
    assert events[2].subject_id == "entity:友人"
    assert events[2].polarity == Polarity.NEGATIVE


def test_state_extractor_does_not_mix_other_subject() -> None:
    item = turn("友人はジャズが好きです。私はロックが好きです。")
    events = RuleBasedEventExtractor().extract(item)
    episode = Episode(
        episode_id="p1",
        conversation_id="c1",
        title="music",
        summary=item.text,
        event_ids=tuple(event.event_id for event in events),
        source_turn_ids=("t1",),
        observed_start=item.observed_at,
        observed_end=item.observed_at,
        boundary_reason="test",
    )
    candidates = RuleBasedStateExtractor().extract(episode, events)

    assert len(candidates) == 1
    assert "ロック" in candidates[0].value
