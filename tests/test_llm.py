import json
from datetime import datetime, timezone

from relchain.llm import (
    LLMEventExtractor,
    LLMRelationTextGenerator,
    ValidatedJSONCaller,
)
from relchain.models import ConversationTurn, Episode, RelationCategory


class FakeTransport:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0)


def test_event_adapter_retries_once_and_validates_grounding() -> None:
    text = "駅近が好きです。"
    invalid = json.dumps(
        {
            "events": [
                {
                    "event_text": "駅近が好き",
                    "source_start": 0,
                    "source_end": 999,
                    "subject_id": "user",
                }
            ]
        },
        ensure_ascii=False,
    )
    valid = json.dumps(
        {
            "events": [
                {
                    "event_text": "駅近が好き",
                    "source_start": 0,
                    "source_end": len(text),
                    "subject_id": "user",
                    "modality": "actual",
                    "polarity": "positive",
                }
            ]
        },
        ensure_ascii=False,
    )
    transport = FakeTransport([invalid, valid])
    extractor = LLMEventExtractor(ValidatedJSONCaller(transport))
    turn = ConversationTurn(
        conversation_id="c1",
        turn_id="t1",
        speaker_id="user",
        text=text,
        observed_at=datetime(2026, 10, 23, tzinfo=timezone.utc),
    )

    events = extractor.extract(turn)
    assert len(events) == 1
    assert events[0].source_span.end == len(text)
    assert len(transport.prompts) == 2
    assert "previous output was invalid" in transport.prompts[1]


def test_relation_adapter_rejects_generic_name_then_accepts_specific() -> None:
    transport = FakeTransport(
        [
            '{"relation":"related_to","description":"x","confidence":0.8}',
            '{"relation":"planning_preceded_booking","description":"x","confidence":0.8}',
        ]
    )
    generator = LLMRelationTextGenerator(ValidatedJSONCaller(transport))
    now = datetime(2026, 10, 23, tzinfo=timezone.utc)
    source = Episode(
        episode_id="p1",
        conversation_id="c1",
        title="plan",
        summary="planned a trip",
        event_ids=("e1",),
        source_turn_ids=("t1",),
        observed_start=now,
        observed_end=now,
        boundary_reason="test",
    )
    target = source.model_copy(
        update={
            "episode_id": "p2",
            "title": "booking",
            "summary": "booked a hotel",
            "event_ids": ("e2",),
            "source_turn_ids": ("t2",),
        }
    )

    name, _, confidence = generator.hybrid_relation(
        source, target, RelationCategory.TEMPORAL
    )
    assert name == "planning_preceded_booking"
    assert confidence == 0.8
    assert len(transport.prompts) == 2
