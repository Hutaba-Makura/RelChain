from datetime import datetime, timedelta, timezone

import pytest

from relchain.config import MemoryConfig, RelationMode
from relchain.embeddings import HashingEmbedder
from relchain.models import Episode, Event, SourceSpan
from relchain.relations import RelationBuilder


def make_episode(index: int, embedder: HashingEmbedder) -> tuple[Episode, Event]:
    observed = datetime(2026, 10, 20, tzinfo=timezone.utc) + timedelta(days=index)
    event = Event(
        event_id=f"e{index}",
        conversation_id="c1",
        turn_id=f"t{index}",
        speaker_id="user",
        subject_id="user",
        event_text=f"旅行 ホテル 予約 {index}",
        source_span=SourceSpan(turn_id=f"t{index}", start=0, end=10),
        observed_at=observed,
    )
    episode = Episode(
        episode_id=f"p{index}",
        conversation_id="c1",
        title=f"旅行 {index}",
        summary=event.event_text,
        event_ids=(event.event_id,),
        source_turn_ids=(event.turn_id,),
        entity_ids=("entity:旅行",),
        topic="旅行",
        observed_start=observed,
        observed_end=observed,
        boundary_reason="test",
        embedding=embedder.embed(event.event_text),
    )
    return episode, event


@pytest.mark.parametrize("mode", list(RelationMode))
def test_relation_modes_keep_common_topology(mode: RelationMode) -> None:
    embedder = HashingEmbedder()
    pairs = [make_episode(index, embedder) for index in range(3)]
    episodes = [item[0] for item in pairs]
    events = {item[1].event_id: item[1] for item in pairs}
    relations = RelationBuilder(
        embedder,
        MemoryConfig(relation_mode=mode, neighbors_per_node=1, semantic_threshold=0.0),
    ).build(episodes, events)

    assert relations
    assert all(relation.mode == mode.value for relation in relations)
    if mode == RelationMode.FREE:
        assert all(relation.category is None for relation in relations)
    else:
        assert all(relation.category is not None for relation in relations)


def test_all_modes_use_same_source_target_pairs() -> None:
    embedder = HashingEmbedder()
    pairs = [make_episode(index, embedder) for index in range(3)]
    episodes = [item[0] for item in pairs]
    events = {item[1].event_id: item[1] for item in pairs}
    topologies = []
    for mode in RelationMode:
        relations = RelationBuilder(
            embedder,
            MemoryConfig(
                relation_mode=mode, neighbors_per_node=2, semantic_threshold=0.0
            ),
        ).build(episodes, events)
        topologies.append(
            {(item.source_episode_id, item.target_episode_id) for item in relations}
        )
    assert topologies[0] == topologies[1] == topologies[2]
