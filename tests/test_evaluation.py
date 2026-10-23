from relchain.evaluation import evidence_recall, relation_statistics, token_f1
from relchain.models import Relation, RelationCategory


def test_token_f1_and_evidence_recall() -> None:
    assert token_f1("駅近のホテル", "駅近ホテル") > 0.7
    assert evidence_recall(("t1", "t2"), ("t2", "t3")) == 0.5
    assert evidence_recall(("t1",), ()) is None


def test_relation_statistics() -> None:
    relations = [
        Relation(
            edge_id="r1",
            source_episode_id="p1",
            target_episode_id="p2",
            mode="hybrid",
            category=RelationCategory.TEMPORAL,
            relation_text="planning_preceded_booking",
        ),
        Relation(
            edge_id="r2",
            source_episode_id="p2",
            target_episode_id="p3",
            mode="hybrid",
            category=RelationCategory.TEMPORAL,
            relation_text="planning_preceded_booking",
        ),
    ]
    stats = relation_statistics(relations)
    assert stats["total_edges"] == 2
    assert stats["unique_relation_texts"] == 1
    assert stats["single_use_relation_texts"] == 0
