import json
from datetime import datetime, timedelta, timezone

import pytest

from relchain.artifacts import build_mode_relations, freeze_common, load_common, materialize_common
from relchain.cli import main
from relchain.config import MemoryConfig, RelationMode
from relchain.models import ConversationTurn
from relchain.pipeline import MemorySystem
from relchain.quick_eval import load_set, select_set


def make_dataset(path, *, samples=2, per_category=5):
    payload = []
    for sample_index in range(samples):
        qa = []
        for category in (1, 2, 4, 5):
            for index in range(per_category):
                qa.append({"question": f"sample {sample_index} category {category} question {index}?",
                    "answer": "旅行" if category != 5 else None,
                    "evidence": ["t1", "t2"] if category == 1 else (["t1"] if category != 5 else []),
                    "category": category})
        payload.append({"sample_id": sample_index,
            "conversation": {
                "session_1_date_time": "1:00 pm on 1 May, 2023",
                "session_1": [{"dia_id": "t1", "speaker": "A", "text": "旅行を計画した。"}],
                "session_2_date_time": "1:00 pm on 2 May, 2023",
                "session_2": [{"dia_id": "t2", "speaker": "A", "text": "ホテルを予約した。"}],
            }, "qa": qa})
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_fixed_selection_seed_balance_and_shortage(tmp_path):
    dataset = tmp_path / "data.json"
    make_dataset(dataset, per_category=3)
    first = select_set(dataset, profile="dev60", seed=17)
    second = select_set(dataset, profile="dev60", seed=17)
    assert first["content_hash"] == second["content_hash"]
    assert first["selected_count"] == 24
    assert first["shortage_by_category"]["multi_hop"] > 0
    assert len({(row["sample_id"], row["question_id"]) for row in first["questions"]}) == 24
    assert all(row["question_id"].startswith("locomo-q_") for row in first["questions"])
    assert all(row["evidence_session_count"] == 2 for row in first["questions"] if row["category"] == 1)
    assert {int(index) for index in first["actual_by_sample"]} == {0, 1}
    different = select_set(dataset, profile="dev60", seed=18)
    assert different["content_hash"] != first["content_hash"]


def test_staged_smoke_reuses_kg_and_skips_answer_calls(tmp_path, monkeypatch, capsys):
    dataset = tmp_path / "data.json"
    make_dataset(dataset, per_category=2)
    selection = tmp_path / "set.json"
    artifacts = tmp_path / "artifacts"
    modes = tmp_path / "modes"
    retrieval = tmp_path / "retrieval.jsonl"
    answers = tmp_path / "answers.jsonl"
    scores = tmp_path / "scores.jsonl"
    main(["locomo-set", "--dataset", str(dataset), "--profile", "smoke",
          "--samples", "0", "1", "--output", str(selection)])
    assert load_set(selection, dataset)["selected_count"] == 10
    build_common = ["locomo-build-common", "--dataset", str(dataset),
                    "--set", str(selection), "--artifact-dir", str(artifacts)]
    main(build_common)
    main(["locomo-build-mode", "--set", str(selection), "--artifact-dir", str(artifacts),
          "--mode-db-dir", str(modes), "--relation-mode", "fixed"])

    original_ingest = MemorySystem.ingest
    def forbidden_ingest(*args, **kwargs):
        raise AssertionError("KG rebuild was attempted")
    monkeypatch.setattr(MemorySystem, "ingest", forbidden_ingest)
    main(build_common)
    other_selection = tmp_path / "set-other.json"
    main(["locomo-set", "--dataset", str(dataset), "--profile", "smoke",
          "--samples", "0", "1", "--seed", "99", "--output", str(other_selection)])
    main(["locomo-build-common", "--dataset", str(dataset), "--set", str(other_selection),
          "--artifact-dir", str(artifacts)])
    larger_selection = tmp_path / "set-larger.json"
    main(["locomo-set", "--dataset", str(dataset), "--profile", "dev60",
          "--samples", "0", "1", "--output", str(larger_selection)])
    assert load_set(larger_selection)["selected_count"] > load_set(selection)["selected_count"]
    main(["locomo-build-common", "--dataset", str(dataset), "--set", str(larger_selection),
          "--artifact-dir", str(artifacts)])

    def forbidden_answer(*args, **kwargs):
        raise AssertionError("answer LLM called during retrieval")
    monkeypatch.setattr("relchain.llm.OpenAICompatibleTransport.complete", forbidden_answer)

    retrieve = ["locomo-retrieve", "--set", str(selection), "--artifact-dir", str(artifacts),
                "--mode-db-dir", str(modes), "--relation-mode", "fixed", "--output", str(retrieval)]
    main(retrieve)
    main(retrieve)
    main([*retrieve[:-1], str(tmp_path / "retrieval-zero-hop.jsonl"), "--max-hops", "0"])

    calls = []
    def complete(self, prompt):
        calls.append(self.model)
        ids = json.loads(prompt.split("allowed_evidence_event_ids=")[1].splitlines()[0])
        return json.dumps({"answer": "旅行", "evidence_event_ids": ids[:1]})
    monkeypatch.setattr("relchain.llm.OpenAICompatibleTransport.complete", complete)
    answer = ["locomo-answer", "--set", str(selection), "--retrieval", str(retrieval),
              "--output", str(answers), "--base-url", "http://localhost:8000/v1",
              "--model", "reader-a"]
    main(answer)
    first_calls = len(calls)
    assert first_calls == 10
    main(answer)
    assert len(calls) == first_calls
    other_answers = tmp_path / "answers-other.jsonl"
    main([*answer[:answer.index("--output") + 1], str(other_answers),
          "--base-url", "http://localhost:8000/v1", "--model", "reader-b"])
    assert len(calls) == first_calls * 2
    assert set(calls) == {"reader-a", "reader-b"}

    main(["locomo-score", "--set", str(selection), "--answers", str(answers),
          "--output", str(scores)])
    main(["locomo-score", "--set", str(selection), "--answers", str(answers),
          "--output", str(scores)])
    other_scores = tmp_path / "scores-other.jsonl"
    main(["locomo-score", "--set", str(selection), "--answers", str(other_answers),
          "--output", str(other_scores)])
    capsys.readouterr()
    main(["locomo-compare", "--left", str(scores), "--right", str(other_scores)])
    comparison = json.loads(capsys.readouterr().out)
    assert comparison["conditions"]["same_kg"]
    assert comparison["conditions"]["same_search"]
    assert not comparison["conditions"]["same_reader"]
    assert len(calls) == first_calls * 2
    assert "by_conversation" in comparison
    monkeypatch.setattr(MemorySystem, "ingest", original_ingest)


def test_relation_generation_resumes_after_partial_failure(tmp_path):
    now = datetime(2026, 10, 30, tzinfo=timezone.utc)
    turns = [ConversationTurn(conversation_id="c", turn_id=f"t{i}", speaker_id="user",
        observed_at=now + timedelta(days=i), text=f"旅行の予定 {i}。") for i in range(3)]
    common = MemorySystem(tmp_path / "common.db", MemoryConfig(relation_mode=RelationMode.FIXED))
    common.ingest(turns, build_relations=False)
    artifact = freeze_common(common, tmp_path / "common.json")

    class InterruptedGenerator:
        def __init__(self):
            self.calls = 0
            self.fail = True
        def free_relation(self, source, target, category_hint):
            self.calls += 1
            if self.fail and self.calls == 2:
                raise RuntimeError("interrupted")
            return "planning_to_booking", "link", 0.7
        def hybrid_relation(self, source, target, category):
            return self.free_relation(source, target, category)

    generator = InterruptedGenerator()
    system = MemorySystem(tmp_path / "free.db", MemoryConfig(relation_mode=RelationMode.FREE),
                          relation_text_generator=generator)
    with pytest.raises(RuntimeError, match="interrupted"):
        build_mode_relations(system, artifact, generation_signature="v1")
    saved_before = len(system.store.list_relations())
    assert saved_before == 1
    assert system.store.get_metadata("relation_artifact_hash") is None
    generator.fail = False
    build_mode_relations(system, artifact, generation_signature="v1")
    assert generator.calls == len(artifact["candidates"]) + 1
    assert len(system.store.list_relations()) == len(artifact["candidates"])
    assert system.store.get_metadata("relation_artifact_hash")


def test_incomplete_mode_kg_cannot_be_retrieved(tmp_path):
    dataset = tmp_path / "data.json"
    make_dataset(dataset, samples=1, per_category=1)
    selected = tmp_path / "set.json"
    artifact_dir = tmp_path / "artifact"
    modes = tmp_path / "modes"
    main(["locomo-set", "--dataset", str(dataset), "--profile", "smoke", "--output", str(selected)])
    main(["locomo-build-common", "--dataset", str(dataset), "--set", str(selected),
          "--artifact-dir", str(artifact_dir)])
    sample_id = load_set(selected)["sample_ids"][0]
    from relchain.quick_commands import _sample_path
    artifact = load_common(_sample_path(artifact_dir, sample_id, "common.json"))
    system = MemorySystem(_sample_path(modes, sample_id, "free.db"),
                          MemoryConfig(relation_mode=RelationMode.FREE))
    materialize_common(system, artifact)
    with pytest.raises(ValueError, match="incomplete"):
        main(["locomo-retrieve", "--set", str(selected), "--artifact-dir", str(artifact_dir),
              "--mode-db-dir", str(modes), "--relation-mode", "free",
              "--output", str(tmp_path / "retrieval.jsonl")])


def test_relation_prompt_version_changes_mode_key(monkeypatch):
    from relchain.cli import build_parser
    from relchain.quick_commands import _relation_signature
    from relchain.llm import LLMRelationTextGenerator
    args = build_parser().parse_args(["locomo-build-mode", "--set", "x",
        "--artifact-dir", "a", "--mode-db-dir", "b", "--relation-mode", "free",
        "--relation-generation", "llm", "--relation-model", "writer",
        "--relation-base-url", "http://localhost:8000/v1"])
    before = _relation_signature(args)
    monkeypatch.setattr(LLMRelationTextGenerator, "PROMPT_VERSION", "changed")
    assert _relation_signature(args) != before
