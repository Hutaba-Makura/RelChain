import json
from datetime import datetime, timedelta, timezone

import pytest

from relchain.artifacts import build_mode_relations, freeze_common, load_common
from relchain.config import MemoryConfig, RelationMode
from relchain.cli import main
from relchain.evaluation import EvaluationLogger, EvaluationRecord, all_evidence_retrieved, evidence_recall
from relchain.llm import LLMAnswerGenerator, ValidatedJSONCaller
from relchain.locomo import LoCoMoRunner
from relchain.models import ConversationTurn, Episode, Event, EvidenceItem, QueryIntent, Relation, RetrievalResult, RetrievalTrace, SourceSpan
from relchain.pipeline import MemorySystem
from relchain.retrieval import MemoryRetriever
from relchain.embeddings import HashingEmbedder
from relchain.storage import SQLiteMemoryStore
from relchain.tokens import ApproximateTokenCounter


class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        return self.responses.pop(0)


def retrieval_fixture():
    now = datetime(2026, 10, 30, tzinfo=timezone.utc)
    evidence = tuple(EvidenceItem(
        conversation_id="c", turn_id=f"t{i}", event_id=f"e{i}", event_ids=(f"e{i}",),
        episode_id=f"p{i}", speaker_id="user", observed_at=now,
        raw_text=f"fact {i}", source_start=0, source_end=6,
    ) for i in (1, 2, 3))
    context = "\n".join(
        f"[TURN t{i} EVENTS e{i} {now.isoformat()}] fact {i}" for i in (1, 2)
    )
    trace = RetrievalTrace(
        question="What happened?", intent=QueryIntent(), seed_episode_ids=(),
        state_ids=(), traversed_edge_ids=(), retrieved_episode_ids=(),
        evidence_event_ids=("e1", "e2", "e3"), evidence_turn_ids=("t1", "t2", "t3"),
        relation_descriptions=(), context_event_ids=("e1", "e2"),
        context_turn_ids=("t1", "t2"),
    )
    return RetrievalResult(question="What happened?", states=(), episodes=(),
                           evidence=evidence, trace=trace, context=context)


def test_three_evidence_stages_and_run_aggregation(tmp_path):
    required = ("t1", "t2")
    stages = (("t1", "t2", "t3"), ("t1", "t2"), ("t1",))
    assert [evidence_recall(ids, required) for ids in stages] == [1.0, 1.0, 0.5]
    assert [all_evidence_retrieved(ids, required) for ids in stages] == [True, True, False]
    logger = EvaluationLogger(tmp_path / "results.jsonl")
    def record(run, q, required_ids):
        return EvaluationRecord(
            run_id=run, relation_mode="fixed", question_id=q, category="1",
            question="q", expected_answer="a", predicted_answer="a", f1=1.0,
            required_turn_ids=required_ids,
            retrieved_evidence_recall=evidence_recall(stages[0], required_ids),
            retrieved_all_required_evidence=all_evidence_retrieved(stages[0], required_ids),
            context_evidence_recall=evidence_recall(stages[1], required_ids),
            context_all_required_evidence=all_evidence_retrieved(stages[1], required_ids),
            citation_evidence_recall=evidence_recall(stages[2], required_ids),
            citation_all_required_evidence=all_evidence_retrieved(stages[2], required_ids),
        )
    for q, required_ids in (("a", required), ("b", required), ("c", ())):
        logger.append(record("run-a", q, required_ids))
    logger.append(record("run-b", "x", required))
    logger.append_failure(run_id="run-a", question_id="d", stage="query", exception_type="ValueError")
    result = logger.aggregate("run-a", total_questions=5)
    assert result["overall"]["retrieved_evaluable_count"] == 2
    assert result["overall"]["retrieved_all_required_rate"] == 1.0
    assert result["overall"]["citation_all_required_rate"] == 0.0
    assert (result["successful_questions"], result["failed_questions"], result["unprocessed_questions"]) == (3, 1, 1)
    assert logger.aggregate("run-b")["successful_questions"] == 1


def test_multiple_citations_and_input_only_validation():
    transport = FakeTransport([json.dumps({"answer": "both", "evidence_event_ids": ["e1", "e2"]})])
    answer = LLMAnswerGenerator(ValidatedJSONCaller(transport)).generate(retrieval_fixture())
    assert answer.evidence_event_ids == ("e1", "e2")
    assert answer.evidence_turn_ids == ("t1", "t2")
    assert answer.citation_status == "valid"
    assert "e3" not in transport.prompts[0].split("allowed_evidence_event_ids=")[1].splitlines()[0]

    invalid = FakeTransport([json.dumps({"answer": "wrong", "evidence_event_ids": ["e3"]})] * 2)
    answer = LLMAnswerGenerator(ValidatedJSONCaller(invalid)).generate(retrieval_fixture())
    assert answer.evidence_event_ids == ()
    assert answer.model_cited_event_ids == ("e3",)
    assert answer.invalid_cited_event_ids == ("e3",)
    assert answer.citation_status == "citation_validation_failed"

    missing = FakeTransport([json.dumps({"answer": "uncited", "evidence_event_ids": []})] * 2)
    answer = LLMAnswerGenerator(ValidatedJSONCaller(missing)).generate(retrieval_fixture())
    assert answer.citation_status == "citation_validation_failed"
    assert answer.evidence_event_ids == ()

    abstain = FakeTransport([json.dumps({"answer": "", "evidence_event_ids": [], "status": "information_not_found"})])
    answer = LLMAnswerGenerator(ValidatedJSONCaller(abstain)).generate(retrieval_fixture())
    assert answer.citation_status == "valid_abstention"


def test_legacy_results_do_not_become_retrieval_metrics(tmp_path):
    path = tmp_path / "legacy.jsonl"
    path.write_text(json.dumps({
        "run_id": "old", "relation_mode": "fixed", "question_id": "q", "category": "1",
        "question": "q", "expected_answer": "a", "predicted_answer": "a", "f1": 1.0,
        "required_turn_ids": ["t1"], "retrieved_turn_ids": ["t1"],
        "evidence_recall": 1.0, "all_evidence_retrieved": True,
        "retrieved_episode_ids": [], "traversed_edge_ids": [],
        "information_not_found": False,
    }) + "\n", encoding="utf-8")
    record = EvaluationLogger(path).load()[0]
    assert record.schema_version == 1
    assert record.retrieved_turn_ids == ("t1",)
    assert record.retrieved_evidence_recall is None
    assert EvaluationLogger(path).aggregate("old")["overall"]["retrieved_all_required_rate"] is None


def test_frozen_common_graph_and_no_regeneration(tmp_path):
    now = datetime(2026, 10, 30, tzinfo=timezone.utc)
    turns = [ConversationTurn(conversation_id="c", turn_id=f"t{i}", speaker_id="user",
        observed_at=now + timedelta(days=i), text=f"旅行の予定 {i}。") for i in range(3)]
    common = MemorySystem(tmp_path / "common.db", MemoryConfig(relation_mode=RelationMode.FIXED))
    common.ingest(turns)
    artifact_path = tmp_path / "common.json"
    frozen = freeze_common(common, artifact_path)
    assert load_common(artifact_path)["content_hash"] == frozen["content_hash"]

    class Generator:
        def __init__(self):
            self.calls = []
        def free_relation(self, source, target, category_hint):
            self.calls.append("free")
            return "planning_to_booking", "link", 0.1
        def hybrid_relation(self, source, target, category):
            self.calls.append("hybrid")
            return "planning_to_booking", "link", 0.9

    hashes = []
    for mode in RelationMode:
        generator = Generator()
        system = MemorySystem(tmp_path / f"{mode}.db", MemoryConfig(relation_mode=mode), relation_text_generator=generator)
        digest = build_mode_relations(system, frozen, generation_signature="same")
        count = len(generator.calls)
        build_mode_relations(system, frozen, generation_signature="same")
        system.query("旅行は？")
        assert len(generator.calls) == count
        assert (count == 0) == (mode == RelationMode.FIXED)
        assert system.store.get_metadata("common_artifact_hash") == frozen["content_hash"]
        relations = system.store.list_relations()
        assert relations
        assert all(item.confidence == item.candidate_confidence for item in relations)
        if mode == RelationMode.FREE:
            assert all(item.category is None for item in relations)
        hashes.append({item.candidate_id for item in relations})
        with pytest.raises(ValueError, match="conditions differ"):
            build_mode_relations(system, frozen, generation_signature="changed")
    assert hashes[0] == hashes[1] == hashes[2]


class CharacterCounter:
    name = "fixture-character-counter"
    exact = True

    def count(self, text):
        return len(text)


def test_context_uses_whole_turns_and_skips_oversized_first_candidate(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    config = MemoryConfig(context_token_budget=100, include_relation_context=False)
    retriever = MemoryRetriever(store, HashingEmbedder(), config, CharacterCounter())
    now = datetime(2026, 10, 30, tzinfo=timezone.utc)
    evidence = [EvidenceItem(conversation_id="c", turn_id=f"t{i}", event_id=f"e{i}",
        event_ids=(f"e{i}",), episode_id=f"p{i}", speaker_id="user",
        observed_at=now, raw_text=text, source_start=0, source_end=len(text))
        for i, text in ((1, "x" * 200), (2, "短い根拠"))]
    context, ids = retriever._context("何？", (), (), evidence, ())
    assert "t1" not in context and "t2" in context
    assert ids["context_turn_ids"] == ("t2",)
    assert ids["context_event_ids"] == ("e2",)
    assert ids["excluded_turn_ids"] == ("t1",)
    assert ids["memory_token_count"] <= config.context_token_budget
    assert ids["token_count_method"] == "fixture-character-counter"

    relation = Relation(edge_id="r", source_episode_id="p1", target_episode_id="p2",
        mode="free", relation_text="planning_to_booking")
    _, ids = retriever._context("何？", (), (), (), (relation,))
    assert ids["context_relation_ids"] == ()
    enabled = MemoryRetriever(store, HashingEmbedder(),
        config.model_copy(update={"include_relation_context": True}), CharacterCounter())
    context, ids = enabled._context("何？", (), (), (), (relation,))
    assert "[RELATION r]" in context
    assert ids["context_relation_ids"] == ("r",)


def test_backtrack_retains_all_events_once_per_turn_and_zero_hop(tmp_path):
    system = MemorySystem(tmp_path / "memory.db", MemoryConfig(max_hops=0, max_episodes=1))
    now = datetime(2026, 10, 30, tzinfo=timezone.utc)
    turn = ConversationTurn(conversation_id="c", turn_id="t", speaker_id="user",
        observed_at=now, text="alpha beta")
    system.store.save_turn(turn)
    events = [Event(event_id=f"e{i}", conversation_id="c", turn_id="t", speaker_id="user",
        subject_id="user", event_text=f"fact {i}", source_span=SourceSpan(turn_id="t", start=0, end=5),
        observed_at=now) for i in (1, 2)]
    system.store.save_events(events)
    episode = Episode(episode_id="p", conversation_id="c", title="alpha", summary="alpha beta",
        event_ids=("e1", "e2"), source_turn_ids=("t",), observed_start=now,
        observed_end=now, boundary_reason="test", embedding=system.embedder.embed("alpha beta"))
    system.store.save_episodes([episode])
    backtracked = system.retriever._backtrack([episode])
    assert len(backtracked) == 1
    assert backtracked[0].event_ids == ("e1", "e2")
    result = system.retriever.retrieve("alpha")
    assert result.trace.evidence_event_ids == ("e1", "e2")
    assert result.trace.traversed_edge_ids == ()
    assert len(result.episodes) == 1


def test_full_prompt_budget_reduces_allowed_ids_and_rejects_fixed_overflow():
    class ChatCounter(CharacterCounter):
        def count_chat(self, system, user):
            return len(system) + len(user)

    source = retrieval_fixture()
    counter = ChatCounter()
    probe = LLMAnswerGenerator(ValidatedJSONCaller(FakeTransport([])), token_counter=counter)
    empty = source.model_copy(update={
        "context": "",
        "trace": source.trace.model_copy(update={"context_event_ids": (), "context_turn_ids": ()}),
    })
    retry_suffix = "\nYour previous output was invalid. Correct the citations using only allowed_evidence_event_ids."
    fixed_size = len(probe.SYSTEM_MESSAGE) + len(probe._prompt(empty) + retry_suffix)
    transport = FakeTransport([json.dumps({"answer": "first", "evidence_event_ids": ["e1"]})])
    generator = LLMAnswerGenerator(ValidatedJSONCaller(transport), token_counter=counter,
                                   full_input_token_budget=fixed_size + 80)
    answer = generator.generate(source)
    assert answer.prompt_token_count <= fixed_size + 80
    assert answer.retrieval.trace.context_event_ids == ("e1",)
    assert answer.evidence_event_ids == ("e1",)
    assert "e2" not in transport.prompts[0].split("allowed_evidence_event_ids=")[1].splitlines()[0]
    too_small = LLMAnswerGenerator(ValidatedJSONCaller(FakeTransport([])), token_counter=counter,
                                   full_input_token_budget=fixed_size - 1)
    with pytest.raises(ValueError, match="fixed answer prompt"):
        too_small.generate(source)


def test_end_to_end_three_modes_save_aggregate_and_resume(tmp_path):
    now = datetime(2026, 10, 30, tzinfo=timezone.utc)
    turns = [ConversationTurn(conversation_id="c", turn_id=f"t{i}", speaker_id="user",
        observed_at=now + timedelta(days=i), text=f"旅行の予定 {i}。") for i in range(2)]
    common = MemorySystem(tmp_path / "common.db", MemoryConfig(relation_mode=RelationMode.FIXED))
    common.ingest(turns)
    artifact = freeze_common(common, tmp_path / "common.json")

    class DynamicTransport:
        def __init__(self):
            self.prompts = []
        def complete(self, prompt):
            self.prompts.append(prompt)
            allowed = json.loads(prompt.split("allowed_evidence_event_ids=")[1].splitlines()[0])
            return json.dumps({"answer": "旅行", "evidence_event_ids": allowed[:1]})

    output = tmp_path / "results.jsonl"
    for mode in RelationMode:
        transport = DynamicTransport()
        runner = LoCoMoRunner(database=tmp_path / f"{mode}.db", relation_mode=mode,
            reader_name="fake-reader", answer_generator=LLMAnswerGenerator(ValidatedJSONCaller(transport)))
        build_mode_relations(runner.system, artifact)
        questions = [{"question": "旅行の予定は？", "answer": "旅行", "evidence": ["t0"], "category": 1}]
        manifest = {"fixture": "small", "mode": mode.value}
        first = runner.answer_questions(questions=questions, output_path=output,
                                        run_id=f"run-{mode.value}", manifest=manifest)
        second = runner.answer_questions(questions=questions, output_path=output,
                                         run_id=f"run-{mode.value}", manifest=manifest)
        assert first["new_results"] == 1
        assert second["new_results"] == 0
        assert second["metrics"]["successful_questions"] == 1
        assert len(transport.prompts) == 1
        with pytest.raises(ValueError, match="manifest mismatch"):
            runner.answer_questions(questions=questions, output_path=output,
                run_id=f"run-{mode.value}", manifest={"fixture": "changed"})
    assert len(EvaluationLogger(output).load()) == 3


def test_cli_connects_llm_relation_generator_only_for_free_and_hybrid(tmp_path, monkeypatch, capsys):
    dataset = tmp_path / "locomo.json"
    dataset.write_text(json.dumps([{
        "sample_id": 0,
        "conversation": {
            "session_1_date_time": "1:00 pm on 1 May, 2023",
            "session_1": [{"dia_id": "t1", "speaker": "A", "text": "旅行を計画した。"}],
            "session_2_date_time": "1:00 pm on 2 May, 2023",
            "session_2": [{"dia_id": "t2", "speaker": "A", "text": "ホテルを予約した。"}],
        },
        "qa": [{"question": "旅行とホテルの予定は？", "answer": "旅行", "evidence": ["t1"]}],
    }], ensure_ascii=False), encoding="utf-8")
    calls = []

    def complete(self, prompt):
        if prompt.startswith("Generate a short"):
            calls.append("relation")
            assert "fixed upper category" not in prompt or self.model == "relation-model"
            return json.dumps({"relation": "planning_to_booking", "description": "link", "confidence": 0.5})
        calls.append("answer")
        allowed = json.loads(prompt.split("allowed_evidence_event_ids=")[1].splitlines()[0])
        return json.dumps({"answer": "旅行", "evidence_event_ids": allowed[:1]})

    monkeypatch.setattr("relchain.llm.OpenAICompatibleTransport.complete", complete)
    common = tmp_path / "common.json"
    output = tmp_path / "answers.jsonl"
    for mode in RelationMode:
        before = calls.count("relation")
        args = ["locomo", "--dataset", str(dataset), "--db", str(tmp_path / f"{mode}.db"),
            "--output", str(output), "--common-artifact", str(common),
            "--relation-mode", mode.value, "--base-url", "http://localhost:8000/v1",
            "--model", "reader-model", "--relation-generation", "deterministic" if mode == RelationMode.FIXED else "llm",
            "--relation-model", "relation-model"]
        main(args)
        assert (calls.count("relation") == before) == (mode == RelationMode.FIXED)
    assert len(EvaluationLogger(output).load()) == 3
    assert common.exists()
    assert "new_results" in capsys.readouterr().out
