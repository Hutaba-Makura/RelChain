import json

from relchain.answering import EvidenceExtractiveAnswerGenerator
from relchain.config import RelationMode
from relchain.locomo import LoCoMoRunner, load_locomo_sample


def test_load_and_run_single_locomo_conversation(tmp_path) -> None:
    dataset = tmp_path / "locomo.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "sample_id": 0,
                    "conversation": {
                        "speaker_a": "Alice",
                        "speaker_b": "Bob",
                        "session_1_date_time": "1:56 pm on 8 May, 2023",
                        "session_1": [
                            {
                                "speaker": "Alice",
                                "dia_id": "D1:1",
                                "text": "I moved from New York last month.",
                            },
                            {
                                "speaker": "Bob",
                                "dia_id": "D1:2",
                                "text": "Welcome!",
                                "blip_caption": "a city skyline",
                            },
                        ],
                    },
                    "qa": [
                        {
                            "question": "Where did Alice move from?",
                            "answer": "New York",
                            "evidence": ["D1:1"],
                            "category": 1,
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    turns, questions, sample_id = load_locomo_sample(dataset, 0)
    assert sample_id == "0"
    assert [item.turn_id for item in turns] == ["D1:1", "D1:2"]
    assert "Image description" in turns[1].text

    runner = LoCoMoRunner(
        database=tmp_path / "memory.db",
        relation_mode=RelationMode.HYBRID,
        reader_name="test-reader",
        answer_generator=EvidenceExtractiveAnswerGenerator(),
    )
    report = runner.prepare(turns)
    assert not report.failures
    result = runner.answer_questions(
        questions=questions,
        output_path=tmp_path / "results.jsonl",
        run_id="test-run",
    )
    assert result["new_results"] == 1
    assert result["failures"] == []
