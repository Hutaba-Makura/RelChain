import json

from relchain.cli import main


def test_cli_ingest_query_and_inspect(tmp_path, capsys) -> None:
    database = tmp_path / "cli.db"
    source = tmp_path / "turns.jsonl"
    source.write_text(
        json.dumps(
            {
                "conversation_id": "c1",
                "turn_id": "t1",
                "speaker_id": "user",
                "observed_at": "2026-10-23T10:00:00+09:00",
                "text": "旅行では駅近のホテルが好きです。",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    main(["init", "--db", str(database), "--relation-mode", "hybrid"])
    main(["ingest", "--db", str(database), "--input", str(source)])
    main(["query", "--db", str(database), "--question", "旅行の好みは？"])
    main(["inspect", "--db", str(database)])

    output = capsys.readouterr().out
    assert "hybrid" in output
    assert "preference" in output
    assert '"turns": 1' in output
