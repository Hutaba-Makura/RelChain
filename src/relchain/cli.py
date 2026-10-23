from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import MemoryConfig, RelationMode
from .evaluation import relation_statistics
from .models import ConversationTurn
from .pipeline import MemorySystem
from .storage import SQLiteMemoryStore


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _mode_for_database(path: Path, requested: str | None = None) -> RelationMode:
    if requested:
        return RelationMode(requested)
    mode = SQLiteMemoryStore(path).get_metadata("relation_mode")
    return RelationMode(mode or RelationMode.HYBRID.value)


def _system(args: argparse.Namespace) -> MemorySystem:
    mode = _mode_for_database(args.db, getattr(args, "relation_mode", None))
    return MemorySystem(args.db, MemoryConfig(relation_mode=mode))


def command_init(args: argparse.Namespace) -> None:
    system = _system(args)
    _print(
        {
            "database": str(system.store.path),
            "relation_mode": system.config.relation_mode,
        }
    )


def command_ingest(args: argparse.Namespace) -> None:
    system = _system(args)
    turns: list[ConversationTurn] = []
    with args.input.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                turns.append(ConversationTurn.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
    report = system.ingest(turns)
    _print(report.model_dump(mode="json"))


def command_query(args: argparse.Namespace) -> None:
    answer = _system(args).query(args.question)
    payload = (
        answer.model_dump(mode="json")
        if args.trace
        else {
            "answer": answer.text,
            "evidence_turn_ids": answer.evidence_turn_ids,
            "evidence_event_ids": answer.evidence_event_ids,
            "uncertainty": answer.uncertainty,
        }
    )
    _print(payload)


def command_inspect(args: argparse.Namespace) -> None:
    store = SQLiteMemoryStore(args.db)
    _print(
        {
            "database": str(store.path),
            "counts": store.counts(),
            "metadata": {
                key: store.get_metadata(key)
                for key in (
                    "relation_mode",
                    "writer_name",
                    "reader_name",
                    "embedding_name",
                )
            },
        }
    )


def command_export(args: argparse.Namespace) -> None:
    store = SQLiteMemoryStore(args.db)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(store.export_json(), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    _print({"output": str(args.output)})


def command_relation_stats(args: argparse.Namespace) -> None:
    store = SQLiteMemoryStore(args.db)
    _print(relation_statistics(store.list_relations()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="relchain")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="initialize a mode-isolated database")
    init.add_argument("--db", type=Path, required=True)
    init.add_argument(
        "--relation-mode",
        choices=[item.value for item in RelationMode],
        default="hybrid",
    )
    init.set_defaults(handler=command_init)

    ingest = subparsers.add_parser("ingest", help="ingest a JSONL conversation")
    ingest.add_argument("--db", type=Path, required=True)
    ingest.add_argument("--input", type=Path, required=True)
    ingest.add_argument(
        "--relation-mode", choices=[item.value for item in RelationMode]
    )
    ingest.set_defaults(handler=command_ingest)

    query = subparsers.add_parser(
        "query", help="retrieve evidence and answer a question"
    )
    query.add_argument("--db", type=Path, required=True)
    query.add_argument("--question", required=True)
    query.add_argument("--trace", action="store_true")
    query.set_defaults(handler=command_query)

    inspect = subparsers.add_parser("inspect", help="show database counts and versions")
    inspect.add_argument("--db", type=Path, required=True)
    inspect.set_defaults(handler=command_inspect)

    export = subparsers.add_parser(
        "export", help="export latest graph and state records"
    )
    export.add_argument("--db", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.set_defaults(handler=command_export)

    stats = subparsers.add_parser(
        "relation-stats", help="summarize relation vocabulary"
    )
    stats.add_argument("--db", type=Path, required=True)
    stats.set_defaults(handler=command_relation_stats)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
