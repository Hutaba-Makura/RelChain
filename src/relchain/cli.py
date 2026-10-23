from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .config import MemoryConfig, RelationMode
from .evaluation import relation_statistics
from .llm import LLMAnswerGenerator, OpenAICompatibleTransport, ValidatedJSONCaller
from .locomo import LoCoMoRunner, load_locomo_sample
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


def command_locomo(args: argparse.Namespace) -> None:
    api_key = os.getenv(args.api_key_env) if args.api_key_env else None
    transport = OpenAICompatibleTransport(
        base_url=args.base_url,
        model=args.model,
        api_key=api_key,
        timeout_seconds=args.timeout,
        seed=args.seed,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    runner = LoCoMoRunner(
        database=args.db,
        relation_mode=RelationMode(args.relation_mode),
        reader_name=args.model,
        answer_generator=LLMAnswerGenerator(ValidatedJSONCaller(transport)),
        context_token_budget=args.context_tokens,
    )
    turns, questions, sample_id = load_locomo_sample(args.dataset, args.sample)
    report = runner.prepare(turns)
    if args.prepare_only:
        _print(
            {
                "sample_id": sample_id,
                "turns": len(turns),
                "questions": len(questions),
                "ingest": {
                    "new_turns": len(report.turn_ids),
                    "new_events": len(report.event_ids),
                    "affected_episodes": len(report.episode_ids),
                    "relations": len(report.relation_ids),
                    "state_updates": len(report.transitions),
                    "failures": report.failures,
                },
            }
        )
        return
    run_id = f"locomo-{sample_id}-{args.relation_mode}-{args.model}"
    result = runner.answer_questions(
        questions=questions,
        output_path=args.output,
        run_id=run_id,
        max_questions=args.max_questions,
    )
    result["sample_id"] = sample_id
    result["turn_count"] = len(turns)
    result["question_count"] = len(questions)
    _print(result)


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

    locomo = subparsers.add_parser(
        "locomo",
        help="run one LoCoMo conversation with an OpenAI-compatible Qwen reader",
    )
    locomo.add_argument("--dataset", type=Path, required=True)
    locomo.add_argument("--sample", type=int, default=0)
    locomo.add_argument("--db", type=Path, required=True)
    locomo.add_argument("--output", type=Path, required=True)
    locomo.add_argument(
        "--relation-mode",
        choices=[item.value for item in RelationMode],
        default="hybrid",
    )
    locomo.add_argument("--base-url", required=True)
    locomo.add_argument("--model", default="Qwen/Qwen3-8B")
    locomo.add_argument("--api-key-env", default="RELCHAIN_LLM_API_KEY")
    locomo.add_argument("--max-questions", type=int)
    locomo.add_argument("--context-tokens", type=int, default=4096)
    locomo.add_argument("--timeout", type=float, default=180.0)
    locomo.add_argument("--seed", type=int, default=0)
    locomo.add_argument("--prepare-only", action="store_true")
    locomo.set_defaults(handler=command_locomo)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
