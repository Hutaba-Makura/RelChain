from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

from .answering import EvidenceExtractiveAnswerGenerator
from .config import MemoryConfig
from .embeddings import HashingEmbedder, MeasuredEmbedder
from .episodes import EpisodeBuilder
from .extraction import RuleBasedEventExtractor, RuleBasedStateExtractor
from .models import Answer, ConversationTurn, IngestReport, StateTransition
from .ports import (
    AnswerGenerator,
    Embedder,
    EpisodeAnnotator,
    EventExtractor,
    RelationTextGenerator,
    StateExtractor,
)
from .relations import RelationBuilder
from .retrieval import MemoryRetriever
from .state import StateUpdater
from .storage import SQLiteMemoryStore
from .tokens import TokenCounter


class MemorySystem:
    def __init__(
        self,
        database: str | Path,
        config: MemoryConfig | None = None,
        *,
        embedder: Embedder | None = None,
        event_extractor: EventExtractor | None = None,
        state_extractor: StateExtractor | None = None,
        episode_annotator: EpisodeAnnotator | None = None,
        relation_text_generator: RelationTextGenerator | None = None,
        answer_generator: AnswerGenerator | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.config = config or MemoryConfig()
        self.store = SQLiteMemoryStore(database)
        existing_mode = self.store.get_metadata("relation_mode")
        if (
            existing_mode is not None
            and existing_mode != self.config.relation_mode.value
        ):
            raise ValueError(
                "relation mode mismatch; use a separate database per mode "
                f"(database={existing_mode}, requested={self.config.relation_mode.value})"
            )
        self.store.set_metadata("relation_mode", self.config.relation_mode.value)
        for key, value in (("writer_name", self.config.writer_name),
                           ("reader_name", self.config.reader_name),
                           ("embedding_name", self.config.embedding_name)):
            if self.store.get_metadata(key) is None:
                self.store.set_metadata(key, value)
        self.embedder = MeasuredEmbedder(embedder or HashingEmbedder(name=self.config.embedding_name))
        self.event_extractor = event_extractor or RuleBasedEventExtractor()
        self.state_extractor = state_extractor or RuleBasedStateExtractor()
        self.episode_builder = EpisodeBuilder(
            self.embedder, self.config, annotator=episode_annotator
        )
        self.relation_builder = RelationBuilder(
            self.embedder, self.config, relation_text_generator
        )
        self.state_updater = StateUpdater(self.store, self.config)
        self.retriever = MemoryRetriever(self.store, self.embedder, self.config, token_counter)
        self.answer_generator = answer_generator or EvidenceExtractiveAnswerGenerator()

    def ingest(self, turns: Sequence[ConversationTurn], *, build_relations: bool = True) -> IngestReport:
        started = perf_counter()
        timings = {key: 0.0 for key in (
            "event_extraction", "episode_construction", "state_update",
            "candidate_graph", "relation_generation", "db_write",
        )}
        embed_before = self.embedder.embed_seconds
        new_turn_ids: list[str] = []
        new_event_ids: list[str] = []
        failures: list[str] = []
        affected_conversations: set[str] = set()
        for turn in sorted(turns, key=lambda item: (item.observed_at, item.turn_id)):
            existing = self.store.get_turn(turn.conversation_id, turn.turn_id)
            if existing is not None:
                self.store.save_turn(turn)  # validates append-only equality
                if self.store.has_events_for_turn(turn.conversation_id, turn.turn_id):
                    continue
            try:
                if existing is None:
                    write_start = perf_counter()
                    self.store.save_turn(turn)
                    timings["db_write"] += perf_counter() - write_start
                stage_start = perf_counter()
                events = self.event_extractor.extract(turn)
                timings["event_extraction"] += perf_counter() - stage_start
                if not events:
                    raise ValueError("extractor returned no grounded events")
                write_start = perf_counter()
                self.store.save_events(events)
                timings["db_write"] += perf_counter() - write_start
                if existing is None:
                    new_turn_ids.append(turn.turn_id)
                new_event_ids.extend(item.event_id for item in events)
                affected_conversations.add(turn.conversation_id)
            except Exception as exc:
                message = f"{turn.turn_id}: {exc}"
                failures.append(message)
                self.store.record_failure("event_extraction", turn.turn_id, str(exc))

        affected_episode_ids: set[str] = set()
        for conversation_id in affected_conversations:
            events = self.store.list_events(conversation_id)
            stage_start = perf_counter()
            episodes = self.episode_builder.build(events)
            timings["episode_construction"] += perf_counter() - stage_start
            write_start = perf_counter()
            self.store.save_episodes(episodes)
            timings["db_write"] += perf_counter() - write_start
            for episode in episodes:
                if set(episode.event_ids) & set(new_event_ids):
                    affected_episode_ids.add(episode.episode_id)

        all_events = {item.event_id: item for item in self.store.list_events()}
        all_episodes = self.store.list_episodes()
        if build_relations and (new_event_ids or (all_episodes and not self.store.list_relations())):
            stage_start = perf_counter()
            candidates = self.relation_builder.candidate_pairs(all_episodes, all_events)
            timings["candidate_graph"] += perf_counter() - stage_start
            stage_start = perf_counter()
            relations = self.relation_builder.build(all_episodes, all_events, candidates)
            timings["relation_generation"] += perf_counter() - stage_start
            write_start = perf_counter()
            self.store.replace_relations(self.config.relation_mode.value, relations)
            timings["db_write"] += perf_counter() - write_start
        else:
            relations = self.store.list_relations()

        changed_states: list[str] = []
        transitions: list[StateTransition] = []
        for episode in all_episodes:
            if episode.episode_id not in affected_episode_ids:
                continue
            episode_events = [all_events[event_id] for event_id in episode.event_ids]
            try:
                stage_start = perf_counter()
                candidates = self.state_extractor.extract(episode, episode_events)
                for candidate in candidates:
                    states, state_transitions = self.state_updater.apply(candidate)
                    changed_states.extend(item.state_id for item in states)
                    transitions.extend(state_transitions)
                timings["state_update"] += perf_counter() - stage_start
            except Exception as exc:
                message = f"{episode.episode_id}: {exc}"
                failures.append(message)
                self.store.record_failure("state_update", episode.episode_id, str(exc))

        timings["embedding"] = self.embedder.embed_seconds - embed_before
        timings["total"] = perf_counter() - started
        return IngestReport(
            turn_ids=tuple(new_turn_ids),
            event_ids=tuple(new_event_ids),
            episode_ids=tuple(sorted(affected_episode_ids)),
            relation_ids=tuple(item.edge_id for item in relations),
            state_ids=tuple(dict.fromkeys(changed_states)),
            transitions=tuple(transitions),
            failures=tuple(failures),
            timings_seconds=timings,
            counts={"turns": self.store.counts()["turns"],
                    "events": self.store.counts()["events"],
                    "episodes": self.store.counts()["episodes"],
                    "relations": self.store.counts()["relations"]},
        )

    def query(self, question: str) -> Answer:
        retrieval = self.retriever.retrieve(question)
        return self.answer_generator.generate(retrieval)
