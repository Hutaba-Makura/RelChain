from __future__ import annotations

import heapq
from collections.abc import Sequence

import networkx as nx  # type: ignore[import-untyped]

from .config import MemoryConfig
from .models import (
    Episode,
    EvidenceItem,
    QueryIntent,
    Relation,
    RelationCategory,
    RetrievalResult,
    RetrievalTrace,
    StateStatus,
    UserState,
)
from .ports import Embedder
from .storage import SQLiteMemoryStore
from .text import jaccard, tokenize


class QueryAnalyzer:
    def analyze(self, question: str) -> QueryIntent:
        lowered = question.lower()
        reference_time = "current"
        if any(cue in question for cue in ("以前", "昔", "当時", "過去")) or any(
            cue in lowered for cue in ("previously", "before", "in the past")
        ):
            reference_time = "past"
        categories: list[RelationCategory] = []
        if any(cue in question for cue in ("いつ", "前", "後", "時系列")) or any(
            cue in lowered for cue in ("when", "before", "after")
        ):
            categories.append(RelationCategory.TEMPORAL)
        if any(cue in question for cue in ("なぜ", "理由", "原因")) or any(
            cue in lowered for cue in ("why", "cause", "reason")
        ):
            categories.append(RelationCategory.CAUSAL)
        if not categories:
            categories.append(RelationCategory.SEMANTIC)
        attribute = None
        if any(cue in question for cue in ("好き", "好み", "重視")) or any(
            cue in lowered for cue in ("like", "prefer", "preference")
        ):
            attribute = "preference"
        elif any(cue in question for cue in ("目標", "したい")) or "goal" in lowered:
            attribute = "goal"
        elif (
            "ホテル" in question and "立地" in question
        ) or "hotel location" in lowered:
            attribute = "hotel_location_preference"
        return QueryIntent(
            attribute=attribute,
            reference_time=reference_time,
            preferred_categories=tuple(categories),
            tokens=tokenize(question),
        )


class MemoryRetriever:
    def __init__(
        self, store: SQLiteMemoryStore, embedder: Embedder, config: MemoryConfig
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.config = config
        self.analyzer = QueryAnalyzer()

    def retrieve(self, question: str) -> RetrievalResult:
        intent = self.analyzer.analyze(question)
        episodes = self.store.list_episodes()
        states = self._rank_states(question, intent)
        seeds = self._rank_episodes(question, episodes)[
            : self.config.episode_candidates
        ]
        state_episode_ids = {
            episode_id for state in states for episode_id in state.evidence_episode_ids
        }
        seed_map = {item.episode_id: item for item in seeds}
        for episode_id in state_episode_ids:
            item = self.store.get_episode(episode_id)
            if item is not None:
                seed_map[item.episode_id] = item
        selected, traversed = self._traverse(
            question, intent, list(seed_map.values()), self.store.list_relations()
        )
        selected = selected[: self.config.max_episodes]
        evidence = self._backtrack(selected)
        context = self._context(states, selected, evidence, traversed)
        trace = RetrievalTrace(
            question=question,
            intent=intent,
            seed_episode_ids=tuple(item.episode_id for item in seeds),
            state_ids=tuple(item.state_id for item in states),
            traversed_edge_ids=tuple(item.edge_id for item in traversed),
            retrieved_episode_ids=tuple(item.episode_id for item in selected),
            evidence_event_ids=tuple(item.event_id for item in evidence),
            evidence_turn_ids=tuple(dict.fromkeys(item.turn_id for item in evidence)),
            relation_descriptions=tuple(
                self._relation_label(item) for item in traversed
            ),
        )
        self.store.log_retrieval(trace)
        return RetrievalResult(
            question=question,
            states=tuple(states),
            episodes=tuple(selected),
            evidence=tuple(evidence),
            trace=trace,
            context=context,
        )

    def _rank_episodes(
        self, question: str, episodes: Sequence[Episode]
    ) -> list[Episode]:
        query_vector = self.embedder.embed(question)
        query_tokens = tokenize(question)
        return sorted(
            episodes,
            key=lambda item: (
                self.embedder.similarity(query_vector, item.embedding)
                + jaccard(query_tokens, tokenize(item.title + " " + item.summary)),
                item.observed_end,
            ),
            reverse=True,
        )

    def _rank_states(self, question: str, intent: QueryIntent) -> list[UserState]:
        statuses = None
        if intent.reference_time == "current":
            statuses = (StateStatus.ACTIVE, StateStatus.UNCERTAIN)
        states = self.store.list_states(
            subject_id=intent.subject_id,
            attribute=intent.attribute,
            statuses=statuses,
        )
        query_tokens = tokenize(question)
        states.sort(
            key=lambda item: (
                item.status == StateStatus.ACTIVE,
                jaccard(
                    query_tokens,
                    tokenize(f"{item.attribute} {item.value} {item.scope}"),
                ),
                item.recorded_at,
            ),
            reverse=True,
        )
        return states[: self.config.state_candidates]

    def _traverse(
        self,
        question: str,
        intent: QueryIntent,
        seeds: Sequence[Episode],
        relations: Sequence[Relation],
    ) -> tuple[list[Episode], list[Relation]]:
        graph = nx.MultiDiGraph()
        for relation in relations:
            graph.add_edge(
                relation.source_episode_id,
                relation.target_episode_id,
                key=relation.edge_id,
                relation=relation,
            )
            # Traversal may backtrack evidence, while the stored relation direction is preserved.
            graph.add_edge(
                relation.target_episode_id,
                relation.source_episode_id,
                key=f"{relation.edge_id}:reverse",
                relation=relation,
            )
        selected: dict[str, Episode] = {item.episode_id: item for item in seeds}
        traversed: list[Relation] = []
        visited_edges: set[str] = set()
        frontier: list[tuple[float, int, str]] = []
        for seed in seeds:
            heapq.heappush(frontier, (-1.0, 0, seed.episode_id))
        while frontier and len(selected) < self.config.max_episodes:
            _, depth, node_id = heapq.heappop(frontier)
            if depth >= self.config.max_hops:
                continue
            neighbors = sorted(
                (
                    (target_id, data["relation"])
                    for _, target_id, _, data in graph.out_edges(
                        node_id, keys=True, data=True
                    )
                ),
                key=lambda item: self._edge_score(question, intent, item[1]),
                reverse=True,
            )[: self.config.neighbors_per_node]
            for neighbor_id, relation in neighbors:
                if relation.edge_id in visited_edges:
                    continue
                visited_edges.add(relation.edge_id)
                score = self._edge_score(question, intent, relation)
                if score <= 0:
                    continue
                episode = self.store.get_episode(neighbor_id)
                if episode is None:
                    continue
                traversed.append(relation)
                if neighbor_id not in selected:
                    selected[neighbor_id] = episode
                    heapq.heappush(frontier, (-score, depth + 1, neighbor_id))
        return list(selected.values()), traversed

    def _edge_score(
        self, question: str, intent: QueryIntent, relation: Relation
    ) -> float:
        score = relation.confidence
        if relation.category in intent.preferred_categories:
            score += 1.0
        semantic = jaccard(
            tokenize(question),
            tokenize(relation.relation_text + " " + relation.description),
        )
        return score + semantic

    def _backtrack(self, episodes: Sequence[Episode]) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []
        seen_turns: set[tuple[str, str]] = set()
        for episode in episodes:
            for event_id in episode.event_ids:
                event = self.store.get_event(event_id)
                if event is None:
                    continue
                turn = self.store.get_turn(event.conversation_id, event.turn_id)
                if turn is None:
                    continue
                key = (event.conversation_id, event.turn_id)
                if key in seen_turns:
                    continue
                seen_turns.add(key)
                evidence.append(
                    EvidenceItem(
                        turn_id=turn.turn_id,
                        event_id=event.event_id,
                        episode_id=episode.episode_id,
                        speaker_id=turn.speaker_id,
                        observed_at=turn.observed_at,
                        raw_text=turn.text,
                        source_start=event.source_span.start,
                        source_end=event.source_span.end,
                    )
                )
        return sorted(evidence, key=lambda item: (item.observed_at, item.turn_id))

    @staticmethod
    def _relation_label(relation: Relation) -> str:
        prefix = f"{relation.category.value}/" if relation.category else ""
        return prefix + relation.relation_text

    def _context(
        self,
        states: Sequence[UserState],
        episodes: Sequence[Episode],
        evidence: Sequence[EvidenceItem],
        relations: Sequence[Relation],
    ) -> str:
        lines: list[str] = []
        for state in states:
            lines.append(
                f"[STATE {state.state_id} r{state.revision} {state.status.value}] "
                f"{state.subject_id}.{state.attribute}={state.value} scope={state.scope}"
            )
        for episode in episodes:
            lines.append(
                f"[EPISODE {episode.episode_id}] {episode.title}: {episode.summary}"
            )
        for relation in relations:
            lines.append(
                f"[RELATION {relation.edge_id}] {relation.source_episode_id} --"
                f"{self._relation_label(relation)}--> {relation.target_episode_id}"
            )
        for item in evidence:
            lines.append(
                f"[TURN {item.turn_id} EVENT {item.event_id} {item.observed_at.isoformat()}] "
                f"{item.raw_text}"
            )
        # A conservative approximation that keeps all providers under the same budget.
        char_budget = self.config.context_token_budget * 4
        output: list[str] = []
        used = 0
        for line in lines:
            if used + len(line) + 1 > char_budget:
                break
            output.append(line)
            used += len(line) + 1
        return "\n".join(output)
