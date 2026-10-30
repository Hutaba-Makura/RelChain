from __future__ import annotations

import heapq
import math
from collections import Counter
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
from .tokens import ApproximateTokenCounter, TokenCounter

_QUERY_EXPANSIONS = {
    "educaton": ("education", "edu", "career", "study"),
    "education": ("edu", "career", "study"),
    "fields": ("field", "career", "jobs", "work"),
    "field": ("career", "job", "work"),
    "pursue": ("continue", "career", "keen"),
}


def _stem(token: str) -> str:
    """Small English normalizer for the dependency-light lexical retriever."""
    irregular = {"went": "go", "gone": "go", "did": "do", "was": "be"}
    if token in irregular:
        return irregular[token]
    for suffix in ("ing", "ied", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            base = token[: -len(suffix)]
            if suffix == "ied":
                return base + "y"
            if suffix == "ing" and len(base) >= 2 and base[-1] == base[-2]:
                base = base[:-1]
            return base
    return token


def _search_tokens(text: str, *, expand: bool = False) -> tuple[str, ...]:
    values: list[str] = []
    for token in tokenize(text):
        values.append(_stem(token))
        if expand:
            values.extend(_stem(item) for item in _QUERY_EXPANSIONS.get(token, ()))
    return tuple(values)


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
        self, store: SQLiteMemoryStore, embedder: Embedder, config: MemoryConfig,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.config = config
        self.analyzer = QueryAnalyzer()
        self.token_counter = token_counter or ApproximateTokenCounter()

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
        for episode_id in sorted(state_episode_ids):
            item = self.store.get_episode(episode_id)
            if item is not None:
                seed_map[item.episode_id] = item
        ranked = self._rank_episodes(question, list(seed_map.values()))
        selected, traversed, inspected = self._traverse(
            question, intent, ranked[: self.config.max_episodes], self.store.list_relations()
        )
        evidence = self._backtrack(selected)
        context, context_ids = self._context(question, states, selected, evidence, traversed)
        trace = RetrievalTrace(
            question=question,
            intent=intent,
            seed_episode_ids=tuple(item.episode_id for item in seeds),
            state_ids=tuple(item.state_id for item in states),
            traversed_edge_ids=tuple(item.edge_id for item in traversed),
            retrieved_episode_ids=tuple(item.episode_id for item in selected),
            evidence_event_ids=tuple(dict.fromkeys(eid for item in evidence for eid in item.event_ids)),
            evidence_turn_ids=tuple(dict.fromkeys(item.turn_id for item in evidence)),
            retrieved_turn_refs=tuple(f"{item.conversation_id}:{item.turn_id}" for item in evidence),
            relation_descriptions=tuple(
                self._relation_label(item) for item in traversed
            ),
            inspected_edge_ids=tuple(inspected),
            **context_ids,
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
        query_tokens = _search_tokens(question, expand=True)
        documents: list[tuple[str, ...]] = []
        for episode in episodes:
            speakers: list[str] = []
            for event_id in episode.event_ids:
                event = self.store.get_event(event_id)
                if event is not None:
                    speakers.append(event.speaker_id)
            documents.append(
                _search_tokens(
                    " ".join((episode.title, episode.summary, episode.topic, *speakers))
                )
            )
        document_frequency = Counter(
            token for document in documents for token in set(document)
        )
        document_count = max(len(documents), 1)
        average_length = max(
            sum(len(document) for document in documents) / document_count, 1.0
        )

        def bm25(document: tuple[str, ...]) -> float:
            frequencies = Counter(document)
            score = 0.0
            for token in set(query_tokens):
                frequency = frequencies[token]
                if not frequency:
                    continue
                frequency_in_documents = document_frequency[token]
                inverse_frequency = math.log(
                    1.0
                    + (document_count - frequency_in_documents + 0.5)
                    / (frequency_in_documents + 0.5)
                )
                denominator = frequency + 1.2 * (
                    0.25 + 0.75 * len(document) / average_length
                )
                score += inverse_frequency * frequency * 2.2 / denominator
            return score

        scores = {
            episode.episode_id: (
                bm25(document)
                + 0.1 * self.embedder.similarity(query_vector, episode.embedding)
                + jaccard(query_tokens, document)
            )
            for episode, document in zip(episodes, documents)
        }
        return sorted(
            episodes,
            key=lambda item: (
                scores[item.episode_id],
                item.observed_end,
                item.episode_id,
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
    ) -> tuple[list[Episode], list[Relation], list[str]]:
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
        inspected: list[str] = []
        visited_edges: set[str] = set()
        frontier: list[tuple[float, int, str]] = []
        for seed in seeds:
            heapq.heappush(frontier, (-1.0, 0, seed.episode_id))
        while frontier and len(selected) < self.config.max_episodes:
            _, depth, node_id = heapq.heappop(frontier)
            if depth >= self.config.max_hops:
                continue
            if node_id not in graph:
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
                inspected.append(relation.edge_id)
                if relation.edge_id in visited_edges:
                    continue
                visited_edges.add(relation.edge_id)
                score = self._edge_score(question, intent, relation)
                if score <= 0:
                    continue
                episode = self.store.get_episode(neighbor_id)
                if episode is None:
                    continue
                if neighbor_id not in selected:
                    if len(selected) >= self.config.max_episodes:
                        break
                    selected[neighbor_id] = episode
                    traversed.append(relation)
                    heapq.heappush(frontier, (-score, depth + 1, neighbor_id))
        return list(selected.values()), traversed, inspected

    def _edge_score(
        self, question: str, intent: QueryIntent, relation: Relation
    ) -> float:
        score = relation.confidence
        if relation.category in intent.preferred_categories:
            score += self.config.category_boost
        semantic = jaccard(
            tokenize(question),
            tokenize(relation.relation_text + " " + relation.description),
        )
        return score + semantic

    def _backtrack(self, episodes: Sequence[Episode]) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []
        turn_items: dict[tuple[str, str], EvidenceItem] = {}
        for episode in episodes:
            for event_id in episode.event_ids:
                event = self.store.get_event(event_id)
                if event is None:
                    continue
                turn = self.store.get_turn(event.conversation_id, event.turn_id)
                if turn is None:
                    continue
                key = (event.conversation_id, event.turn_id)
                if key in turn_items:
                    prior = turn_items[key]
                    turn_items[key] = prior.model_copy(update={
                        "event_ids": tuple(dict.fromkeys((*prior.event_ids, event.event_id)))
                    })
                    continue
                turn_items[key] = (
                    EvidenceItem(
                        conversation_id=turn.conversation_id,
                        turn_id=turn.turn_id,
                        event_id=event.event_id,
                        event_ids=(event.event_id,),
                        episode_id=episode.episode_id,
                        speaker_id=turn.speaker_id,
                        observed_at=turn.observed_at,
                        raw_text=turn.text,
                        source_start=event.source_span.start,
                        source_end=event.source_span.end,
                    )
                )
        return sorted(turn_items.values(), key=lambda item: (item.observed_at, item.conversation_id, item.turn_id))

    @staticmethod
    def _relation_label(relation: Relation) -> str:
        prefix = f"{relation.category.value}/" if relation.category else ""
        return prefix + relation.relation_text

    def _context(
        self,
        question: str,
        states: Sequence[UserState],
        episodes: Sequence[Episode],
        evidence: Sequence[EvidenceItem],
        relations: Sequence[Relation],
    ) -> tuple[str, dict[str, object]]:
        # Select whole records by a common deterministic ranking, then display raw
        # turns chronologically. A large record cannot block smaller later records.
        rank = {episode.episode_id: index for index, episode in enumerate(episodes)}
        candidates: list[tuple[str, str, str, object]] = []
        for item in sorted(evidence, key=lambda x: (rank.get(x.episode_id, 999999), x.turn_id)):
            ids = ",".join(item.event_ids)
            line = f"[TURN {item.turn_id} EVENTS {ids} {item.observed_at.isoformat()}] {item.raw_text.replace(chr(10), r'\n')}"
            candidates.append(("turn", item.turn_id, line, item))
        for state in states:
            candidates.append(("state", state.state_id,
                f"[STATE {state.state_id} r{state.revision} {state.status.value}] {state.subject_id}.{state.attribute}={state.value} scope={state.scope}", state))
        for episode in episodes:
            candidates.append(("episode", episode.episode_id,
                f"[EPISODE {episode.episode_id}] {episode.title}: {episode.summary}", episode))
        if self.config.include_relation_context:
            for relation in relations:
                candidates.append(("relation", relation.edge_id,
                    f"[RELATION {relation.edge_id}] {relation.source_episode_id} --{self._relation_label(relation)}--> {relation.target_episode_id}", relation))
        candidates = [
            (kind, identifier, line.replace("\r", "\\r").replace("\n", "\\n"), item)
            for kind, identifier, line, item in candidates
        ]
        chosen: list[tuple[str, str, str, object]] = []
        excluded: dict[str, list[str]] = {"turn": [], "state": [], "episode": [], "relation": []}
        for candidate in candidates:
            proposed = "\n".join([item[2] for item in chosen] + [candidate[2]])
            if self.token_counter.count(proposed) <= self.config.context_token_budget:
                chosen.append(candidate)
            elif candidate[0] in excluded:
                excluded[candidate[0]].append(candidate[1])
        raw = sorted((item for item in chosen if item[0] == "turn"), key=lambda x: (x[3].observed_at, x[3].conversation_id, x[1]))
        displayed = raw + [item for item in chosen if item[0] != "turn"]
        context = "\n".join(item[2] for item in displayed)
        while displayed and self.token_counter.count(context) > self.config.context_token_budget:
            removed = displayed.pop()
            if removed[0] in excluded:
                excluded[removed[0]].append(removed[1])
            context = "\n".join(item[2] for item in displayed)
        chosen = displayed
        raw = [item for item in displayed if item[0] == "turn"]
        turns = [item[3] for item in raw]
        return context, {
            "context_turn_ids": tuple(item.turn_id for item in turns),
            "context_turn_refs": tuple(f"{item.conversation_id}:{item.turn_id}" for item in turns),
            "context_event_ids": tuple(eid for item in turns for eid in item.event_ids),
            "context_state_ids": tuple(item[1] for item in chosen if item[0] == "state"),
            "context_episode_ids": tuple(item[1] for item in chosen if item[0] == "episode"),
            "context_relation_ids": tuple(item[1] for item in chosen if item[0] == "relation"),
            "excluded_state_ids": tuple(excluded["state"]),
            "excluded_turn_ids": tuple(excluded["turn"]),
            "excluded_episode_ids": tuple(excluded["episode"]),
            "excluded_relation_ids": tuple(excluded["relation"]),
            "memory_token_count": self.token_counter.count(context),
            "token_count_method": self.token_counter.name,
        }
