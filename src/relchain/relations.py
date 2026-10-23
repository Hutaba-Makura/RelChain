from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .config import MemoryConfig, RelationMode
from .models import Episode, Event, Relation, RelationCategory
from .ports import Embedder, RelationTextGenerator
from .text import snake_case, stable_id


@dataclass(frozen=True)
class RelationCandidate:
    source: Episode
    target: Episode
    category: RelationCategory
    evidence_event_ids: tuple[str, ...]
    confidence: float


class DeterministicRelationTextGenerator:
    def free_relation(
        self, source: Episode, target: Episode, category_hint: RelationCategory
    ) -> tuple[str, str, float]:
        left = snake_case(source.topic, "source_topic")
        right = snake_case(target.topic, "target_topic")
        verb = {
            RelationCategory.TEMPORAL: "preceded",
            RelationCategory.CAUSAL: "contributed_to",
            RelationCategory.SEMANTIC: "shares_context_with",
        }[category_hint]
        return f"{left}_{verb}_{right}", f"{source.title} -> {target.title}", 0.75

    def hybrid_relation(
        self, source: Episode, target: Episode, category: RelationCategory
    ) -> tuple[str, str, float]:
        name, description, confidence = self.free_relation(source, target, category)
        return name, description, confidence


class RelationBuilder:
    def __init__(
        self,
        embedder: Embedder,
        config: MemoryConfig,
        text_generator: RelationTextGenerator | None = None,
    ) -> None:
        self.embedder = embedder
        self.config = config
        self.text_generator = text_generator or DeterministicRelationTextGenerator()

    def candidate_pairs(
        self, episodes: Sequence[Episode], events: dict[str, Event]
    ) -> list[RelationCandidate]:
        candidates: dict[tuple[str, str, RelationCategory], RelationCandidate] = {}
        ordered = sorted(
            episodes, key=lambda item: (item.observed_start, item.episode_id)
        )
        # Temporal/causal LLM candidates are limited to chronological neighbors.
        for index, source in enumerate(ordered):
            for target in ordered[
                index + 1 : index + 1 + self.config.neighbors_per_node
            ]:
                evidence = tuple(
                    dict.fromkeys(source.event_ids[-1:] + target.event_ids[:1])
                )
                key = (source.episode_id, target.episode_id, RelationCategory.TEMPORAL)
                candidates[key] = RelationCandidate(
                    source=source,
                    target=target,
                    category=RelationCategory.TEMPORAL,
                    evidence_event_ids=evidence,
                    confidence=1.0,
                )
                target_text = " ".join(
                    events[event_id].event_text for event_id in target.event_ids
                )
                causal_cues = ("ため", "ので", "結果", "because", "due to", "therefore")
                if any(cue in target_text.lower() for cue in causal_cues):
                    causal_key = (
                        source.episode_id,
                        target.episode_id,
                        RelationCategory.CAUSAL,
                    )
                    candidates[causal_key] = RelationCandidate(
                        source=source,
                        target=target,
                        category=RelationCategory.CAUSAL,
                        evidence_event_ids=evidence,
                        confidence=0.7,
                    )

        # Semantic candidates use a common top-N budget per source.
        for source in ordered:
            scored: list[tuple[float, Episode]] = []
            for target in ordered:
                if target.episode_id == source.episode_id:
                    continue
                similarity = self.embedder.similarity(
                    source.embedding, target.embedding
                )
                shared = bool(set(source.entity_ids) & set(target.entity_ids))
                score = max(similarity, 0.6 if shared else 0.0)
                if score >= self.config.semantic_threshold:
                    scored.append((score, target))
            for score, target in sorted(scored, key=lambda item: item[0], reverse=True)[
                : self.config.neighbors_per_node
            ]:
                left, right = (
                    (source, target)
                    if (source.observed_start, source.episode_id)
                    <= (target.observed_start, target.episode_id)
                    else (target, source)
                )
                evidence = tuple(
                    dict.fromkeys(left.event_ids[-1:] + right.event_ids[:1])
                )
                key = (left.episode_id, right.episode_id, RelationCategory.SEMANTIC)
                existing = candidates.get(key)
                if existing is None or score > existing.confidence:
                    candidates[key] = RelationCandidate(
                        source=left,
                        target=right,
                        category=RelationCategory.SEMANTIC,
                        evidence_event_ids=evidence,
                        confidence=score,
                    )
        return list(candidates.values())

    def build(
        self, episodes: Sequence[Episode], events: dict[str, Event]
    ) -> list[Relation]:
        relations: list[Relation] = []
        for candidate in self.candidate_pairs(episodes, events):
            mode = self.config.relation_mode
            if mode == RelationMode.FIXED:
                relation_text = {
                    RelationCategory.TEMPORAL: "precedes",
                    RelationCategory.SEMANTIC: "similar_topic",
                    RelationCategory.CAUSAL: "causes",
                }[candidate.category]
                category: RelationCategory | None = candidate.category
                description = relation_text
                confidence = candidate.confidence
            elif mode == RelationMode.FREE:
                relation_text, description, confidence = (
                    self.text_generator.free_relation(
                        candidate.source, candidate.target, candidate.category
                    )
                )
                category = None
            else:
                relation_text, description, confidence = (
                    self.text_generator.hybrid_relation(
                        candidate.source, candidate.target, candidate.category
                    )
                )
                category = candidate.category
            edge_id = stable_id(
                "relation",
                candidate.source.episode_id,
                candidate.target.episode_id,
                candidate.category.value,
                mode.value,
            )
            relations.append(
                Relation(
                    edge_id=edge_id,
                    source_episode_id=candidate.source.episode_id,
                    target_episode_id=candidate.target.episode_id,
                    mode=mode.value,
                    category=category,
                    relation_text=relation_text,
                    description=description,
                    evidence_event_ids=candidate.evidence_event_ids,
                    confidence=min(candidate.confidence, confidence),
                    symmetric=candidate.category == RelationCategory.SEMANTIC,
                )
            )
        return relations
