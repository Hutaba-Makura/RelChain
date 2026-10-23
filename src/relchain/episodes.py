from __future__ import annotations

from collections import Counter
from datetime import timedelta
from itertools import pairwise
from typing import Sequence

from .config import MemoryConfig
from .models import Episode, Event
from .ports import Embedder, EpisodeAnnotator
from .text import jaccard, stable_id, tokenize


class EpisodeBuilder:
    def __init__(
        self,
        embedder: Embedder,
        config: MemoryConfig,
        annotator: EpisodeAnnotator | None = None,
    ) -> None:
        self.embedder = embedder
        self.config = config
        self.annotator = annotator

    def build(self, events: Sequence[Event]) -> list[Episode]:
        if not events:
            return []
        ordered = sorted(events, key=lambda item: (item.observed_at, item.event_id))
        clusters: list[list[Event]] = [[ordered[0]]]
        reasons: list[str] = ["conversation_start"]
        for previous, current in pairwise(ordered):
            gap = current.observed_at - previous.observed_at
            similarity = jaccard(
                tokenize(previous.event_text), tokenize(current.event_text)
            )
            shared_entities = set(previous.entity_ids) & set(current.entity_ids)
            if gap > timedelta(minutes=self.config.episode_gap_minutes):
                clusters.append([current])
                reasons.append("time_gap")
            elif (
                similarity < self.config.episode_boundary_threshold
                and not shared_entities
            ):
                split, reason = (
                    self.annotator.decide_boundary(previous, current)
                    if self.annotator is not None
                    else (True, "topic_shift")
                )
                if split:
                    clusters.append([current])
                    reasons.append(reason)
                else:
                    clusters[-1].append(current)
            else:
                clusters[-1].append(current)
        return [
            self._make_episode(cluster, reason)
            for cluster, reason in zip(clusters, reasons)
        ]

    def _make_episode(self, events: Sequence[Event], boundary_reason: str) -> Episode:
        first, last = events[0], events[-1]
        token_counts = Counter(
            token
            for event in events
            for token in tokenize(event.event_text)
            if len(token) >= 2
        )
        if self.annotator is not None:
            title, summary, topic = self.annotator.summarize(events)
        else:
            topic = token_counts.most_common(1)[0][0] if token_counts else "general"
            title = " / ".join(
                event.event_text.rstrip("。.!?")[:30] for event in events[:2]
            )
            summary = " ".join(event.event_text for event in events)
        entity_ids = tuple(
            dict.fromkeys(entity for event in events for entity in event.entity_ids)
        )
        event_times = [
            event.event_time_start for event in events if event.event_time_start
        ]
        episode_id = stable_id(
            first.conversation_id + ":episode", first.event_id, last.event_id
        )
        return Episode(
            episode_id=episode_id,
            conversation_id=first.conversation_id,
            title=title or topic,
            summary=summary,
            event_ids=tuple(event.event_id for event in events),
            source_turn_ids=tuple(dict.fromkeys(event.turn_id for event in events)),
            entity_ids=entity_ids,
            topic=topic,
            observed_start=first.observed_at,
            observed_end=last.observed_at,
            event_start=min(event_times) if event_times else None,
            event_end=max(event_times) if event_times else None,
            boundary_reason=boundary_reason,
            embedding=self.embedder.embed(f"{title}\n{summary}"),
        )
