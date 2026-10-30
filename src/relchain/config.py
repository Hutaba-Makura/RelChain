from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class RelationMode(StrEnum):
    FIXED = "fixed"
    FREE = "free"
    HYBRID = "hybrid"


class MemoryConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    relation_mode: RelationMode = RelationMode.HYBRID
    episode_candidates: int = Field(default=10, ge=1)
    state_candidates: int = Field(default=5, ge=1)
    neighbors_per_node: int = Field(default=5, ge=1)
    max_hops: int = Field(default=2, ge=0)
    max_episodes: int = Field(default=30, ge=1)
    context_token_budget: int = Field(default=4096, ge=1)
    full_input_token_budget: int | None = Field(default=None, ge=1)
    include_relation_context: bool = True
    category_boost: float = Field(default=1.0, ge=0.0)
    confidence_policy: str = "common"
    semantic_threshold: float = Field(default=0.18, ge=0.0, le=1.0)
    episode_boundary_threshold: float = Field(default=0.08, ge=0.0, le=1.0)
    episode_gap_minutes: int = Field(default=180, ge=1)
    inferred_state_min_episodes: int = Field(default=2, ge=1)
    writer_name: str = "deterministic-v1"
    reader_name: str = "evidence-extractive-v1"
    embedding_name: str = "hashing-384-v1"
    seed: int = 0
