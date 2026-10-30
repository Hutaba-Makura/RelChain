from __future__ import annotations

import hashlib
import math
from time import perf_counter
from collections.abc import Sequence

from .text import tokenize
from .ports import Embedder


class MeasuredEmbedder:
    def __init__(self, delegate: Embedder) -> None:
        self.delegate = delegate
        self.name = delegate.name
        self.embed_seconds = 0.0
        self.embed_calls = 0
        self.similarity_seconds = 0.0
        self.similarity_calls = 0

    def embed(self, text: str) -> tuple[float, ...]:
        start = perf_counter()
        try:
            return self.delegate.embed(text)
        finally:
            self.embed_seconds += perf_counter() - start
            self.embed_calls += 1

    def similarity(self, left: Sequence[float], right: Sequence[float]) -> float:
        start = perf_counter()
        try:
            return self.delegate.similarity(left, right)
        finally:
            self.similarity_seconds += perf_counter() - start
            self.similarity_calls += 1


class HashingEmbedder:
    """Deterministic, dependency-light baseline suitable for tests and ablations."""

    def __init__(self, dimensions: int = 384, name: str = "hashing-384-v1") -> None:
        self.dimensions = dimensions
        self.name = name

    def embed(self, text: str) -> tuple[float, ...]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "big") % self.dimensions
            sign = 1.0 if digest[8] & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return tuple(vector)
        return tuple(value / norm for value in vector)

    def similarity(self, left: Sequence[float], right: Sequence[float]) -> float:
        if len(left) != len(right) or not left:
            return 0.0
        return sum(a * b for a, b in zip(left, right))
