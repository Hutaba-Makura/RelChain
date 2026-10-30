from __future__ import annotations

import math
import hashlib
import json
from typing import Protocol


class TokenCounter(Protocol):
    name: str
    exact: bool

    def count(self, text: str) -> int: ...


class ApproximateTokenCounter:
    name = "approximate-characters-divided-by-four"
    exact = False

    def count(self, text: str) -> int:
        return math.ceil(len(text) / 4)


class TransformersTokenCounter:
    """Explicit local tokenizer; never silently falls back to an approximation."""

    exact = True

    def __init__(self, tokenizer_path: str) -> None:
        from transformers import AutoTokenizer  # optional experiment dependency

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=True, trust_remote_code=False
        )
        fingerprint = hashlib.sha256(json.dumps({
            "vocab": self.tokenizer.get_vocab(),
            "chat_template": self.tokenizer.chat_template,
            "special_tokens": self.tokenizer.special_tokens_map,
        }, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()[:16]
        self.name = f"transformers:{tokenizer_path}:{fingerprint}"

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def count_chat(self, system: str, user: str) -> int:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return len(self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        ))
