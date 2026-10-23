from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Iterable

_WORD_RE = re.compile(r"[A-Za-z0-9_]+|[\u3040-\u30ff\u3400-\u9fff]")
_SENTENCE_RE = re.compile(r"[^。！？!?\n]+[。！？!?]?", re.MULTILINE)


def stable_id(prefix: str, *parts: object) -> str:
    value = "\x1f".join(str(part) for part in parts)
    digest = uuid.uuid5(uuid.NAMESPACE_URL, value).hex[:20]
    return f"{prefix}_{digest}"


def content_hash(*parts: object) -> str:
    value = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def tokenize(text: str) -> tuple[str, ...]:
    base = [item.lower() for item in _WORD_RE.findall(text)]
    cjk = [item for item in base if len(item) == 1 and ord(item) > 127]
    bigrams = [a + b for a, b in zip(cjk, cjk[1:])]
    latin = [item for item in base if not (len(item) == 1 and ord(item) > 127)]
    return tuple(dict.fromkeys(latin + cjk + bigrams))


def sentence_spans(text: str) -> Iterable[tuple[str, int, int]]:
    for match in _SENTENCE_RE.finditer(text):
        sentence = match.group(0).strip()
        if sentence:
            leading = len(match.group(0)) - len(match.group(0).lstrip())
            start = match.start() + leading
            yield sentence, start, start + len(sentence)


def snake_case(text: str, fallback: str = "specific_relation") -> str:
    latin = re.findall(r"[A-Za-z0-9]+", text.lower())
    if latin:
        return "_".join(latin[:8])
    tokens = [token for token in tokenize(text) if len(token) >= 2]
    if not tokens:
        return fallback
    ascii_digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"topic_{ascii_digest}"


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
