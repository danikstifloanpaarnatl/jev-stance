"""Context Reconstruction (CoVer CR, lightweight).

Optional knowledge concat + sentence keep-all / simple length filter.
Embedder interface is pluggable; default is a no-op keep-all path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from .types import BatchItem


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


@dataclass
class ContextReconstruction:
    """Reconstruct input text for stance reasoning.

    Parameters
    ----------
    min_sent_chars:
        Drop ultra-short fragments below this length (noise filter).
    attach_knowledge:
        If True and item.knowledge is set, prepend it.
    max_chars:
        Soft cap on reconstructed text length.
    embedder:
        Optional embedder for future BGE-style SE filtering (unused in keep-all).
    use_embed_filter:
        If True and embedder is set, keep sentences with cosine to target above
        `relatedness_threshold` (simple stand-in for CoVer SE filter).
    relatedness_threshold:
        Cosine threshold when embed filtering is on.
    """

    min_sent_chars: int = 8
    attach_knowledge: bool = True
    max_chars: int = 1200
    embedder: Embedder | None = None
    use_embed_filter: bool = False
    relatedness_threshold: float = 0.15

    def reconstruct(self, item: BatchItem) -> str:
        text = (item.text or "").strip()
        sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
        if not sents:
            sents = [text] if text else [""]

        kept: list[str] = []
        for s in sents:
            if len(s) < self.min_sent_chars and len(sents) > 1:
                continue
            kept.append(s)
        if not kept:
            kept = sents

        if self.use_embed_filter and self.embedder is not None and item.target:
            kept = self._filter_by_relatedness(kept, item.target)

        body = " ".join(kept).strip()
        if self.attach_knowledge and item.knowledge:
            body = f"[Knowledge] {item.knowledge.strip()}\n[Text] {body}"

        if len(body) > self.max_chars:
            body = body[: self.max_chars].rsplit(" ", 1)[0] + "…"
        return body

    def _filter_by_relatedness(self, sents: list[str], target: str) -> list[str]:
        assert self.embedder is not None
        vecs = self.embedder.embed(sents + [target])
        tvec = vecs[-1]
        scored: list[tuple[float, str]] = []
        for s, v in zip(sents, vecs[:-1]):
            scored.append((_cosine(v, tvec), s))
        filtered = [s for c, s in scored if c >= self.relatedness_threshold]
        return filtered or sents


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class HashingEmbedder:
    """Tiny bag-of-char-ngram embedder for offline relatedness (no deps)."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            v = [0.0] * self.dim
            s = (t or "").lower()
            for i in range(len(s) - 1):
                h = hash(s[i : i + 2]) % self.dim
                v[h] += 1.0
            n = sum(x * x for x in v) ** 0.5
            if n > 0:
                v = [x / n for x in v]
            out.append(v)
        return out
