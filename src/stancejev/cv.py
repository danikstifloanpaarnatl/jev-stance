"""Consistency Verification (CoVer CV, heuristic SLM stand-in).

Computes κ between LLM explanation+probs and a lightweight verifier.
Gate: if κ < τ, caller may re-query; else accept.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .br import AGAINST_CUES, FAVOR_CUES, NONE_CUES, _normalize_probs
from .types import LABELS, StanceLabel


def _token_set(text: str) -> set[str]:
    return {t for t in re.split(r"\W+", (text or "").lower()) if len(t) > 2}


def _lexicon_probs(text: str, target: str) -> dict[str, float]:
    blob = f"{text} {target}".lower()
    f = sum(1 for c in FAVOR_CUES if c in blob)
    a = sum(1 for c in AGAINST_CUES if c in blob)
    n = sum(1 for c in NONE_CUES if c in blob)
    # soft counts
    raw = {
        "Favor": 0.15 + f,
        "Against": 0.15 + a,
        "None": 0.15 + n + (0.4 if f + a == 0 else 0.0),
    }
    return _normalize_probs(raw)


def _js_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """Jensen–Shannon divergence on stance simplex (base-2 bits, clipped)."""
    keys = [l.value for l in LABELS]
    eps = 1e-9

    def _h(dist: list[float]) -> float:
        return -sum(x * __import__("math").log(x + eps, 2) for x in dist)

    pv = [max(eps, float(p.get(k, 0.0))) for k in keys]
    qv = [max(eps, float(q.get(k, 0.0))) for k in keys]
    sp, sq = sum(pv), sum(qv)
    pv = [x / sp for x in pv]
    qv = [x / sq for x in qv]
    m = [(a + b) / 2 for a, b in zip(pv, qv)]
    js = _h(m) - 0.5 * _h(pv) - 0.5 * _h(qv)
    return max(0.0, min(1.0, js))


@dataclass
class ConsistencyVerifier:
    """Heuristic consistency scorer (no transformers required)."""

    threshold: float = 0.55
    # weight on explanation–target lexical overlap vs distributional agreement
    overlap_weight: float = 0.35

    def slm_probs(self, explanation: str, text: str, target: str) -> dict[str, float]:
        # Prefer explanation; fall back to text
        return _lexicon_probs(explanation or text, target)

    def kappa(
        self,
        llm_probs: dict[str, float],
        explanation: str,
        text: str,
        target: str,
    ) -> tuple[float, dict[str, float]]:
        slm = self.slm_probs(explanation, text, target)
        js = _js_divergence(llm_probs, slm)
        # agreement in [0,1]
        dist_agree = 1.0 - js
        # explanation should mention target-ish tokens or stance words
        expl_toks = _token_set(explanation)
        tgt_toks = _token_set(target)
        overlap = 0.0
        if tgt_toks:
            overlap = len(expl_toks & tgt_toks) / max(1, len(tgt_toks))
        # also reward stance lexicon presence in explanation
        stance_hit = any(
            c in (explanation or "").lower()
            for c in ("favor", "against", "support", "oppose", "neutral", "unrelated", "cue")
        )
        if stance_hit:
            overlap = min(1.0, overlap + 0.4)
        kappa = (1.0 - self.overlap_weight) * dist_agree + self.overlap_weight * min(
            1.0, overlap
        )
        return max(0.0, min(1.0, kappa)), slm

    def accept(self, kappa: float) -> bool:
        return kappa >= self.threshold

    def weighted_aggregate(
        self, rounds: list[tuple[dict[str, float], float]]
    ) -> dict[str, float]:
        """Consistency-weighted average of probability vectors.

        rounds: list of (probs, kappa)
        """
        if not rounds:
            return {l.value: 1.0 / 3.0 for l in LABELS}
        keys = [l.value for l in LABELS]
        acc = {k: 0.0 for k in keys}
        wsum = 0.0
        for probs, k in rounds:
            w = max(1e-3, float(k))
            wsum += w
            for key in keys:
                acc[key] += w * float(probs.get(key, 0.0))
        return _normalize_probs({k: acc[k] / wsum for k in keys})
