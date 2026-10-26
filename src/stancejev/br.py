"""Batch Reasoning (CoVer BR).

OpenAI-compatible chat returning JSON {favor,against,none probs + explanation}.
Includes a deterministic mock backend for offline / smoke runs.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Literal

from .types import BatchItem, LABELS, StanceLabel

BackendName = Literal["mock", "api", "jev"]


FAVOR_CUES = (
    "support", "supports", "favor", "favour", "yes to", "must", "proud",
    "save lives", "triumph", "qualified", "vote", "protect", "expand",
    "non-negotiable", "glad", "works", "improved", "beat", "pass them",
    "healthcare", "autonomy", "ready", "america first",
)
AGAINST_CUES = (
    "against", "oppose", "refuse", "ban", "never", "hysteria", "corrupt",
    "jail", "danger", "murder", "nihilism", "man-hating", "ruined",
    "disarm", "bankrupt", "hate", "experiment", "not negotiable",
    "empty", "chaos", "ignores", "tax grab",
)
NONE_CUES = (
    "coffee", "traffic", "recipe", "phone", "rainy", "dog", "stock",
    "trip", "library", "soccer", "rust", "weekend", "pasta", "hike",
)


def _normalize_probs(raw: dict[str, float]) -> dict[str, float]:
    keys = [l.value for l in LABELS]
    vals = []
    for k in keys:
        v = float(raw.get(k, raw.get(k.lower(), 0.0)))
        vals.append(max(0.0, v))
    s = sum(vals)
    if s <= 0:
        return {k: 1.0 / 3.0 for k in keys}
    return {k: v / s for k, v in zip(keys, vals)}


@dataclass
class BRResult:
    probs: dict[str, float]
    explanation: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchReasoner:
    """LLM (or mock) batch stance reasoner."""

    backend: BackendName = "mock"
    base_url: str = ""
    model: str = ""
    api_key: str = ""
    temperature: float = 0.0
    max_tokens: int = 1200
    timeout: float = 60.0
    batch_size: int = 8

    def __post_init__(self) -> None:
        if self.backend == "api":
            self._load_env_if_needed()

    def _load_env_if_needed(self) -> None:
        from .envutil import env, load_env

        load_env()
        if not self.base_url:
            self.base_url = env("JEVTREE_LLM_BASE_URL") or env("META_JEV_LLM_BASE_URL")
        if not self.model:
            self.model = env("JEVTREE_LLM_MODEL") or env("META_JEV_LLM_MODEL")
        if not self.api_key:
            self.api_key = (
                env("JEVTREE_LLM_API_KEY")
                or env("META_JEV_LLM_API_KEY")
                or env("OPENAI_API_KEY")
            )
        cr = chr(13)
        self.base_url = (self.base_url or "").replace(cr, "").strip().rstrip("/")
        if self.base_url.endswith("/chat/completions"):
            self.base_url = self.base_url[: -len("/chat/completions")]
        self.model = (self.model or "").replace(cr, "").strip()
        self.api_key = (self.api_key or "").replace(cr, "").strip()


    def reason_one(self, item: BatchItem, reconstructed: str) -> BRResult:
        if self.backend in ("mock", "jev"):
            return self._mock_one(item, reconstructed)
        return self._api_one(item, reconstructed)

    def reason_batch(
        self, items: list[BatchItem], reconstructed: list[str]
    ) -> list[BRResult]:
        assert len(items) == len(reconstructed)
        if self.backend in ("mock", "jev"):
            return [
                self._mock_one(it, rec) for it, rec in zip(items, reconstructed)
            ]
        out: list[BRResult] = []
        for i in range(0, len(items), self.batch_size):
            chunk_items = items[i : i + self.batch_size]
            chunk_rec = reconstructed[i : i + self.batch_size]
            out.extend(self._api_batch(chunk_items, chunk_rec))
        return out

    # ── mock ──────────────────────────────────────────────────────────
    def _mock_one(self, item: BatchItem, reconstructed: str) -> BRResult:
        text = f"{reconstructed} {item.target}".lower()
        # Target-aware: if text does not mention target tokens and looks generic → None
        target_toks = [t for t in re.split(r"\W+", item.target.lower()) if len(t) > 2]
        mentions_target = any(t in text for t in target_toks) if target_toks else True

        f_hits = sum(1 for c in FAVOR_CUES if c in text)
        a_hits = sum(1 for c in AGAINST_CUES if c in text)
        n_hits = sum(1 for c in NONE_CUES if c in text)

        # Strong none: lifestyle / unrelated cues and weak stance cues
        if n_hits >= 1 and f_hits + a_hits == 0:
            probs = {"Favor": 0.08, "Against": 0.08, "None": 0.84}
            expl = "Text appears unrelated to the target (lifestyle/neutral cues)."
        elif not mentions_target and f_hits + a_hits <= 1 and n_hits >= 1:
            probs = {"Favor": 0.12, "Against": 0.12, "None": 0.76}
            expl = "Little target overlap; leaning None."
        elif a_hits > f_hits:
            conf = min(0.55 + 0.1 * (a_hits - f_hits), 0.92)
            rest = (1 - conf) / 2
            probs = {"Favor": rest, "Against": conf, "None": rest}
            expl = f"Against cues ({a_hits}) outweigh favor ({f_hits})."
        elif f_hits > a_hits:
            conf = min(0.55 + 0.1 * (f_hits - a_hits), 0.92)
            rest = (1 - conf) / 2
            probs = {"Favor": conf, "Against": rest, "None": rest}
            expl = f"Favor cues ({f_hits}) outweigh against ({a_hits})."
        elif f_hits == a_hits and f_hits > 0:
            probs = {"Favor": 0.40, "Against": 0.40, "None": 0.20}
            expl = "Mixed cues; near-tie between Favor and Against."
        else:
            probs = {"Favor": 0.22, "Against": 0.22, "None": 0.56}
            expl = "No strong stance cues; defaulting toward None."

        probs = _normalize_probs(probs)
        return BRResult(probs=probs, explanation=expl, raw={"backend": "mock"})

    # ── api ───────────────────────────────────────────────────────────
    def _system_prompt(self) -> str:
        return (
            "You are a stance detection assistant. "
            "For each item, judge Favor / Against / None toward the given target. "
            "Return ONLY valid JSON."
        )

    def _user_prompt_batch(
        self, items: list[BatchItem], reconstructed: list[str]
    ) -> str:
        payload = []
        for it, rec in zip(items, reconstructed):
            payload.append(
                {"id": it.id, "target": it.target, "text": rec}
            )
        return (
            "For each object, output an object with keys: "
            "id, favor, against, none, explanation. "
            "favor/against/none are probabilities that sum to 1.\n"
            f"Items:\n{json.dumps(payload, ensure_ascii=False)}\n"
            'Respond as: {"results":[...]}'
        )

    def _api_one(self, item: BatchItem, reconstructed: str) -> BRResult:
        return self._api_batch([item], [reconstructed])[0]

    def _api_batch(
        self, items: list[BatchItem], reconstructed: list[str]
    ) -> list[BRResult]:
        if not self.base_url or not self.model:
            raise RuntimeError(
                "API backend requires JEVTREE_LLM_BASE_URL and JEVTREE_LLM_MODEL"
            )
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {
                "role": "user",
                "content": self._user_prompt_batch(items, reconstructed),
            },
        ]
        resp = self._chat(messages)
        content = ""
        try:
            content = resp["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Unexpected LLM response shape: {resp!r}") from e
        try:
            parsed = _extract_json(content)
        except Exception:
            parsed = {"results": [], "_parse_error": True}
        by_id: dict[str, dict] = {}
        results = parsed.get("results") if isinstance(parsed, dict) else None
        if isinstance(results, list):
            for r in results:
                if isinstance(r, dict) and "id" in r:
                    by_id[str(r["id"])] = r
        # If batch parse largely missed and batch>1, fall back to per-item calls
        miss = [it for it in items if it.id not in by_id]
        if miss and len(items) > 1 and len(by_id) < len(items) // 2:
            out: list[BRResult] = []
            for it, rec in zip(items, reconstructed):
                out.append(self._api_one(it, rec))
            return out
        out = []
        for it in items:
            r = by_id.get(it.id)
            if not r:
                if len(items) == 1 and isinstance(parsed, dict) and "favor" in parsed:
                    r = parsed
                else:
                    # last resort: mock heuristic so pipeline continues
                    m = self._mock_one(it, next(rec for i2, rec in zip(items, reconstructed) if i2.id == it.id))
                    out.append(BRResult(probs=m.probs, explanation=m.explanation + " [br_parse_fallback]", raw={"llm": None, "fallback": "mock"}))
                    continue
            probs = _normalize_probs(
                {
                    "Favor": float(r.get("favor", 0)),
                    "Against": float(r.get("against", 0)),
                    "None": float(r.get("none", 0)),
                }
            )
            expl = str(r.get("explanation", ""))
            out.append(BRResult(probs=probs, explanation=expl, raw={"llm": r}))
        return out

    def _chat(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        data = json.dumps(body).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP {e.code}: {err_body[:500]}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"LLM network error: {e}") from e
        return json.loads(raw)



def _extract_json(text: str) -> Any:
    """Best-effort JSON object extraction from LLM text."""
    text = (text or "").strip()
    if not text:
        return {}
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # direct
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # largest {...} span
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        chunk = m.group(0)
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            # common repairs: trailing commas, single quotes
            repaired = re.sub(r",\s*([}\]])", r"\1", chunk)
            repaired = repaired.replace("'", '"')
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass
    # try to salvage a results array of objects with id/favor/against/none
    objs = re.findall(
        r"\{[^{}]*\"id\"\s*:\s*\"[^\"]+\"[^{}]*\}",
        text,
        re.DOTALL,
    )
    salvaged = []
    for o in objs:
        try:
            salvaged.append(json.loads(re.sub(r",\s*([}\]])", r"\1", o)))
        except json.JSONDecodeError:
            continue
    if salvaged:
        return {"results": salvaged}
    return {"results": [], "_parse_error": True, "_raw": text[:500]}
