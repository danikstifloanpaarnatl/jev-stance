"""Typed JEV choice decisions — local emit + real OpenRouter Decisions API."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .envutil import env, load_env
from .types import LABELS, Prediction, StanceLabel

STANCE_CRITERIA = {
    "favor": "The text supports or favors the target.",
    "against": "The text opposes or is against the target.",
    "none": "The text is neutral, unrelated, or expresses no clear stance.",
}

_LABEL_FROM_CHOICE = {
    "favor": StanceLabel.FAVOR,
    "against": StanceLabel.AGAINST,
    "none": StanceLabel.NONE,
}


def stance_question(target: str) -> dict[str, Any]:
    return {
        "type": "choice",
        "instructions": (
            f"Select the stance of the text toward the target '{target}'. "
            "Report probabilities over favor, against, and none."
        ),
        "criteria": dict(STANCE_CRITERIA),
    }


def argmax_label(probs: dict[str, float]) -> StanceLabel:
    return max(LABELS, key=lambda l: float(probs.get(l.value, 0.0)))


def _normalize_label_probs(raw: dict[str, float]) -> dict[str, float]:
    keys = [l.value for l in LABELS]
    # accept favor/Favor etc.
    vals = []
    for k in keys:
        v = float(raw.get(k, raw.get(k.lower(), 0.0)))
        vals.append(max(0.0, v))
    s = sum(vals) or 1.0
    return {k: v / s for k, v in zip(keys, vals)}


def emit_typed_decision(
    *,
    item_id: str,
    target: str,
    probs: dict[str, float],
    explanation: str,
    kappa: float,
    rounds: int,
    queries: int,
    reconstructed_text: str,
    backend: str,
    meta_extra: dict[str, Any] | None = None,
) -> Prediction:
    """Always schema-valid: label ∈ {Favor,Against,None}, probs sum ≈ 1."""
    norm = _normalize_label_probs(probs)
    label = argmax_label(norm)
    meta: dict[str, Any] = {"target": target, "question": stance_question(target)}
    if meta_extra:
        meta.update(meta_extra)
    return Prediction(
        id=item_id,
        label=label,
        probs=norm,
        explanation=explanation,
        kappa=kappa,
        rounds=rounds,
        queries=queries,
        reconstructed_text=reconstructed_text,
        backend=backend,
        schema_valid=True,
        meta=meta,
    )


def to_jev_choice_payload(pred: Prediction) -> dict[str, Any]:
    q = pred.meta.get("question") or stance_question(pred.meta.get("target", ""))
    return {
        "type": "choice",
        "choice": pred.label.value.lower()
        if pred.label != StanceLabel.NONE
        else "none",
        "probs": {
            "favor": pred.probs.get("Favor", 0.0),
            "against": pred.probs.get("Against", 0.0),
            "none": pred.probs.get("None", 0.0),
        },
        "instructions": q.get("instructions"),
        "criteria": q.get("criteria"),
        "kappa": pred.kappa,
        "explanation": pred.explanation,
        "schema_valid": True,
    }


def build_stance_sop_dict(name: str = "stancejev-zssd") -> dict[str, Any]:
    return {
        "version": 2,
        "name": name,
        "description": "Typed probabilistic stance toward a target (StanceJEV)",
        "jev_model": env("JEV_MODEL", "typesafe/jev-1.13") or "typesafe/jev-1.13",
        "nodes": [
            {
                "id": "reconstructed",
                "kind": "expression",
                "expression": "obs['reconstructed_text']",
            },
            {
                "id": "stance",
                "kind": "jev",
                "state": (
                    "{'text': nodes['reconstructed'], 'target': obs['target']}"
                ),
                "questions": {
                    "label": {
                        "type": "choice",
                        "instructions": "Stance of text toward target.",
                        "criteria": dict(STANCE_CRITERIA),
                    }
                },
            },
        ],
        "output": "nodes['stance']['label']['choice']",
    }


# ── Real Jev Decisions API (OpenRouter) ─────────────────────────────


def jev_configured() -> bool:
    load_env()
    return bool(env("OPENROUTER_API_KEY") and env("JEV_DECISIONS_URL"))


def call_jev_stance(
    *,
    text: str,
    target: str,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """POST typed stance choice to OpenRouter Decisions API.

    Returns dict with keys: probs (Favor/Against/None), choice, confidence,
    raw_answer, usage, model.
    Never logs API keys.
    """
    load_env()
    api_key = env("OPENROUTER_API_KEY")
    url = env("JEV_DECISIONS_URL", "https://openrouter.ai/api/alpha/decisions")
    model = env("JEV_MODEL", "typesafe/jev-1.13") or "typesafe/jev-1.13"
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set (secrets/.env)")

    payload = {
        "model": model,
        "state": {
            "text": text,
            "target": target,
        },
        "questions": {
            "stance": {
                "type": "choice",
                "instructions": (
                    f"What stance does the text express toward the target '{target}'?"
                ),
                "criteria": dict(STANCE_CRITERIA),
            }
        },
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Jev Decisions HTTP {e.code}: {err[:500]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Jev Decisions network error: {e}") from e

    answers = body.get("answers") or {}
    ans = answers.get("stance") or {}
    probs_raw = ans.get("probabilities") or {}
    probs = _normalize_label_probs(
        {
            "Favor": float(probs_raw.get("favor", 0.0)),
            "Against": float(probs_raw.get("against", 0.0)),
            "None": float(probs_raw.get("none", 0.0)),
        }
    )
    choice = str(ans.get("choice") or argmax_label(probs).value.lower())
    return {
        "probs": probs,
        "choice": choice,
        "confidence": float(ans.get("confidence") or max(probs.values())),
        "raw_answer": ans,
        "usage": body.get("usage"),
        "model": body.get("model") or model,
        "id": body.get("id"),
    }
