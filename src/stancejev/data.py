"""Dataset loaders: SemEval-2016-style JSONL + bundled synthetic fixture."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .types import BatchItem, StanceLabel

FIXTURE_PATH = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "synthetic_stance.jsonl"


def load_jsonl(path: str | Path) -> list[BatchItem]:
    """Load stance JSONL.

    Accepted fields (case-insensitive aliases):
      id, text|tweet|sentence, target|Topic, label|stance|Stance
      optional knowledge
    """
    path = Path(path)
    items: list[BatchItem] = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append(_row_to_item(obj, fallback_id=f"row_{i}"))
    return items


def iter_jsonl(path: str | Path) -> Iterator[BatchItem]:
    for it in load_jsonl(path):
        yield it


def load_synthetic_fixture() -> list[BatchItem]:
    if not FIXTURE_PATH.is_file():
        raise FileNotFoundError(f"Missing fixture: {FIXTURE_PATH}")
    return load_jsonl(FIXTURE_PATH)


def _row_to_item(obj: dict, fallback_id: str) -> BatchItem:
    def pick(*keys: str, default=None):
        for k in keys:
            if k in obj and obj[k] is not None:
                return obj[k]
            # case-insensitive
            for ok, ov in obj.items():
                if ok.lower() == k.lower() and ov is not None:
                    return ov
        return default

    text = pick("text", "tweet", "sentence", "Tweet")
    target = pick("target", "Topic", "topic")
    label_raw = pick("label", "stance", "Stance", "Opinion Towards")
    kid = str(pick("id", "ID", default=fallback_id))
    knowledge = pick("knowledge", default=None)
    if text is None or target is None:
        raise ValueError(f"Row missing text/target: {obj!r}")
    gold = None
    if label_raw is not None and str(label_raw).strip() != "":
        gold = StanceLabel.from_str(str(label_raw))
    return BatchItem(
        id=kid,
        text=str(text),
        target=str(target),
        gold=gold,
        knowledge=str(knowledge) if knowledge else None,
    )
