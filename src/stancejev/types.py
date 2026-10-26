"""Core types for StanceJEV."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StanceLabel(str, Enum):
    FAVOR = "Favor"
    AGAINST = "Against"
    NONE = "None"

    @classmethod
    def from_str(cls, s: str) -> "StanceLabel":
        key = (s or "").strip().lower()
        mapping = {
            "favor": cls.FAVOR,
            "favour": cls.FAVOR,
            "support": cls.FAVOR,
            "pro": cls.FAVOR,
            "against": cls.AGAINST,
            "oppose": cls.AGAINST,
            "con": cls.AGAINST,
            "none": cls.NONE,
            "neutral": cls.NONE,
            "unrelated": cls.NONE,
        }
        if key not in mapping:
            raise ValueError(f"Unknown stance label: {s!r}")
        return mapping[key]


LABELS: tuple[StanceLabel, ...] = (
    StanceLabel.FAVOR,
    StanceLabel.AGAINST,
    StanceLabel.NONE,
)


@dataclass
class BatchItem:
    """One stance instance (text + target [+ gold])."""

    id: str
    text: str
    target: str
    gold: StanceLabel | None = None
    knowledge: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Prediction:
    """Typed stance prediction with probabilities and audit fields."""

    id: str
    label: StanceLabel
    probs: dict[str, float]  # keys: Favor / Against / None
    explanation: str = ""
    kappa: float = 1.0
    rounds: int = 1
    queries: int = 1
    reconstructed_text: str = ""
    backend: str = "mock"
    schema_valid: bool = True
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label.value,
            "probs": dict(self.probs),
            "explanation": self.explanation,
            "kappa": self.kappa,
            "rounds": self.rounds,
            "queries": self.queries,
            "reconstructed_text": self.reconstructed_text,
            "backend": self.backend,
            "schema_valid": self.schema_valid,
            "meta": self.meta,
        }
