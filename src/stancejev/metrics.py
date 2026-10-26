"""Stance metrics: F_AVG (Favor/Against macro-F1) and Q_AVG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .types import Prediction, StanceLabel


@dataclass
class MetricReport:
    f_avg: float
    q_avg: float
    n: int
    favor_f1: float
    against_f1: float
    none_f1: float
    accuracy: float
    schema_valid_rate: float
    f_macro3: float = 0.0  # mean F1 over Favor/Against/None (VAST)

    def to_dict(self) -> dict:
        return {
            "F_AVG": round(self.f_avg, 4),
            "Q_AVG": round(self.q_avg, 4),
            "n": self.n,
            "Favor_F1": round(self.favor_f1, 4),
            "Against_F1": round(self.against_f1, 4),
            "None_F1": round(self.none_f1, 4),
            "accuracy": round(self.accuracy, 4),
            "schema_valid_rate": round(self.schema_valid_rate, 4),
            "F_macro3": round(self.f_macro3, 4),
        }


def _f1(tp: int, fp: int, fn: int) -> float:
    if tp == 0 and fp == 0 and fn == 0:
        return 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def evaluate(
    golds: Iterable[StanceLabel],
    preds: Iterable[Prediction],
) -> MetricReport:
    gold_list = list(golds)
    pred_list = list(preds)
    if len(gold_list) != len(pred_list):
        raise ValueError("golds and preds length mismatch")
    n = len(gold_list)
    if n == 0:
        return MetricReport(0, 0, 0, 0, 0, 0, 0, 1.0, 0.0)

    f1s: dict[StanceLabel, float] = {}
    for lab in (StanceLabel.FAVOR, StanceLabel.AGAINST, StanceLabel.NONE):
        tp = fp = fn = 0
        for g, p in zip(gold_list, pred_list):
            if p.label == lab and g == lab:
                tp += 1
            elif p.label == lab and g != lab:
                fp += 1
            elif p.label != lab and g == lab:
                fn += 1
        f1s[lab] = _f1(tp, fp, fn)

    f_avg = 0.5 * (f1s[StanceLabel.FAVOR] + f1s[StanceLabel.AGAINST])
    q_avg = sum(p.queries for p in pred_list) / n
    acc = sum(1 for g, p in zip(gold_list, pred_list) if g == p.label) / n
    sv = sum(1 for p in pred_list if p.schema_valid) / n
    f_macro3 = (
        f1s[StanceLabel.FAVOR] + f1s[StanceLabel.AGAINST] + f1s[StanceLabel.NONE]
    ) / 3.0
    return MetricReport(
        f_avg=f_avg,
        q_avg=q_avg,
        n=n,
        favor_f1=f1s[StanceLabel.FAVOR],
        against_f1=f1s[StanceLabel.AGAINST],
        none_f1=f1s[StanceLabel.NONE],
        accuracy=acc,
        schema_valid_rate=sv,
        f_macro3=f_macro3,
    )
