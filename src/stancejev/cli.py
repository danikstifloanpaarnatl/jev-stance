"""CLI: python -m stancejev predict|eval|smoke"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from .br import BatchReasoner
from .cr import ContextReconstruction
from .cv import ConsistencyVerifier
from .data import load_jsonl, load_synthetic_fixture
from .decision import build_stance_sop_dict, to_jev_choice_payload
from .metrics import evaluate
from .pipeline import StanceJEVPipeline
from .types import StanceLabel

SHANGHAI = ZoneInfo("Asia/Shanghai")


def _pipeline(backend: str, max_rounds: int, batch_size: int) -> StanceJEVPipeline:
    return StanceJEVPipeline(
        cr=ContextReconstruction(),
        br=BatchReasoner(backend=backend, batch_size=batch_size),  # type: ignore[arg-type]
        cv=ConsistencyVerifier(),
        max_rounds=max_rounds,
    )


def cmd_smoke(args: argparse.Namespace) -> int:
    items = load_synthetic_fixture()
    pipe = _pipeline("mock", max_rounds=args.rounds, batch_size=args.batch_size)
    preds = pipe.predict(items)
    golds = [it.gold for it in items]
    assert all(g is not None for g in golds)
    report = evaluate(golds, preds)  # type: ignore[arg-type]
    out = {
        "experiment": "SJ-1-smoke",
        "dataset": "synthetic_stance_fixture",
        "backend": "mock",
        "F_AVG": report.f_avg,
        "Q_AVG": report.q_avg,
        "n": report.n,
        "Favor_F1": report.favor_f1,
        "Against_F1": report.against_f1,
        "None_F1": report.none_f1,
        "accuracy": report.accuracy,
        "schema_valid_rate": report.schema_valid_rate,
        "timestamp": datetime.now(SHANGHAI).isoformat(timespec="seconds"),
        "timezone": "Asia/Shanghai",
        "note": "[[SMOKE-FIXTURE]] offline mock on bundled synthetic data — NOT SemEval/VAST",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    # also dump predictions
    pred_path = out_path.with_name(out_path.stem + "_preds.jsonl")
    with pred_path.open("w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
    # SOP export
    sop_path = out_path.with_name("stancejev_sop.json")
    sop_path.write_text(
        json.dumps(build_stance_sop_dict(), indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: out[k] for k in ("F_AVG", "Q_AVG", "n", "backend", "accuracy")}, indent=2))
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    items = load_jsonl(args.input)
    if args.limit:
        items = items[: args.limit]
    missing = [it.id for it in items if it.gold is None]
    if missing:
        print(f"error: {len(missing)} items lack gold labels", file=sys.stderr)
        return 2
    pipe = _pipeline(args.backend, max_rounds=args.rounds, batch_size=args.batch_size)
    preds = pipe.predict(items)
    report = evaluate([it.gold for it in items], preds)  # type: ignore[arg-type]
    out = report.to_dict()
    # Primary metric: SemEval Task A uses F_AVG (Favor/Against);
    # VAST protocol uses three-label mean F_macro3.
    primary = "F_macro3" if "vast" in str(args.input).lower() else "F_AVG"
    out.update(
        {
            "experiment": args.experiment,
            "backend": args.backend,
            "input": str(args.input),
            "rounds": args.rounds,
            "batch_size": args.batch_size,
            "primary_metric": primary,
            "primary_score": out.get(primary),
            "timestamp": datetime.now(SHANGHAI).isoformat(timespec="seconds"),
            "timezone": "Asia/Shanghai",
        }
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    if args.preds:
        with Path(args.preds).open("w", encoding="utf-8") as f:
            for p in preds:
                f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
    print(json.dumps(out, indent=2))
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    items = load_jsonl(args.input)
    if args.limit:
        items = items[: args.limit]
    if args.target:
        for it in items:
            it.target = args.target
    pipe = _pipeline(args.backend, max_rounds=args.rounds, batch_size=args.batch_size)
    preds = pipe.predict(items)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for p in preds:
            row = p.to_dict()
            row["jev_choice"] = to_jev_choice_payload(p)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(preds)} predictions → {out_path}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="stancejev", description="StanceJEV CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("smoke", help="Offline mock SJ-1 on synthetic fixture")
    s.add_argument("--out", default="/workspace/jev-stance/results/sj1_smoke.json")
    s.add_argument("--rounds", type=int, default=2)
    s.add_argument("--batch-size", type=int, default=8)
    s.set_defaults(func=cmd_smoke)

    e = sub.add_parser("eval", help="Evaluate labeled JSONL")
    e.add_argument("--input", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--preds", default="")
    e.add_argument("--backend", choices=("mock", "api", "jev"), default="mock")
    e.add_argument("--rounds", type=int, default=2)
    e.add_argument("--batch-size", type=int, default=8)
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--experiment", default="SJ-1")
    e.set_defaults(func=cmd_eval)

    pr = sub.add_parser("predict", help="Predict stance for JSONL")
    pr.add_argument("--input", required=True)
    pr.add_argument("--out", required=True)
    pr.add_argument("--target", default="")
    pr.add_argument("--backend", choices=("mock", "api", "jev"), default="mock")
    pr.add_argument("--rounds", type=int, default=2)
    pr.add_argument("--batch-size", type=int, default=8)
    pr.add_argument("--limit", type=int, default=0)
    pr.set_defaults(func=cmd_predict)

    return p


def main(argv: list[str] | None = None) -> int:
    from .envutil import load_env
    load_env()
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
