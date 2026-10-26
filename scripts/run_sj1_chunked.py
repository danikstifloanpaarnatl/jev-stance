#!/usr/bin/env python3
"""Chunked SJ-1 eval with resume — writes metrics + preds incrementally."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stancejev.data import load_jsonl  # noqa: E402
from stancejev.envutil import load_env  # noqa: E402
from stancejev.metrics import evaluate  # noqa: E402
from stancejev.cli import _pipeline  # noqa: E402

SHANGHAI = ZoneInfo("Asia/Shanghai")


def main() -> int:
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--backend", default="api", choices=("mock", "api", "jev"))
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--experiment", default="SJ-1")
    args = ap.parse_args()

    items = load_jsonl(args.input)
    if args.limit:
        items = items[: args.limit]
    missing = [it.id for it in items if it.gold is None]
    if missing:
        print(f"error: {len(missing)} unlabeled", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    pred_path = Path(args.preds)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = out_path.with_suffix(".ckpt.json")

    done_ids: set[str] = set()
    preds_by_id: dict = {}
    if pred_path.is_file():
        for line in pred_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            preds_by_id[obj["id"]] = obj
            done_ids.add(obj["id"])
        print(f"resume: {len(done_ids)} preds already on disk", file=sys.stderr)

    pipe = _pipeline(args.backend, max_rounds=args.rounds, batch_size=args.batch_size)
    pending = [it for it in items if it.id not in done_ids]
    print(f"total={len(items)} pending={len(pending)} backend={args.backend}", file=sys.stderr)

    for i in range(0, len(pending), args.chunk):
        chunk = pending[i : i + args.chunk]
        print(
            f"chunk {i // args.chunk + 1}: {len(chunk)} items "
            f"({i + len(done_ids)}/{len(items)} done-ish)",
            file=sys.stderr,
            flush=True,
        )
        try:
            chunk_preds = pipe.predict(chunk)
        except Exception as e:
            print(f"chunk failed: {type(e).__name__}: {e}", file=sys.stderr)
            # flush what we have
            _flush(pred_path, items, preds_by_id)
            _write_partial(out_path, ckpt, items, preds_by_id, args, error=str(e))
            return 1
        for p in chunk_preds:
            preds_by_id[p.id] = p.to_dict()
        _flush(pred_path, items, preds_by_id)
        _write_partial(out_path, ckpt, items, preds_by_id, args)
        print(f"checkpoint n={len(preds_by_id)}", file=sys.stderr, flush=True)

    # final metrics on full ordered list
    from stancejev.types import Prediction, StanceLabel

    ordered = []
    golds = []
    for it in items:
        d = preds_by_id[it.id]
        ordered.append(
            Prediction(
                id=d["id"],
                label=StanceLabel.from_str(d["label"]),
                probs=d["probs"],
                explanation=d.get("explanation", ""),
                kappa=float(d.get("kappa", 0)),
                rounds=int(d.get("rounds", 1)),
                queries=int(d.get("queries", 1)),
                schema_valid=bool(d.get("schema_valid", True)),
                backend=d.get("backend", args.backend),
            )
        )
        golds.append(it.gold)
    report = evaluate(golds, ordered)
    primary = "F_macro3" if "vast" in str(args.input).lower() else "F_AVG"
    out = report.to_dict()
    out.update(
        {
            "experiment": args.experiment,
            "backend": args.backend,
            "input": str(args.input),
            "rounds": args.rounds,
            "batch_size": args.batch_size,
            "primary_metric": primary,
            "primary_score": out.get(primary),
            "complete": True,
            "timestamp": datetime.now(SHANGHAI).isoformat(timespec="seconds"),
            "timezone": "Asia/Shanghai",
        }
    )
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    if ckpt.is_file():
        ckpt.unlink()
    print(json.dumps(out, indent=2))
    return 0


def _flush(pred_path: Path, items, preds_by_id: dict) -> None:
    with pred_path.open("w", encoding="utf-8") as f:
        for it in items:
            if it.id in preds_by_id:
                f.write(json.dumps(preds_by_id[it.id], ensure_ascii=False) + "\n")


def _write_partial(out_path, ckpt, items, preds_by_id, args, error=None) -> None:
    from stancejev.types import Prediction, StanceLabel
    from stancejev.metrics import evaluate

    subset_items = [it for it in items if it.id in preds_by_id]
    if not subset_items:
        return
    ordered = []
    golds = []
    for it in subset_items:
        d = preds_by_id[it.id]
        ordered.append(
            Prediction(
                id=d["id"],
                label=StanceLabel.from_str(d["label"]),
                probs=d["probs"],
                queries=int(d.get("queries", 1)),
                schema_valid=bool(d.get("schema_valid", True)),
            )
        )
        golds.append(it.gold)
    report = evaluate(golds, ordered)
    primary = "F_macro3" if "vast" in str(args.input).lower() else "F_AVG"
    out = report.to_dict()
    out.update(
        {
            "experiment": args.experiment,
            "backend": args.backend,
            "input": str(args.input),
            "complete": False,
            "n_done": len(subset_items),
            "n_total": len(items),
            "primary_metric": primary,
            "primary_score": out.get(primary),
            "error": error,
            "timestamp": datetime.now(SHANGHAI).isoformat(timespec="seconds"),
            "timezone": "Asia/Shanghai",
        }
    )
    ckpt.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    # also mirror to out so readers see partial
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
