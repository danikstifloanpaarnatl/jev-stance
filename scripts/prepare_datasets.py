#!/usr/bin/env python3
"""Convert SemEval-2016 Task 6 and VAST into StanceJEV JSONL."""
from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# VAST: 0=con, 1=pro, 2=neutral (Allaway repo / community docs)
VAST_LABEL = {"0": "Against", "1": "Favor", "2": "None"}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    labs = Counter(r["stance"] for r in rows)
    print(f"wrote {path} n={len(rows)} stance={dict(labs)}")


def convert_semeval() -> None:
    mapping = [
        (
            RAW / "semeval2016" / "SemEval2016-Task6-subtaskA-testdata-gold.txt",
            OUT / "semeval2016_test.jsonl",
        ),
        (
            RAW / "semeval2016" / "semeval2016-task6-trainingdata.txt",
            OUT / "semeval2016_train.jsonl",
        ),
    ]
    for src, dst in mapping:
        rows = []
        with src.open(encoding="latin-1") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                stance = (r.get("Stance") or "").strip()
                if stance.upper() == "UNKNOWN" or not stance:
                    continue
                # normalize FAVOR/AGAINST/NONE → Favor/Against/None
                stance_n = stance.strip().capitalize()
                if stance_n == "Favor" or stance.upper() == "FAVOR":
                    stance_n = "Favor"
                elif stance.upper() == "AGAINST":
                    stance_n = "Against"
                elif stance.upper() == "NONE":
                    stance_n = "None"
                tweet = (r.get("Tweet") or "").replace("#SemST", "").strip()
                rows.append(
                    {
                        "id": str(r["ID"]),
                        "text": tweet,
                        "target": r["Target"].strip(),
                        "stance": stance_n,
                        "label": stance_n,
                        "dataset": "semeval2016",
                        "split": "test" if "test" in dst.name else "train",
                    }
                )
        write_jsonl(dst, rows)


def convert_vast() -> None:
    src = RAW / "vast" / "vast_test.csv"
    rows_zs = []
    rows_all = []
    with src.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lab = VAST_LABEL.get(str(r["label"]).strip())
            if lab is None:
                continue
            topic = (r.get("topic_str") or r.get("new_topic") or "").strip()
            text = (r.get("post") or r.get("text_s") or "").strip()
            if not topic or not text:
                continue
            item = {
                "id": str(r.get("new_id") or r.get("ori_id")),
                "text": text,
                "target": topic,
                "stance": lab,
                "label": lab,
                "dataset": "vast",
                "seen": int(r.get("seen?", "0") or 0),
                "type_idx": int(r.get("type_idx") or 0),
            }
            rows_all.append(item)
            if item["seen"] == 0:
                rows_zs.append(item)
    write_jsonl(OUT / "vast_test_all.jsonl", rows_all)
    write_jsonl(OUT / "vast_zero_shot_test.jsonl", rows_zs)
    # native-only (exclude synthetic neutrals type_idx=4)
    native = [r for r in rows_zs if r["type_idx"] != 4]
    write_jsonl(OUT / "vast_zero_shot_test_native.jsonl", native)


def stratified_sample(src: Path, dst: Path, n: int, seed: int = 42) -> None:
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    by = defaultdict(list)
    for r in rows:
        by[r["stance"]].append(r)
    rng = random.Random(seed)
    # proportional allocation
    total = len(rows)
    alloc = {k: max(1, round(n * len(v) / total)) for k, v in by.items()}
    # fix sum
    while sum(alloc.values()) > n:
        k = max(alloc, key=alloc.get)
        if alloc[k] > 1:
            alloc[k] -= 1
        else:
            break
    while sum(alloc.values()) < n:
        k = max(by, key=lambda x: len(by[x]) - alloc.get(x, 0))
        if alloc.get(k, 0) < len(by[k]):
            alloc[k] = alloc.get(k, 0) + 1
        else:
            break
    out = []
    for k, take in alloc.items():
        pool = by[k][:]
        rng.shuffle(pool)
        out.extend(pool[:take])
    rng.shuffle(out)
    write_jsonl(dst, out[:n])


def main() -> None:
    convert_semeval()
    convert_vast()
    # stratified eval subsets for cost control / ablations
    stratified_sample(
        OUT / "semeval2016_test.jsonl",
        OUT / "semeval2016_test_strat400.jsonl",
        400,
    )
    stratified_sample(
        OUT / "vast_zero_shot_test.jsonl",
        OUT / "vast_zero_shot_test_strat400.jsonl",
        400,
    )
    print("done")


if __name__ == "__main__":
    main()
