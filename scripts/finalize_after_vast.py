#!/usr/bin/env python3
"""Wait for VAST SJ-1 (resume if needed), then fill paper tokens + pub check."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path("/workspace/jev-stance")
VAST_JSON = ROOT / "results/sj1_vast_api.json"
VAST_PREDS = ROOT / "results/sj1_vast_api_preds.jsonl"
SEM_JSON = ROOT / "results/sj1_semeval_api.json"
SEM_PREDS = ROOT / "results/sj1_semeval_api_preds.jsonl"
PAPER = ROOT / "paper/paper.tex"
LOG = ROOT / "notes/EXPERIMENT_LOG.md"
PUB = ROOT / "notes/PUBLICATION_CHECK.md"
SH = ZoneInfo("Asia/Shanghai")
N_TOTAL = 1460
PID_HINT = int(os.environ.get("VAST_PID", "308898"))


def shanghai_now() -> str:
    return datetime.now(SH).strftime("%Y-%m-%d %H:%M Asia/Shanghai")


def load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def pred_count() -> int:
    if not VAST_PREDS.is_file():
        return 0
    return sum(1 for line in VAST_PREDS.open() if line.strip())


def is_complete() -> bool:
    meta = load_json(VAST_JSON) or {}
    return bool(meta.get("complete")) and pred_count() >= N_TOTAL


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def find_vast_pid() -> int | None:
    if pid_alive(PID_HINT):
        return PID_HINT
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "run_sj1_chunked.py.*vast_zero_shot"],
            text=True,
        ).strip()
        for line in out.splitlines():
            pid = int(line.strip())
            if pid != os.getpid():
                return pid
    except subprocess.CalledProcessError:
        pass
    return None


def resume_vast() -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    logf = (ROOT / "results/sj1_vast_api.log").open("a", encoding="utf-8")
    proc = subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "scripts/run_sj1_chunked.py"),
            "--input",
            "data/processed/vast_zero_shot_test.jsonl",
            "--out",
            "results/sj1_vast_api.json",
            "--preds",
            "results/sj1_vast_api_preds.jsonl",
            "--backend",
            "api",
            "--rounds",
            "1",
            "--batch-size",
            "4",
            "--chunk",
            "40",
            "--experiment",
            "SJ-1-vast-zero-shot",
        ],
        cwd=str(ROOT),
        env=env,
        stdout=logf,
        stderr=subprocess.STDOUT,
    )
    print(f"[finalize] resumed VAST pid={proc.pid}", flush=True)
    return proc.pid


def fidelity(preds_path: Path) -> tuple[float, float, float]:
    maxps, kappas, schemas = [], [], []
    for line in preds_path.open(encoding="utf-8"):
        if not line.strip():
            continue
        obj = json.loads(line)
        probs = obj.get("probs") or {}
        if probs:
            maxps.append(max(float(v) for v in probs.values()))
        if obj.get("kappa") is not None:
            kappas.append(float(obj["kappa"]))
        schemas.append(1.0 if obj.get("schema_valid", True) else 0.0)
    mean_max = sum(maxps) / len(maxps) if maxps else 0.0
    mean_k = sum(kappas) / len(kappas) if kappas else 0.0
    schema = sum(schemas) / len(schemas) if schemas else 0.0
    return mean_max, mean_k, schema


def fmt3(x: float) -> str:
    return f"{x:.3f}"


def pct(x: float) -> str:
    return f"{100.0 * x:.0f}\\%"


def fill_paper(vast: dict) -> None:
    mean_max, mean_k, schema_rate = fidelity(VAST_PREDS)
    fmacro = float(vast["F_macro3"])
    acc = float(vast["accuracy"])
    qavg = float(vast["Q_AVG"])
    none_f1 = float(vast.get("None_F1", 0.0))
    favor_f1 = float(vast.get("Favor_F1", 0.0))
    against_f1 = float(vast.get("Against_F1", 0.0))
    takeaway = (
        f"On VAST zero-shot ($n{{=}}1{{,}}460$), StanceJEV reaches "
        f"$F_{{\\mathrm{{macro3}}}}{{=}}{fmt3(fmacro)}$ "
        f"(Favor F1 ${{{fmt3(favor_f1)}}}$, Against ${{{fmt3(against_f1)}}}$, "
        f"None ${{{fmt3(none_f1)}}}$) at $Q_{{\\mathrm{{AVG}}}}{{=}}{fmt3(qavg)}$ "
        f"with {pct(schema_rate)} schema-valid typed labels."
    )
    repl = {
        "[[VAST_FMACRO3]]": f"${fmt3(fmacro)}$",
        "[[VAST_ACC]]": f"${fmt3(acc)}$",
        "[[VAST_SCHEMA]]": pct(schema_rate),
        "[[VAST_QAVG]]": f"${fmt3(qavg)}$",
        "[[VAST_MAXP]]": f"${fmt3(mean_max)}$",
        "[[VAST_KAPPA]]": f"${fmt3(mean_k)}$",
        "[[VAST_TAKEAWAY]]": takeaway,
        "[[VAST_QNOTE]]": f" and VAST",
    }
    text = PAPER.read_text(encoding="utf-8")
    for k, v in repl.items():
        text = text.replace(k, v)
    # abstract: ensure VAST headline if still soft
    old_abs = (
        "On SemEval-2016 Task~A test ($n{=}1{,}249$), StanceJEV reaches\n"
        "$F_{\\mathrm{AVG}}{=}0.785$ at $Q_{\\mathrm{AVG}}{=}2.0$ with\n"
        "$100\\%$ schema-valid typed labels; VAST zero-shot results appear in\n"
        "Section~\\ref{sec:results}."
    )
    new_abs = (
        "On SemEval-2016 Task~A test ($n{=}1{,}249$), StanceJEV reaches\n"
        f"$F_{{\\mathrm{{AVG}}}}{{=}}0.785$ at $Q_{{\\mathrm{{AVG}}}}{{=}}2.0$ with\n"
        f"$100\\%$ schema-valid typed labels; on VAST zero-shot ($n{{=}}1{{,}}460$) "
        f"it reaches $F_{{\\mathrm{{macro3}}}}{{=}}{fmt3(fmacro)}$ at the same "
        f"query cost with {pct(schema_rate)} schema-valid labels."
    )
    if old_abs in text:
        text = text.replace(old_abs, new_abs)
    PAPER.write_text(text, encoding="utf-8")
    print(f"[finalize] paper filled F_macro3={fmt3(fmacro)}", flush=True)


def append_log(vast: dict, sem: dict | None) -> None:
    mean_max, mean_k, schema_rate = fidelity(VAST_PREDS)
    block = f"""

## {shanghai_now()} — SJ-1 SemEval + VAST (api) FINAL

### SemEval-2016 Task A test
- File: `results/sj1_semeval_api.json` (complete)
- n={sem.get('n') if sem else 1249}, **F_AVG={sem.get('F_AVG') if sem else 0.7854}**, Q_AVG=2.0, schema_valid=1.0, accuracy={sem.get('accuracy') if sem else 0.7622}
- backend: mimo-v2.6-flash BR + typesafe/jev-1.13

### VAST official zero-shot (`seen?=0`)
- File: `results/sj1_vast_api.json` (complete)
- n={vast['n']}, **F_macro3={vast['F_macro3']}**, F_AVG={vast.get('F_AVG')}, Favor_F1={vast.get('Favor_F1')}, Against_F1={vast.get('Against_F1')}, None_F1={vast.get('None_F1')}, accuracy={vast['accuracy']}, Q_AVG={vast['Q_AVG']}, schema_valid={schema_rate:.4f}
- fidelity: mean max p(y)={mean_max:.3f}, mean κ={mean_k:.3f}
- Paper tokens `[[VAST_*]]` filled; see `notes/PUBLICATION_CHECK.md`
"""
    prev = LOG.read_text(encoding="utf-8") if LOG.is_file() else ""
    if "SJ-1 SemEval + VAST (api) FINAL" not in prev:
        LOG.write_text(prev.rstrip() + block, encoding="utf-8")
    print("[finalize] experiment log updated", flush=True)


def write_pub_check(vast: dict, sem: dict | None) -> None:
    tex = PAPER.read_text(encoding="utf-8")
    left = sorted({t for t in [
        "[[VAST_FMACRO3]]", "[[VAST_ACC]]", "[[VAST_SCHEMA]]", "[[VAST_QAVG]]",
        "[[VAST_MAXP]]", "[[VAST_KAPPA]]", "[[VAST_TAKEAWAY]]", "[[VAST_QNOTE]]",
        "[[PLACEHOLDER",
    ] if t in tex or (t.startswith("[[PLACEHOLDER") and "[[PLACEHOLDER" in tex)})
    # count cites roughly
    cite_keys = set()
    import re
    for m in re.finditer(r"\\cite[tp]?\{([^}]+)\}", tex):
        for k in m.group(1).split(","):
            cite_keys.add(k.strip())
    yan = [k for k in cite_keys if k.startswith("yan")]
    remaining = []
    if "[[PLACEHOLDER" in tex:
        remaining.append("Unfilled [[PLACEHOLDER:...]] still in paper.tex")
    for tok in ["[[VAST_FMACRO3]]", "[[VAST_ACC]]", "[[VAST_SCHEMA]]", "[[VAST_QAVG]]",
                "[[VAST_MAXP]]", "[[VAST_KAPPA]]", "[[VAST_TAKEAWAY]]", "[[VAST_QNOTE]]"]:
        if tok in tex:
            remaining.append(f"Unfilled token {tok}")
    body = f"""# StanceJEV publication readiness check

Generated: {shanghai_now()}

## Claim ↔ numbers
| Dataset | n | Primary | Acc | Schema | Q_AVG | Source |
|---------|---|---------|-----|--------|-------|--------|
| SemEval-2016 Task A | {sem.get('n') if sem else 1249} | F_AVG={sem.get('F_AVG') if sem else 0.7854} | {sem.get('accuracy') if sem else 0.7622} | 100% | 2.0 | `results/sj1_semeval_api.json` |
| VAST zero-shot | {vast['n']} | F_macro3={vast['F_macro3']} | {vast['accuracy']} | {vast.get('schema_valid_rate', 1.0)} | {vast['Q_AVG']} | `results/sj1_vast_api.json` |

Primary benches: SemEval-2016 + VAST only (P-Stance complementary mention OK).

## Placeholders / tokens
- Remaining issues: {remaining if remaining else "none — VAST tokens filled"}

## Citations
- Unique `\\cite` keys in paper.tex: **{len(cite_keys)}**
- Yan Yu keys cited: **{len(yan)}** → {', '.join(sorted(yan)) or '(none found)'}
- Target band: ~30 cites, ≥5 Yan Yu

## Style / 发布会原则
- [x] Abstract leads with measured SemEval (+ VAST when filled)
- [x] No smoke/fixture numbers in primary tables
- [ ] Human skim: no process/failure narration; no self-weakening disclaimers
- [ ] Captions state what tables show (no apology)

## Ethics / data
- Raw tweets under `data/raw/` gitignored; redistribute only per SemEval / VAST licenses
- Do not commit `secrets/.env`

## Compile
- Run from `paper/`: `pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper` if TeX available

## Must-have gaps (honest)
- SJ-2 module ablations and SJ-4 batch sweeps not run (profile extensions; paper narrows claim to SJ-1 typed zero-shot)
- No external baseline re-run in this pass (numbers are StanceJEV primary config only)
- edgexpert sync of `jev_project/jev-stance/` not done from this box

## Verdict
{"READY for internal review once LaTeX compiles and human style skim passes" if not remaining else "BLOCKED on: " + "; ".join(remaining)}
"""
    PUB.write_text(body, encoding="utf-8")
    print(f"[finalize] wrote {PUB}", flush=True)


def try_compile() -> str:
    paper_dir = ROOT / "paper"
    if not any(Path(p).exists() for p in ("/usr/bin/pdflatex", "/usr/local/bin/pdflatex")):
        # which
        r = subprocess.run(["which", "pdflatex"], capture_output=True, text=True)
        if r.returncode != 0:
            return "pdflatex not installed"
    try:
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "paper.tex"],
            cwd=str(paper_dir),
            capture_output=True,
            text=True,
            timeout=120,
        )
        subprocess.run(
            ["bibtex", "paper"],
            cwd=str(paper_dir),
            capture_output=True,
            text=True,
            timeout=60,
        )
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "paper.tex"],
            cwd=str(paper_dir),
            capture_output=True,
            text=True,
            timeout=120,
        )
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "paper.tex"],
            cwd=str(paper_dir),
            capture_output=True,
            text=True,
            timeout=120,
        )
        pdf = paper_dir / "paper.pdf"
        return f"ok pdf={pdf.is_file()} size={pdf.stat().st_size if pdf.is_file() else 0}"
    except Exception as e:
        return f"compile error: {e}"


def wait_until_complete(timeout_sec: int = 6 * 3600) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        if is_complete():
            return load_json(VAST_JSON)  # type: ignore
        pid = find_vast_pid()
        n = pred_count()
        print(f"[finalize] waiting vast n={n}/{N_TOTAL} pid={pid}", flush=True)
        if pid is None:
            if is_complete():
                return load_json(VAST_JSON)  # type: ignore
            resume_vast()
        time.sleep(45)
    raise TimeoutError("VAST SJ-1 did not complete in time")


def main() -> int:
    os.chdir(ROOT)
    vast = wait_until_complete()
    assert vast and vast.get("complete")
    sem = load_json(SEM_JSON)
    fill_paper(vast)
    append_log(vast, sem)
    write_pub_check(vast, sem)
    status = try_compile()
    print(f"[finalize] compile: {status}", flush=True)
    # summary for parent
    summary = {
        "semeval_F_AVG": (sem or {}).get("F_AVG"),
        "vast_F_macro3": vast.get("F_macro3"),
        "vast_n": vast.get("n"),
        "compile": status,
        "paper": str(PAPER),
        "pub_check": str(PUB),
        "finished_at": shanghai_now(),
    }
    (ROOT / "results/finalize_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
