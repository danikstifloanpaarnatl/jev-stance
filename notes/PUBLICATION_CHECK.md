# StanceJEV publication readiness check

Generated: 2026-10-24 21:58 Asia/Shanghai (IMRaD rewrite)

## Story lock
Problem → free-form terminal gap → typed choice + pre-emission consistency gate → measured SemEval/VAST.
Memorable: stance is a closed action set; free-form accuracy leaves the decision interface unsolved.

## Claim ↔ numbers
| Dataset | n | Primary | Acc | Schema | Q_AVG | mean max-p | mean κ |
|---------|---|---------|-----|--------|-------|------------|--------|
| SemEval-2016 Task A | 1249 | F_AVG=0.785 | 0.762 | 100% | 2.0 | 0.868 | 0.767 |
| VAST zero-shot | 1460 | F_macro3=0.833 | 0.834 | 100% | 2.0 | 0.907 | 0.784 |

No invented baselines/ablations.

## Submission voice
Banned meta cleared from `paper.tex`. Internal A–E only in `notes/REVIEWER9_GATE.md`.

## Structure
Abstract → Intro → Related → Method (Problem/Principle/Design) → Setup → Results → Discussion → Conclusion → Ethics.
Academic TikZ system figure (Input / Reconstruction / Reasoning / Verification / Typed emission).

## Citations
29 `\bibitem` in `paper.bbl`; 6 Yan Yu keys cited. No undefined citations.

## Compile
- TeX Live 2025 on WIN-1RUTJ70K69H
- `pdflatex` → `bibtex` → `pdflatex` ×2 under `E:\code\jev-stance\paper\`
- **6 pages**, 203991 bytes; no undefined refs; no Overfull/Underfull flagged in summary pass
- Paths: `E:\code\jev-stance\paper\paper.pdf` ; `/workspace/jev-stance/paper/paper.pdf`

## Honest gap (internal)
Matched free-form control, CV ablation, and same-protocol baseline still missing for strong-accept bar — see REVIEWER9_GATE.md. Not blockers for this measured SJ-1 draft.

## Verdict
**READY for internal/adversarial review** — IMRaD rewrite compiled; numbers locked.
