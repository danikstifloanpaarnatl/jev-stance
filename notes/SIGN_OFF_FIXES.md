# StanceJEV sign-off fixes (Hazel Bloomer preprint)

Date: 2026-10-25 (Asia/Shanghai)

## Author / metadata
- Replaced Anonymous Authors / Anonymous submission with **Hazel Bloomer**, Independent Researcher, `floanpaarnatl@outlook.com` (from jevtree/hijack arXiv template).
- Header left: Hazel Bloomer; right: StanceJEV.
- Added `\hypersetup` pdftitle / pdfauthor / pdfsubject / pdfkeywords.

## Must-fix content
1. **Q_AVG** — Defined as remote LLM/decision API calls only. Local lexicon consistency verifier and keep-all CR (no embedder call) are excluded. Under M=1: one mimo BR + one Jev → Q_AVG=2.0. Clarified in Intro, BR paragraph, DecisionSOP q_i, and Experiments.
2. **SemEval zero-shot** — Explicit: official SemEval-2016 Task A **test** only (n=1249); no SemEval train, demos, or target-specific supervision. Softened confusing “target-specific training” phrasing. VAST remains official seen?=0.
3. **Algorithm** — `\mathcal{R}` is an ordered list with append; `\Require ... M\ge 1`; multi-round check uses `length(\mathcal{R})`.
4. **Association wording** — Replaced “strong positive association” / “strongly associated” with monotonic rise across max-p bins (abstract + analysis).
5. **Reproducibility** — Experiments now name: mimo-v2.6-flash @ api.xiaomimimo.com (temp 0.0); OpenRouter typesafe/jev-1.13; local CV (JS base-2 bits, λ=0.35, τ=0.55, M=1, batch=4); keep-all CR; no invented seed; DecisionSOP JSONL for prompts/outputs. Method CR describes design; Experiments state SJ-1 keep-all path.

## Not done / out of scope
- No new baselines or ablations invented.
- No Reviewer-9 / 发布会 / planning meta in paper.tex.

## Sync targets
- Box: `/workspace/jev-stance/`
- Windows: `E:\code\jev-stance\` (and staging under `C:\Users\Administrator\` as needed)
- Server: `/home/qzqdz/Desktop/project/jev_project/jev-stance/` (paper + src/scripts/results/notes/data/processed)
