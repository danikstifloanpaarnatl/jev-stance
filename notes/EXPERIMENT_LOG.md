# StanceJEV experiment log

Timezone: Asia/Shanghai (UTC+8). All numbers below are **fixture / smoke**, not SemEval-2016 / VAST / P-Stance.

## 2026-10-24 — package + smoke + Jev sample

### Implemented
- Package: `/workspace/jev-stance/src/stancejev/` (`types`, `cr`, `br`, `cv`, `decision`, `pipeline`, `metrics`, `data`, `cli`, `envutil`)
- Fixture: `data/fixtures/synthetic_stance.jsonl` (48 labeled) + stratified `synthetic_stance_sample16.jsonl`
- Script: `scripts/run_sj1_smoke.sh`
- Secrets: local `.env` / `secrets/` (gitignored; not in this repo) for OpenRouter Jev + optional LLM keys

### SJ-1 smoke (mock, full fixture n=48)
- File: `results/sj1_smoke.json`
- Backend: `mock` (keyword heuristic BR; no LLM/Jev)
- **F_AVG = 0.8787**, Q_AVG = 1.0, accuracy = 0.8542, schema_valid_rate = 1.0
- Marking: `[[SMOKE-FIXTURE]]` — do **not** cite as SemEval/VAST result

### SJ-1 Jev sample (real typed decisions, n=16 stratified)
- File: `results/sj1_jev_sample.json` + `sj1_jev_sample_preds.jsonl`
- Backend: `jev` = mock BR explanations + **OpenRouter Decisions API** (`typesafe/jev-1.13`)
- Probe: `results/jev_probe.json` (Favor on vaccine support tweet)
- **F_AVG = 1.0**, Q_AVG = 2.0 (1 BR + 1 Jev / item), accuracy = 1.0, schema_valid_rate = 1.0
- Caveat: synthetic, short, high-signal English; perfect score is **not** generalizable

### mimo BR (blocked)
- `JEVTREE_LLM_*` (local env, not committed) → HTTP **401 Invalid API Key** on
  `token-plan-sgp.xiaomimimo.com`, `api.xiaomimimo.com`, `token-plan-cn.xiaomimimo.com`
  for `mimo-v2.5-pro` / `mimo-v2.6-flash` / `mimo-v2.6-pro`
- Earlier `results/mimo_probe.json` shows a prior successful flash probe (different session/key);
  current local key does not authenticate
- **Blocker for full API BR path:** refresh local `JEVTREE_LLM_API_KEY` (and confirm base URL / model)

### Blockers for full SJ-1 (paper-grade)
1. SemEval-2016 Task 6 / VAST / P-Stance datasets not on box (license; do not commit raw tweets)
2. Working mimo key for CoVer-style batch explanations
3. Optional: pull CoVer `E:\code\CoVer` (train_model / language_models) for closer SLM verifier
4. edgexpert-6861 offline — cannot sync to `…/jev_project/jev-stance/`

### How to re-run
```bash
export PYTHONPATH=/workspace/jev-stance/src
./scripts/run_sj1_smoke.sh
python3 -m stancejev eval \
  --input data/fixtures/synthetic_stance_sample16.jsonl \
  --out results/sj1_jev_sample.json \
  --preds results/sj1_jev_sample_preds.jsonl \
  --backend jev --rounds 1
# when mimo key works:
python3 -m stancejev eval --input ... --backend api --limit 16 --rounds 1
```

## 2026-10-24 — mimo+Jev API sample (fixed base URL)
- Base URL set to https://api.xiaomimimo.com/v1 (cn token-plan returns 401)
- File: results/sj1_api_sample.json
- Backend: api = mimo-v2.6-flash BR + typesafe/jev-1.13 decisions
- n=16 stratified synthetic: F_AVG=1.0, Q_AVG=2.0, schema_valid=1.0
- Still fixture-only; not SemEval/VAST/P-Stance

## 2026-10-24 03:49– (Asia/Shanghai) — SemEval + VAST acquisition + SJ-1

### Datasets
- **VAST**: cloned CSVs from `emilyallaway/zero-shot-stance` `data/VAST/` via raw GitHub.
  - Labels: 0=con→Against, 1=pro→Favor, 2=neutral→None
  - Zero-shot test: `seen?=0` → `data/processed/vast_zero_shot_test.jsonl` (**n=1460**)
- **SemEval-2016 Task 6 Subtask A**: Dropbox mirror used by Sheffield NLP / stance-conditional
  (`dropbox_semeval.zip` → `SemEval2016-Task6-subtaskA-testdata-gold.txt`, training file).
  - Test → `data/processed/semeval2016_test.jsonl` (**n=1249**)
  - Train → `data/processed/semeval2016_train.jsonl` (**n=2814**)
- Convert: `python3 scripts/prepare_datasets.py`
- Raw/processed gitignored (`data/README.md`)

### Mock baselines (not paper primary)
- SemEval mock: `results/sj1_semeval_mock.json` F_AVG=0.1594, schema=1.0
- VAST mock: `results/sj1_vast_mock.json` F_macro3=0.3286, schema=1.0

### Live SJ-1 (api = mimo-v2.6-flash BR + typesafe/jev-1.13)
- Runner: `scripts/run_sj1_chunked.py` (resume-safe, chunk=40, rounds=1, batch=4)
- SemEval full test in progress → `results/sj1_semeval_api.json`
- VAST zero-shot queued after SemEval

### Smoke (n=8 SemEval)
- `results/sj1_semeval_api_smoke8.json` schema=1.0 (tiny prefix; not a claim)

## 2026-10-24 06:56 Asia/Shanghai — SJ-1 SemEval + VAST (api) FINAL

### SemEval-2016 Task A test
- File: `results/sj1_semeval_api.json` (complete)
- n=1249, **F_AVG=0.7854**, Q_AVG=2.0, schema_valid=1.0, accuracy=0.7622
- backend: mimo-v2.6-flash BR + typesafe/jev-1.13

### VAST official zero-shot (`seen?=0`)
- File: `results/sj1_vast_api.json` (complete)
- n=1460, **F_macro3=0.8327**, F_AVG=0.7992, Favor_F1=0.8021, Against_F1=0.7963, None_F1=0.8998, accuracy=0.8342, Q_AVG=2.0, schema_valid=1.0000
- fidelity: mean max p(y)=0.907, mean κ=0.784
- Paper tokens `[[VAST_*]]` filled; see `notes/PUBLICATION_CHECK.md`
