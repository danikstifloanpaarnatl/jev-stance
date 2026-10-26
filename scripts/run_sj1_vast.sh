#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
# shellcheck disable=SC1091
set -a; source secrets/.env; set +a
python3 scripts/run_sj1_chunked.py \
  --input data/processed/vast_zero_shot_test.jsonl \
  --out results/sj1_vast_api.json \
  --preds results/sj1_vast_api_preds.jsonl \
  --backend api --rounds 1 --batch-size 8 --chunk 40 \
  --experiment SJ-1-vast-zero-shot
