#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
# shellcheck disable=SC1091
set -a; source secrets/.env; set +a
python3 scripts/run_sj1_chunked.py \
  --input data/processed/semeval2016_test.jsonl \
  --out results/sj1_semeval_api.json \
  --preds results/sj1_semeval_api_preds.jsonl \
  --backend api --rounds 1 --batch-size 8 --chunk 40 \
  --experiment SJ-1-semeval2016-test
