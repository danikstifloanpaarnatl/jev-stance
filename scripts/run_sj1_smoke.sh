#!/usr/bin/env bash
# SJ-1 smoke: mock backend on bundled synthetic fixture (no API key required).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
python3 -m stancejev smoke --out "${ROOT}/results/sj1_smoke.json" --rounds 2
echo "OK: ${ROOT}/results/sj1_smoke.json"
