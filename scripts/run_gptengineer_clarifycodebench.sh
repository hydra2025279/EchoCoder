#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${GPTENGINEER_MODEL:-gpt-5.6-luna}"
OUTPUT_DIR="${GPTENGINEER_OUTPUT_DIR:-$ROOT_DIR/baselines/gptengineer/output/clarifycodebench/$MODEL/gptengineer}"

exec "$PYTHON_BIN" -m baselines.gptengineer.integrations.clarifycodebench.main \
  --benchmark clarifycodebench \
  --model "$MODEL" \
  --question-budget "${GPTENGINEER_QUESTION_BUDGET:-6}" \
  --batch-size "${GPTENGINEER_BATCH_SIZE:-3}" \
  --max-tokens "${GPTENGINEER_MAX_TOKENS:-8192}" \
  --max-retries "${GPTENGINEER_MAX_RETRIES:-5}" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
