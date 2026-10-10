#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${ECHOCODER_MODEL:-gpt-5.6-luna}"

K_R="${ECHOCODER_KR:-3}"
K_Q="${ECHOCODER_KQ:-2}"

BATCH_SIZE="${ECHOCODER_BATCH_SIZE:-3}"
MAX_TOKENS="${ECHOCODER_MAX_TOKENS:-8192}"
MAX_RETRIES="${ECHOCODER_MAX_RETRIES:-5}"
METHOD="echocoder"
for arg in "$@"; do
  if [[ "$arg" == "--no-history" ]]; then
    METHOD="echocoder_no_history"
    break
  fi
done
BENCHMARK="clarifycodebench"
OUTPUT_ROOT="${ECHOCODER_OUTPUT_ROOT:-$ROOT_DIR/echocoder/output}"
MODEL_SLUG="$(printf '%s' "$MODEL" | tr -cs 'A-Za-z0-9._-' '_')"
OUTPUT_DIR="${ECHOCODER_OUTPUT_DIR:-$OUTPUT_ROOT/$BENCHMARK/$MODEL_SLUG/$METHOD}"

printf '[echocoder][setup] model=%s Kr=%s Kq=%s output=%s\n' \
  "$MODEL" "$K_R" "$K_Q" "$OUTPUT_DIR"
printf '[echocoder][setup] auto-resume enabled; completed task IDs will be skipped\n'

exec "$PYTHON_BIN" -m echocoder.integrations.clarifycodebench.main \
  --model "$MODEL" \
  --reconstruction-budget "$K_R" \
  --question-budget "$K_Q" \
  --batch-size "$BATCH_SIZE" \
  --max-tokens "$MAX_TOKENS" \
  --max-retries "$MAX_RETRIES" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
