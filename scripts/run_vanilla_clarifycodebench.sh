#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${VANILLA_MODEL:-gpt-5.6-luna}"
QUESTION_BUDGET="${VANILLA_QUESTION_BUDGET:-6}"
BATCH_SIZE="${VANILLA_BATCH_SIZE:-6}"
MAX_TOKENS="${VANILLA_MAX_TOKENS:-8192}"
MAX_RETRIES="${VANILLA_MAX_RETRIES:-5}"
METHOD="vanilla"
BENCHMARK="clarifycodebench"
OUTPUT_ROOT="${VANILLA_OUTPUT_ROOT:-$ROOT_DIR/baselines/vanilla/output}"
MODEL_SLUG="$(printf '%s' "$MODEL" | tr -cs 'A-Za-z0-9._-' '_')"
OUTPUT_METHOD="$METHOD"
OUTPUT_DIR="${VANILLA_OUTPUT_DIR:-$OUTPUT_ROOT/$BENCHMARK/$MODEL_SLUG/$OUTPUT_METHOD}"

printf '[vanilla][setup] model=%s iterations=%s output=%s\n' \
  "$MODEL" "$QUESTION_BUDGET" "$OUTPUT_DIR"
printf '[vanilla][setup] auto-resume enabled; completed task IDs will be skipped\n'

COMMON_ARGS=(
  --benchmark "$BENCHMARK"
  --model "$MODEL"
  --question-budget "$QUESTION_BUDGET"
  --batch-size "$BATCH_SIZE"
  --max-tokens "$MAX_TOKENS"
  --max-retries "$MAX_RETRIES"
  --output-dir "$OUTPUT_DIR"
)

exec "$PYTHON_BIN" -m baselines.vanilla.main "${COMMON_ARGS[@]}" "$@"
