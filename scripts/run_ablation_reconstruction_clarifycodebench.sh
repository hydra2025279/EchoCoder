#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${PYTHON_BIN:-}" && -x "$ROOT_DIR/benchmarks/hil-bench/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/benchmarks/hil-bench/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi
if [[ "$PYTHON_BIN" == */* ]]; then
  export PATH="$(dirname -- "$PYTHON_BIN"):$PATH"
fi
MODEL="${ABLATION_RECONSTRUCTION_MODEL:-gpt-5.6-luna}"
ITERATION_BUDGET="${ABLATION_RECONSTRUCTION_ITERATION_BUDGET:-6}"
BATCH_SIZE="${ABLATION_RECONSTRUCTION_BATCH_SIZE:-5}"
MAX_TOKENS="${ABLATION_RECONSTRUCTION_MAX_TOKENS:-8192}"
MAX_RETRIES="${ABLATION_RECONSTRUCTION_MAX_RETRIES:-5}"
OUTPUT_ROOT="${ABLATION_RECONSTRUCTION_OUTPUT_ROOT:-$ROOT_DIR/baselines/ablation_reconstruction/output}"
MODEL_SLUG="$(printf '%s' "$MODEL" | tr -cs 'A-Za-z0-9._-' '_')"
OUTPUT_DIR="${ABLATION_RECONSTRUCTION_OUTPUT_DIR:-$OUTPUT_ROOT/clarifycodebench/$MODEL_SLUG/ablation_reconstruction}"

printf '[ablation_reconstruction][setup] model=%s iterations=%s output=%s\n' \
  "$MODEL" "$ITERATION_BUDGET" "$OUTPUT_DIR"
printf '[ablation_reconstruction][setup] auto-resume enabled; completed task IDs will be skipped\n'

exec "$PYTHON_BIN" -m baselines.ablation_reconstruction.main \
  --benchmark clarifycodebench \
  --model "$MODEL" \
  --iteration-budget "$ITERATION_BUDGET" \
  --batch-size "$BATCH_SIZE" \
  --max-tokens "$MAX_TOKENS" \
  --max-retries "$MAX_RETRIES" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
