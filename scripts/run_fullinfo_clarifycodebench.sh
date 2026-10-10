#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${FULLINFO_MODEL:-gpt-5.6-luna}"
BATCH_SIZE="${FULLINFO_BATCH_SIZE:-6}"
MAX_TOKENS="${FULLINFO_MAX_TOKENS:-8192}"
MAX_RETRIES="${FULLINFO_MAX_RETRIES:-5}"
RELEASE_VERSION="${FULLINFO_RELEASE_VERSION:-v6}"
OUTPUT_ROOT="${FULLINFO_OUTPUT_ROOT:-$ROOT_DIR/baselines/fullinfo/output}"
MODEL_SLUG="$(printf '%s' "$MODEL" | tr -cs 'A-Za-z0-9._-' '_')"
METHOD="fullinfo"
OUTPUT_DIR="${FULLINFO_OUTPUT_DIR:-$OUTPUT_ROOT/clarifycodebench/$MODEL_SLUG/$METHOD}"

printf '[fullinfo][setup] model=%s release=%s output=%s\n' \
  "$MODEL" "$RELEASE_VERSION" "$OUTPUT_DIR"
printf '[fullinfo][setup] uses complete LiveCodeBench question_content; auto-resume enabled\n'

COMMON_ARGS=(
  --benchmark clarifycodebench
  --model "$MODEL"
  --batch-size "$BATCH_SIZE"
  --max-tokens "$MAX_TOKENS"
  --max-retries "$MAX_RETRIES"
  --release-version "$RELEASE_VERSION"
  --output-dir "$OUTPUT_DIR"
)

exec "$PYTHON_BIN" -m baselines.fullinfo.main "${COMMON_ARGS[@]}" "$@"
