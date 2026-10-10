#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${CLARIFYGPT_MODEL:-gpt-5.6-luna}"
OUTPUT_DIR="${CLARIFYGPT_OUTPUT_DIR:-$ROOT_DIR/baselines/ClarifyGPT/output/clarifycodebench/$MODEL/clarifygpt}"

exec "$PYTHON_BIN" -m baselines.ClarifyGPT.integrations.clarifycodebench.main \
  --benchmark clarifycodebench \
  --model "$MODEL" \
  --batch-size "${CLARIFYGPT_BATCH_SIZE:-3}" \
  --max-tokens "${CLARIFYGPT_MAX_TOKENS:-8192}" \
  --max-retries "${CLARIFYGPT_MAX_RETRIES:-5}" \
  --candidate-count "${CLARIFYGPT_CANDIDATE_COUNT:-25}" \
  --seed-input-count "${CLARIFYGPT_SEED_INPUT_COUNT:-5}" \
  --test-input-count "${CLARIFYGPT_TEST_INPUT_COUNT:-20}" \
  --candidate-timeout "${CLARIFYGPT_CANDIDATE_TIMEOUT:-2}" \
  --demonstrations "${CLARIFYGPT_DEMONSTRATIONS:-3}" \
  --representative-seed "${CLARIFYGPT_REPRESENTATIVE_SEED:-0}" \
  --input-mutation-seed "${CLARIFYGPT_INPUT_MUTATION_SEED:-0}" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
