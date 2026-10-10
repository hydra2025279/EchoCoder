#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
: "${LLM_BASE_URL:?Export LLM_BASE_URL before running this script}"
: "${LLM_API_KEY:?Export LLM_API_KEY before running this script}"
export AZURE_API_KEY="$LLM_API_KEY"
export AZURE_API_BASE="$LLM_BASE_URL"
export LITELLM_API_KEY="$LLM_API_KEY"

MODEL="${ECHOCODER_MODEL:-gpt-5.6-luna}"
SWEAGENT_MODEL="${ECHOCODER_SWEAGENT_MODEL:-}"
K_R="${ECHOCODER_KR:-3}"
K_Q="${ECHOCODER_KQ:-2}"
BATCH_SIZE="${ECHOCODER_BATCH_SIZE:-1}"
TASK_PARALLEL="${ECHOCODER_TASK_PARALLEL:-1}"
MAX_RETRIES="${ECHOCODER_MAX_RETRIES:-5}"
EVAL_TIMEOUT="${ECHOCODER_EVAL_TIMEOUT:-1800}"
SWEAGENT_MAX_STEPS="${SWEAGENT_MAX_STEPS:-200}"
SWEAGENT_TIMEOUT="${SWEAGENT_TIMEOUT:-21600}"
SWEAGENT_COST_LIMIT="${SWEAGENT_COST_LIMIT:-100}"
SWEAGENT_CALL_LIMIT="${SWEAGENT_CALL_LIMIT:-0}"
SWEAGENT_CONFIG="${SWEAGENT_CONFIG:-$ROOT_DIR/benchmarks/hil-bench/configs/swe/default_swe_config.yaml}"
SWEAGENT_PYTHON="${ECHOCODER_SWEAGENT_PYTHON:-$ROOT_DIR/benchmarks/hil-bench/.venv/bin/python}"
PYTHON_BIN="${PYTHON_BIN:-$SWEAGENT_PYTHON}"
if [[ ! -x "$SWEAGENT_PYTHON" ]] || ! PYTHONPATH="$ROOT_DIR/benchmarks/hil-bench/SWE-agent${PYTHONPATH:+:$PYTHONPATH}" \
  "$SWEAGENT_PYTHON" -c 'import swerex; import sweagent' >/dev/null 2>&1; then
  printf '%s\n' '[echocoder_sweagent][setup] HiL-Bench SWE-agent environment is unavailable.' >&2
  printf '%s\n' "Run 'cd $ROOT_DIR/benchmarks/hil-bench && uv sync' or set ECHOCODER_SWEAGENT_PYTHON." >&2
  exit 2
fi
METHOD="echocoder_sweagent"
for argument in "$@"; do
  if [[ "$argument" == "--progressive-edit" ]]; then
    METHOD="echocoder_sweagent_progressive"
  fi
done
BENCHMARK="hilbench_swe"
OUTPUT_ROOT="${ECHOCODER_OUTPUT_ROOT:-$ROOT_DIR/echocoder/output}"
MODEL_SLUG="$(printf '%s' "$MODEL" | tr -cs 'A-Za-z0-9._-' '_')"
OUTPUT_DIR="${ECHOCODER_OUTPUT_DIR:-$OUTPUT_ROOT/$BENCHMARK/$MODEL_SLUG/$METHOD}"

printf '[echocoder_sweagent][setup] model=%s Kr=%s Kq=%s output=%s\n' \
  "$MODEL" "$K_R" "$K_Q" "$OUTPUT_DIR"
printf '[echocoder_sweagent][setup] auto-resume enabled; completed task IDs will be skipped\n'

SWEAGENT_MODEL_ARGS=()
if [[ -n "$SWEAGENT_MODEL" ]]; then
  SWEAGENT_MODEL_ARGS=(--sweagent-model "$SWEAGENT_MODEL")
fi

exec "$PYTHON_BIN" -m echocoder.integrations.hilbench.main \
  --task-type swe \
  --mode echocoder_sweagent \
  --model "$MODEL" \
  "${SWEAGENT_MODEL_ARGS[@]}" \
  --reconstruction-budget "$K_R" \
  --question-budget "$K_Q" \
  --batch-size "$BATCH_SIZE" \
  --task-parallel "$TASK_PARALLEL" \
  --max-retries "$MAX_RETRIES" \
  --eval-timeout "$EVAL_TIMEOUT" \
  --evaluate-each-code-state \
  --sweagent-config "$SWEAGENT_CONFIG" \
  --sweagent-max-steps "$SWEAGENT_MAX_STEPS" \
  --sweagent-timeout "$SWEAGENT_TIMEOUT" \
  --sweagent-cost-limit "$SWEAGENT_COST_LIMIT" \
  --sweagent-call-limit "$SWEAGENT_CALL_LIMIT" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
