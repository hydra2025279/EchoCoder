#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
HILBENCH_DIR="$ROOT_DIR/benchmarks/hil-bench"

: "${LLM_BASE_URL:?Export LLM_BASE_URL before running this script}"
: "${LLM_API_KEY:?Export LLM_API_KEY before running this script}"

MODE="${HILBENCH_MODE:?HILBENCH_MODE must be set by a mode wrapper}"
MODEL="${HILBENCH_MODEL:-gpt-5.6-luna}"
LITELLM_MODEL="${HILBENCH_LITELLM_MODEL:-azure/$MODEL}"
TASK_TYPE="${HILBENCH_TASK_TYPE:-swe}"
PASSES="${HILBENCH_PASSES:-3}"
TASK_PARALLEL="${HILBENCH_TASK_PARALLEL:-${HILBENCH_WORKERS:-${HILBENCH_CONCURRENCY:-1}}}"
NUM_CONCURRENT="${HILBENCH_CONCURRENCY:-$TASK_PARALLEL}"
NUM_WORKERS="${HILBENCH_WORKERS:-$TASK_PARALLEL}"
MAX_STEPS="${HILBENCH_MAX_STEPS:-200}"
PER_INSTANCE_COST_LIMIT="${HILBENCH_COST_LIMIT:-100}"
OUTPUT_DIR="${HILBENCH_OUTPUT_DIR:-$HILBENCH_DIR/results/original_${MODE}/$MODEL}"
WORKSPACE_DIR="${HILBENCH_WORKSPACE_DIR:-$HILBENCH_DIR/.workspaces}"
RUN_NAME="${HILBENCH_RUN_NAME:-${TASK_TYPE}_${MODEL}_${MODE}}"
NUM_DATAPOINTS="${HILBENCH_NUM_DATAPOINTS:-}"
IMAGE_BATCH_SIZE="${HILBENCH_IMAGE_BATCH_SIZE:-$TASK_PARALLEL}"

export HIL_BENCH_ASSET_ROOT="${HIL_BENCH_ASSET_ROOT:-$HILBENCH_DIR/.assets}"
export ECHOCODER_COST_CONFIG="${ECHOCODER_COST_CONFIG:-$ROOT_DIR/cost.yaml}"
export AZURE_API_KEY="$LLM_API_KEY"
export AZURE_API_BASE="$LLM_BASE_URL"
export LITELLM_API_KEY="$LLM_API_KEY"
export LITELLM_USER="${LITELLM_USER:-hilbench-original}"
export ASK_HUMAN_LITELLM_BASE_URL="$LLM_BASE_URL"

case "$MODE" in
  baseline|ask_human|full_info) ;;
  *) echo "Unsupported HILBENCH_MODE: $MODE" >&2; exit 2 ;;
esac
case "$TASK_TYPE" in
  swe) ;;
  *) echo "This artifact bundles HiL-Bench SWE tasks only" >&2; exit 2 ;;
esac
if ! [[ "$TASK_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
  echo "HILBENCH_TASK_PARALLEL must be a positive integer" >&2
  exit 2
fi
if [[ "$NUM_CONCURRENT" != "$TASK_PARALLEL" || "$NUM_WORKERS" != "$TASK_PARALLEL" ]]; then
  echo "HILBENCH_CONCURRENCY and HILBENCH_WORKERS must equal HILBENCH_TASK_PARALLEL" >&2
  exit 2
fi
if [[ "$TASK_TYPE" == "swe" || "$TASK_TYPE" == "both" ]]; then
  if [[ "$IMAGE_BATCH_SIZE" != "$TASK_PARALLEL" ]]; then
    echo "For SWE, HILBENCH_IMAGE_BATCH_SIZE must equal HILBENCH_TASK_PARALLEL" >&2
    exit 2
  fi
fi

cd "$HILBENCH_DIR"
command -v uv >/dev/null 2>&1 || {
  echo "uv is required. Install it first, then rerun this script." >&2
  exit 2
}

mkdir -p "$OUTPUT_DIR" "$WORKSPACE_DIR"
uv sync

GENERATED_DIR="$(mktemp -d "${TMPDIR:-/tmp}/hilbench-${MODE}.XXXXXX")"
trap 'rm -rf "$GENERATED_DIR"' EXIT

export HILBENCH_GENERATED_DIR="$GENERATED_DIR"
export HILBENCH_LITELLM_MODEL="$LITELLM_MODEL"
export HILBENCH_AZURE_API_VERSION="${AZURE_API_VERSION:-}"

uv run python - <<'PY'
import os
from pathlib import Path

import yaml

root = Path.cwd()
generated = Path(os.environ["HILBENCH_GENERATED_DIR"])
mode = os.environ["HILBENCH_MODE"]
model = os.environ["HILBENCH_LITELLM_MODEL"]
api_version = os.environ.get("HILBENCH_AZURE_API_VERSION", "").strip()

sources = {
    "swe": (
        root / "configs/swe/ask_config_gpt_5-4.yaml"
        if mode == "ask_human"
        else root / "configs/swe/default_swe_config.yaml"
    ),
}

configs = {}
for task_type, source in sources.items():
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["hosting"] = {
        "type": "self_hosted",
        "api_base_env": "LLM_BASE_URL",
    }
    model_config = config.setdefault("agent", {}).setdefault("model", {})
    model_config["name"] = model
    model_config["api_key"] = "$LLM_API_KEY"
    model_config["temperature"] = None
    if api_version:
        model_config["api_version"] = api_version
    destination = generated / f"{task_type}_{mode}.yaml"
    destination.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    configs[task_type] = str(destination)

mapping = {task_type: {mode: {model: path}} for task_type, path in configs.items()}
(generated / "config_mappings.yaml").write_text(
    yaml.safe_dump(mapping, sort_keys=False, allow_unicode=False),
    encoding="utf-8",
)

judge = {
    "model": model,
    "hosting": {"type": "provider_direct"},
}
(generated / "judge_config.yaml").write_text(
    yaml.safe_dump(judge, sort_keys=False, allow_unicode=False),
    encoding="utf-8",
)
PY

COMMAND=(
  uv run python run_hil_bench.py
  --task-type "$TASK_TYPE"
  --task-parallel "$TASK_PARALLEL"
  --models "$LITELLM_MODEL"
  --modes "$MODE"
  --passes "$PASSES"
  --num-concurrent-datapoints "$NUM_CONCURRENT"
  --num-workers "$NUM_WORKERS"
  --max-steps "$MAX_STEPS"
  --per-instance-cost-limit "$PER_INSTANCE_COST_LIMIT"
  --agent-config-mapping "$GENERATED_DIR/config_mappings.yaml"
  --judge-config "$GENERATED_DIR/judge_config.yaml"
  --output-dir "$OUTPUT_DIR"
  --workspace-dir "$WORKSPACE_DIR"
  --run-name "$RUN_NAME"
)

if [[ -n "$NUM_DATAPOINTS" ]]; then
  COMMAND+=(--num-datapoints "$NUM_DATAPOINTS")
fi
if [[ "$TASK_TYPE" == "swe" || "$TASK_TYPE" == "both" ]]; then
  COMMAND+=(--image-batch-size "$IMAGE_BATCH_SIZE")
fi
if [[ "${HILBENCH_RERUN:-0}" == "1" ]]; then
  COMMAND+=(--rerun)
fi
COMMAND+=("$@")

printf '[hilbench-original-%s] task_type=%s model=%s passes=%s output=%s\n' \
  "$MODE" "$TASK_TYPE" "$LITELLM_MODEL" "$PASSES" "$OUTPUT_DIR"
printf '[hilbench-original-%s] task_parallel=%s\n' "$MODE" "$TASK_PARALLEL"
printf '[hilbench-original-%s] asset_root=%s\n' "$MODE" "$HIL_BENCH_ASSET_ROOT"
if [[ "$TASK_TYPE" == "swe" || "$TASK_TYPE" == "both" ]]; then
  printf '[hilbench-original-%s] image_batch_size=%s\n' "$MODE" "$IMAGE_BATCH_SIZE"
fi
printf '[hilbench-original-%s] command:' "$MODE"
printf ' %q' "${COMMAND[@]}"
printf '\n'

if [[ "${HILBENCH_DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

"${COMMAND[@]}"
