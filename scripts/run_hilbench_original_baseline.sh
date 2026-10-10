#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export HILBENCH_MODE=baseline
export HILBENCH_TASK_TYPE="${HILBENCH_TASK_TYPE:-swe}"
export HILBENCH_PASSES="${HILBENCH_PASSES:-1}"
exec "$SCRIPT_DIR/_run_hilbench_original_mode.sh" "$@"
