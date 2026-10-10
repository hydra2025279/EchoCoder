#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export HILBENCH_MODE=full_info
exec "$SCRIPT_DIR/_run_hilbench_original_mode.sh" "$@"
