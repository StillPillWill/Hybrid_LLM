#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_dir="${LLMY_RUN_DIR:-runs/a5000_moe24}"
mkdir -p "$run_dir"
touch "$run_dir/.supervisor_initialized"

if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi

export LLMY_RUN_DIR="$run_dir"
export LLMY_DEBLOAT_SECONDS="${LLMY_DEBLOAT_SECONDS:-7200}"

nohup bash runpod/debloat_supervisor.sh >/dev/null 2>&1 &

echo "Supervisor started (PID $!)"
