#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export LLMY_RESUME_ACTIVATION_OFFLOAD="0"
export CUDA_DEVICE_MAX_CONNECTIONS="1"
export LLMY_DIFF_MOE="1"

config="${1:-runpod/config_a5000_moe24.yaml}"
run_dir="${2:-runs/a5000_finemath4plus_v1}"
data_dir="data/streamed_fineweb"

echo "Starting A5000 training"
echo "Config:  $config"
echo "Run dir: $run_dir"

python3 -u runpod/train_a5000_ultimate.py \
    --config "$config" \
    --data-dir "$data_dir" \
    --run-dir "$run_dir" \
    --device cuda \
    --resume
