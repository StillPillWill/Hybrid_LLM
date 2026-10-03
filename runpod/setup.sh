#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

echo "Installing RunPod system tools"
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y git tmux htop rsync curl

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

echo "Creating the Python environment"
uv venv --system-site-packages .venv --python 3.11
uv pip install --python .venv/bin/python \
    triton numba pyyaml wandb psutil tabulate datasets transformers

echo "Checking the GPU and training dependencies"
.venv/bin/python - <<'PY'
import torch
import triton

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Use a RunPod image with CUDA-enabled PyTorch.")

device = torch.cuda.get_device_properties(0)
print(f"PyTorch CUDA: {torch.version.cuda}")
print(f"GPU: {device.name}")
print(f"VRAM: {device.total_memory / (1024**3):.2f} GiB")
print(f"Triton: {triton.__version__}")
PY

echo "Setup complete. Start training with: bash runpod/run_train.sh"
