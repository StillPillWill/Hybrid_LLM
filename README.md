# LLMY

LLMY is a looped hybrid-MoE language model. This repository contains the
PyTorch reference implementation, an A5000 RunPod training path, and inference
tools for saved checkpoints.

## RunPod training

The training launcher expects a RunPod image with a working NVIDIA PyTorch
installation. From the repository root, install the remaining dependencies:

```bash
bash runpod/setup.sh
```

Start or resume training in the foreground:

```bash
bash runpod/run_train.sh
```

The default configuration is `runpod/config_a5000_moe24.yaml`; a config path
and run directory can be passed as the first and second arguments. The
launcher reads streamed training data from `data/streamed_fineweb`.
The configuration requires an online Weights & Biases run, so authenticate the
RunPod environment before starting training.

For supervised restarts with periodic CUDA cleanup, use:

```bash
bash runpod/launch_moe24.sh
```

Set `LLMY_RUN_DIR`, `LLMY_CONFIG`, or `LLMY_DEBLOAT_SECONDS` before launch to
change the run directory, configuration, or restart interval.

## Inference

`sample_standalone.py` is the self-contained PyTorch sampler for model
checkpoints:

```bash
python sample_standalone.py --checkpoint /path/to/checkpoint.pt --prompt "Hello" --max-tokens 64
```

For an interactive prompt, omit `--prompt`. The optional native CUDA engine
and fused inference kernels live in `llmy_cuda/`; see
[`llmy_cuda/README.md`](llmy_cuda/README.md) for build and CLI details.
Local inference requires a compatible PyTorch installation and Transformers.
Set `LLMY_CHECKPOINT_ARCHIVE` to include an additional directory when the
sampler searches for saved checkpoints.
Add `--gui` to serve the sampler through the local browser interface in
`chat_gui.py`.

Set `LLMY_RUNPOD_SSH_TARGET` or pass `--ssh-target` to configure the
checkpoint backup helper.

## Project layout

- `model/`, `optim/`, `train/`, `data/`, and `eval/` contain the model and
  training dependencies.
- `runpod/` contains the A5000 launch path, configuration, and runtime
  helpers.
- `sample_standalone.py` and `llmy_cuda/` contain the inference paths.

Training data, checkpoints, and run output are local runtime assets and are
not included in this repository.
