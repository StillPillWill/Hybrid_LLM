# llmy — CUDA inference engine for the looped hybrid-MoE architecture

A from-scratch C++/CUDA inference engine for the V6 (2B config) and 300M
scaled variants of the looped hybrid-MoE language model. Built for Pascal
and later NVIDIA GPUs.

## Design principles

- **No PyTorch at inference.** All matmuls, attention, MoE dispatch, and the
  recurrent scans are hand-written CUDA. cuBLAS is used for the large GEMMs.
- **Python is a shell.** The tokenizer, CLI, and sampling math live in Python.
  The model is a shared library; Python holds an opaque handle and reads back
  logits as a flat buffer.
- **mmap'd weights.** The checkpoint is converted once to a flat binary; the
  engine mmap's it read-only. No pickle, no per-tensor object allocation.
- **One kernel per logical op.** No op decomposition. mHC read is one kernel;
  GDN step is one kernel; banded attention is one kernel.

## Layout

- `include/`: public C++ interfaces and CUDA kernel declarations.
- `src/`: inference engine, checkpoint support, and CUDA kernels.
- `python/`: Python bindings for the native engine.
- `tools/`: checkpoint conversion utilities.

## CLI

From the repository root, the recommended entry point is:

```powershell
python .\llmy_cuda\cli.py --checkpoint-dir C:\models\llmy -n 20
```

If `--model`/`--checkpoint` is omitted, the CLI selects the newest `.llmy` or
`.pt` file below the configured search directories. A `.pt` checkpoint is
converted automatically to a same-name `.llmy` file. Set
`LLMY_CHECKPOINT_DIR` to change the default search location.

Use `-p "one line prompt"` for a one-shot invocation. Without `-p`, enter a
multiline prompt and finish with a blank line or `/run`; use `/exit` to quit.
The default `python` engine uses the validated reference sampler and prints
the full timing, NLL/PPL, adaptive-loop, and VRAM statistics. The experimental
CUDA executable can be selected with `--engine native`; it can also be called
directly at `llmy_cuda\build-vs4\llmy.exe` with a converted `.llmy` path.

## Optional fused kernels in the reference sampler

`sample_standalone.py` can use the fused Sinkhorn and selected-expert MoE
kernels from the native library during FP32 CUDA inference. Enable either path
with `LLMY_COMPILE_SINKHORN=1` or `LLMY_COMPILE_MOE=1`. Set
`LLMY_MHC_CUDA_LIBRARY` to the library path when it is outside `llmy_cuda`'s
build directories. The Sinkhorn path applies to 4-by-4 matrices; the MoE path
applies to one-token decode. Both paths fall back to the PyTorch implementation
when the library or exported function is unavailable.
