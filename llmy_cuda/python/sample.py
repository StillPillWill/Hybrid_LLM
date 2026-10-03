#!/usr/bin/env python3
"""Python sampler for the llmy CUDA engine.

Loads a converted .llmy checkpoint through the shared library via ctypes,
then runs a REPL-style generation loop. The tokenizer is the real Qwen
tokenizer from HuggingFace; the model math runs entirely in C++/CUDA.
"""
from __future__ import annotations

import argparse
import ctypes as C
import math
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Locate libllmy.so (or .dll / .dylib) relative to this script.
# ---------------------------------------------------------------------------

def _find_library() -> str:
    here = Path(__file__).resolve().parent
    if os.name == "nt":
        cuda_roots = []
        cuda_env = os.environ.get("CUDA_PATH")
        if cuda_env:
            cuda_roots.append(Path(cuda_env) / "bin")
        cuda_roots.append(Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"))
        for cuda_bin in cuda_roots:
            if cuda_bin.is_dir():
                try:
                    os.add_dll_directory(str(cuda_bin))
                except (AttributeError, OSError):
                    pass
    for cand in (
        here.parent / "build-vs4" / "llmy.dll",
        here.parent / "build-vs" / "llmy.dll",
        here.parent / "build" / "libllmy.so",
        here.parent / "build" / "libllmy.dylib",
        here.parent / "build" / "llmy.dll",
        here.parent / "libllmy.so",
    ):
        if cand.is_file():
            return str(cand)
    raise FileNotFoundError(
        "llmy.dll/libllmy.so not found. Build the native target first."
    )


def find_latest_checkpoint(search_dirs: list[Path]) -> Path:
    """Find the newest .llmy/.pt checkpoint without scanning unrelated roots."""
    candidates: list[Path] = []
    seen: set[Path] = set()
    for root in search_dirs:
        root = root.expanduser().resolve()
        if not root.exists():
            continue
        for suffix in ("*.llmy", "*.pt"):
            for path in root.rglob(suffix):
                if path not in seen and "build" not in path.parts:
                    seen.add(path)
                    candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            "No .llmy or .pt checkpoint found. Use --checkpoint-dir or --model."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def prepare_checkpoint(path: Path) -> Path:
    """Return a converted .llmy file, converting a .pt source when necessary."""
    path = path.resolve()
    if path.suffix.lower() != ".pt":
        return path
    output = path.with_suffix(".llmy")
    if not output.exists() or output.stat().st_mtime < path.stat().st_mtime:
        converter = Path(__file__).resolve().parent.parent / "tools" / "convert_checkpoint.py"
        print(f"[checkpoint] converting {path} -> {output}")
        subprocess.run(
            [sys.executable, str(converter), str(path), str(output), "--keep-fp32"],
            check=True,
        )
    return output


def read_multiline_prompt() -> str | None:
    """Read a multiline prompt; submit with an empty line, /run, or EOF."""
    print("Enter your prompt. Finish with an empty line (or /run); /exit quits.")
    lines: list[str] = []
    while True:
        try:
            line = input("... " if lines else "Prompt: ")
        except (KeyboardInterrupt, EOFError):
            return None if not lines else "\n".join(lines)
        if line.strip().lower() in ("/exit", "/quit"):
            return None
        if line.strip().lower() == "/run" or (not line and lines):
            return "\n".join(lines)
        lines.append(line)


class LlmyModel:
    """ctypes wrapper for the C ABI."""

    def __init__(self, path: str):
        self.lib = C.CDLL(_find_library())

        # --- signatures ---
        self.lib.llmy_load.argtypes = [C.c_char_p]
        self.lib.llmy_load.restype = C.c_void_p

        self.lib.llmy_free.argtypes = [C.c_void_p]
        self.lib.llmy_free.restype = None

        self.lib.llmy_vocab_size.argtypes = [C.c_void_p]
        self.lib.llmy_vocab_size.restype = C.c_int32

        self.lib.llmy_d_model.argtypes = [C.c_void_p]
        self.lib.llmy_d_model.restype = C.c_int32

        self.lib.llmy_num_loops.argtypes = [C.c_void_p]
        self.lib.llmy_num_loops.restype = C.c_int32

        self.lib.llmy_max_loops.argtypes = [C.c_void_p]
        self.lib.llmy_max_loops.restype = C.c_int32

        self.lib.llmy_set_num_loops.argtypes = [C.c_void_p, C.c_int32]
        self.lib.llmy_set_num_loops.restype = None

        self.lib.llmy_state_new.argtypes = [C.c_void_p]
        self.lib.llmy_state_new.restype = C.c_void_p

        self.lib.llmy_state_free.argtypes = [C.c_void_p]
        self.lib.llmy_state_free.restype = None

        self.lib.llmy_prefill.argtypes = [
            C.c_void_p, C.c_void_p,
            C.POINTER(C.c_int32), C.c_int32,
            C.POINTER(C.c_float),
        ]
        self.lib.llmy_prefill.restype = None

        self.lib.llmy_decode.argtypes = [
            C.c_void_p, C.c_void_p,
            C.c_int32, C.c_int32, C.c_int32,
            C.POINTER(C.c_float),
        ]
        self.lib.llmy_decode.restype = None

        self.lib.llmy_sample.argtypes = [
            C.POINTER(C.c_float), C.c_int32,
            C.c_float, C.c_int32, C.c_float,
            C.c_float,
            C.POINTER(C.c_int32), C.c_int32,
            C.POINTER(C.c_uint64),
        ]
        self.lib.llmy_sample.restype = C.c_int32

        self.lib.llmy_last_executed_loops.argtypes = [C.c_void_p]
        self.lib.llmy_last_executed_loops.restype = C.c_int32

        self.lib.llmy_sync.argtypes = []
        self.lib.llmy_sync.restype = None

        # --- load the model ---
        self.handle = self.lib.llmy_load(path.encode("utf-8"))
        if not self.handle:
            raise RuntimeError(f"llmy_load failed for {path}")

        self.vocab_size = int(self.lib.llmy_vocab_size(self.handle))
        self.d_model = int(self.lib.llmy_d_model(self.handle))
        self.num_loops = int(self.lib.llmy_num_loops(self.handle))
        self.max_loops = int(self.lib.llmy_max_loops(self.handle))

    def close(self) -> None:
        if self.handle:
            self.lib.llmy_free(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    # --- state ---
    def new_state(self):
        s = self.lib.llmy_state_new(self.handle)
        if not s:
            raise RuntimeError("llmy_state_new failed")
        return s

    def free_state(self, s):
        self.lib.llmy_state_free(s)

    # --- forward ---
    def prefill(self, tokens: list[int]) -> "np.ndarray":
        import numpy as np
        n = len(tokens)
        tok_arr = (C.c_int32 * n)(*tokens)
        out = (C.c_float * self.vocab_size)()
        self.lib.llmy_prefill(self.handle, self._state,
                              tok_arr, n, out)
        self._logits[:] = out[:]
        return self._logits

    def decode(self, token: int, position: int, num_loops: int = 0):
        import numpy as np
        out = (C.c_float * self.vocab_size)()
        self.lib.llmy_decode(self.handle, self._state,
                             int(token), int(position), int(num_loops), out)
        self._logits[:] = out[:]
        return self._logits

    def sample(self, logits, temperature=0.7, top_k=50, top_p=0.9,
               repetition_penalty=1.1, recent=None, rng_state=1337):
        """Call the C++ sampler directly on the logits array."""
        import numpy as np
        if not isinstance(logits, np.ndarray):
            logits = np.asarray(logits, dtype=np.float32)
        logits = np.ascontiguousarray(logits.astype(np.float32))
        ptr = logits.ctypes.data_as(C.POINTER(C.c_float))

        if recent:
            recent_arr = (C.c_int32 * len(recent))(*[int(x) for x in recent])
        else:
            recent_arr = (C.c_int32 * 0)()
        rng = C.c_uint64(rng_state)

        return int(self.lib.llmy_sample(
            ptr, self.vocab_size,
            float(temperature), int(top_k), float(top_p),
            float(repetition_penalty),
            recent_arr, len(recent) if recent else 0,
            C.byref(rng),
        )), int(rng.value)

    # --- helpers that cache `_logits` and `_state` ---
    def init_state(self):
        self._state = self.new_state()
        import numpy as np
        self._logits = np.zeros(self.vocab_size, dtype=np.float32)

    def reset(self):
        if getattr(self, "_state", None):
            self.free_state(self._state)
        self.init_state()


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def load_tokenizer():
    """Load the real Qwen tokenizer required for semantic inference."""
    cache = Path(__file__).resolve().parent.parent.parent / ".hf_cache"
    snapshots = sorted((cache / "models--Qwen--Qwen3.5-0.8B-Base" /
                        "snapshots").glob("*/tokenizer.json"))
    if snapshots:
        try:
            from tokenizers import Tokenizer
            raw = Tokenizer.from_file(str(snapshots[0]))

            class FastTokenizer:
                eos_token_id = 248044

                def encode(self, text, add_special_tokens=False):
                    return raw.encode(text, add_special_tokens=add_special_tokens).ids

                def decode(self, ids, skip_special_tokens=True):
                    return raw.decode([int(x) for x in ids],
                                      skip_special_tokens=skip_special_tokens)

            return FastTokenizer(), "qwen-tokenizer.json"
        except Exception:
            pass
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3.5-0.8B-Base",
            cache_dir=str(cache), use_fast=True,
            local_files_only=True,
        )
        return tok, "qwen"
    except Exception as e:
        raise RuntimeError(
            "Qwen tokenizer unavailable locally; refusing to run with a "
            "hash-based fallback because it changes the model input semantics."
        ) from e


# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------

def generate(model: LlmyModel, tokenizer, prompt: str, max_tokens: int,
             temp: float, top_k: int, top_p: float, rep: float,
             num_loops: int, seed: int, rng_state: int):
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids:
        print("(empty prompt)", file=sys.stderr)
        return

    model.reset()
    t0 = time.perf_counter()
    logits = model.prefill(ids)
    t_prefill = time.perf_counter() - t0

    sys.stdout.write("Prediction: ")
    sys.stdout.flush()

    generated: list[int] = []
    logprobs: list[float] = []
    eos_id = getattr(tokenizer, "eos_token_id", None)

    t1 = time.perf_counter()
    for i in range(max_tokens):
        tok, rng_state = model.sample(
            logits, temperature=temp, top_k=top_k, top_p=top_p,
            repetition_penalty=rep, recent=generated or None,
            rng_state=rng_state,
        )
        generated.append(tok)

        # Compute logprob for telemetry (softmax over the returned logits).
        import numpy as np
        z = logits.astype(np.float64)
        m = z.max()
        logZ = m + math.log(np.exp(z - m).sum())
        logprobs.append(float(z[tok] - logZ))

        # Emit text (single-token decode).
        text = tokenizer.decode([tok])
        sys.stdout.write(text)
        sys.stdout.flush()

        if eos_id is not None and tok == eos_id:
            break

        pos = len(ids) + i
        t_decode_start = time.perf_counter()
        logits = model.decode(tok, pos, num_loops=num_loops)
        if i == 0:
            print(f"\n[first decode: {(time.perf_counter()-t_decode_start)*1000:.1f} ms]",
                  file=sys.stderr)

    t_total = time.perf_counter() - t1
    n = len(generated)
    tps = n / t_total if t_total > 0 else 0.0
    avg_nll = -sum(logprobs) / n if n else 0.0
    ppl = math.exp(min(avg_nll, 50.0)) if n else 0.0

    print()
    print("-" * 70)
    print(f"Prefill: {len(ids)} tokens in {t_prefill:.2f}s "
          f"({len(ids) / max(t_prefill, 1e-6):.1f} tok/s)")
    print(f"Decode : {n} tokens in {t_total:.2f}s "
          f"({tps:.2f} tok/s, {1000/tps:.1f} ms/tok)")
    print(f"NLL/PPL: {avg_nll:.4f} / {ppl:.2f}")
    print(f"Last executed loops: {model.lib.llmy_last_executed_loops(model._state)}")
    print("-" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", "--checkpoint", default=None,
                    help="checkpoint path; if omitted, newest checkpoint is selected")
    ap.add_argument("--checkpoint-dir", action="append", type=Path,
                    help="directory to search recursively for the newest checkpoint")
    ap.add_argument("-p", "--prompt", default=None)
    ap.add_argument("-n", "--max-tokens", type=int, default=64)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--rep", type=float, default=1.1)
    ap.add_argument("--loops", type=int, default=0,
                    help="override num_loops (0 = model default)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--greedy", action="store_true",
                    help="argmax sampling (temp=0)")
    args = ap.parse_args()

    if args.model:
        checkpoint = Path(args.model)
    else:
        roots = args.checkpoint_dir or []
        if not roots:
            env_root = os.environ.get("LLMY_CHECKPOINT_DIR")
            if env_root:
                roots.append(Path(env_root))
            # Keep the implicit search in the workspace. Other layouts can
            # be passed with --checkpoint-dir or LLMY_CHECKPOINT_DIR.
            roots.append(Path.cwd())
        checkpoint = find_latest_checkpoint(roots)
        print(f"[checkpoint] newest: {checkpoint}")
    checkpoint = prepare_checkpoint(checkpoint)

    tokenizer, kind = load_tokenizer()
    print(f"[tokenizer] {kind} (vocab={len(tokenizer) if hasattr(tokenizer, '__len__') else '?'})")

    with LlmyModel(str(checkpoint)) as model:
        print(f"[model] vocab={model.vocab_size}, d_model={model.d_model}, "
              f"loops={model.num_loops}, max_loops={model.max_loops}")
        model.init_state()

        temp = 0.0 if args.greedy else args.temp
        top_k = 0 if args.greedy else args.top_k
        top_p = 1.0 if args.greedy else args.top_p
        rep = 1.0 if args.greedy else args.rep

        def run(prompt: str):
            generate(model, tokenizer, prompt, args.max_tokens,
                     temp, top_k, top_p, rep, args.loops,
                     args.seed, args.seed)

        if args.prompt:
            run(args.prompt)
            return

        print("\n--- interactive multiline mode ---")
        while True:
            p = read_multiline_prompt()
            if p is None:
                break
            if not p.strip():
                break
            run(p)
            print()


if __name__ == "__main__":
    main()
