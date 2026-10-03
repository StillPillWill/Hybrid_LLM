#!/usr/bin/env python3
"""User-facing CLI for the native CUDA llmy engine.

It discovers the newest checkpoint, converts a .pt source when necessary,
then delegates inference to the tested native executable.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import re
from pathlib import Path


def find_latest(roots: list[Path]) -> Path:
    candidates: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        root = root.expanduser().resolve()
        if not root.exists():
            continue
        for pattern in ("*.llmy", "*.pt"):
            for p in root.rglob(pattern):
                if p not in seen and "build" not in p.parts:
                    seen.add(p)
                    candidates.append(p)
    if not candidates:
        raise FileNotFoundError("No .llmy or .pt checkpoint found")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def prepare_checkpoint(path: Path) -> Path:
    path = path.resolve()
    if path.suffix.lower() != ".pt":
        return path
    out = path.with_suffix(".llmy")
    if not out.exists() or out.stat().st_mtime < path.stat().st_mtime:
        converter = Path(__file__).parent / "tools" / "convert_checkpoint.py"
        print(f"[checkpoint] converting {path} -> {out}", flush=True)
        subprocess.run(
            [sys.executable, str(converter), str(path), str(out), "--keep-fp32"],
            check=True,
        )
    return out


def find_exe() -> Path:
    here = Path(__file__).resolve().parent
    for p in (here / "build-vs4" / "llmy.exe",
              here / "build-vs" / "llmy.exe",
              here / "build" / "llmy.exe"):
        if p.exists():
            return p
    raise FileNotFoundError("llmy.exe not found; build llmy_cuda first")


def load_tokenizer(name: str):
    cache = os.environ.get("LLMY_TOKENIZER_CACHE")
    if cache is None:
        cache = str(Path(__file__).resolve().parent.parent / ".hf_cache")
    snapshot_root = Path(cache) / "models--Qwen--Qwen3.5-0.8B-Base" / "snapshots"
    snapshots = sorted(snapshot_root.glob("*/tokenizer.json"))
    if snapshots and name == "Qwen/Qwen3.5-0.8B-Base":
        try:
            from tokenizers import Tokenizer
            raw = Tokenizer.from_file(str(snapshots[0]))

            class FastTokenizer:
                eos_token_id = 248044

                def __init__(self):
                    self._chat_tokenizer = None

                def encode(self, text, add_special_tokens=False):
                    return raw.encode(text, add_special_tokens=add_special_tokens).ids

                def decode(self, ids, skip_special_tokens=True):
                    return raw.decode([int(x) for x in ids],
                                      skip_special_tokens=skip_special_tokens)

                def apply_chat_template(self, messages, **kwargs):
                    # The standalone sampler uses the fast tokenizer for normal
                    # text, but delegates chat rendering to Transformers. Keep
                    # that split so the native CLI produces the same IDs for
                    # distillation checkpoints without slowing ordinary prompts.
                    if self._chat_tokenizer is None:
                        from transformers import AutoTokenizer
                        self._chat_tokenizer = AutoTokenizer.from_pretrained(
                            name,
                            cache_dir=cache,
                            use_fast=True,
                            local_files_only=True,
                        )
                    return self._chat_tokenizer.apply_chat_template(messages, **kwargs)

            return FastTokenizer()
        except Exception:
            pass
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "The real tokenizer requires transformers. Install it with: "
            "pip install transformers"
        ) from exc
    kwargs = {"use_fast": True, "cache_dir": cache}
    try:
        return AutoTokenizer.from_pretrained(name, local_files_only=True, **kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load tokenizer {name!r} locally. Set --tokenizer or "
            "place the tokenizer in the local Hugging Face cache."
        ) from exc


def is_distillation_checkpoint(path: Path) -> bool:
    """Match sample_standalone's checkpoint-dependent prompt formatting."""
    source = path.resolve()
    if source.suffix.lower() == ".llmy":
        sibling = source.with_suffix(".pt")
        if sibling.is_file():
            source = sibling
    if source.suffix.lower() != ".pt" or not source.is_file():
        return False
    try:
        # Meta tensors let us inspect only the checkpoint metadata without
        # allocating the multi-GB model state a second time.
        import torch
        checkpoint = torch.load(
            str(source), map_location="meta", weights_only=False
        )
        data = checkpoint.get("config", {}).get("data", {})
        after_tokens = data.get("after_tokens", {})
        return str(after_tokens.get("loader", "")).lower() == "distillation"
    except Exception as exc:
        print(f"[tokenizer] could not inspect checkpoint metadata: {exc}; using raw prompt tokens", file=sys.stderr)
        return False


def multiline_prompt() -> str | None:
    print("Enter a multiline prompt. Finish with a blank line or /run; /exit quits.")
    lines: list[str] = []
    while True:
        try:
            line = input("... " if lines else "Prompt: ")
        except (EOFError, KeyboardInterrupt):
            return None if not lines else "\n".join(lines)
        command = line.strip().lower()
        if command in ("/exit", "/quit"):
            return None
        if command == "/run" or (not line and lines):
            return "\n".join(lines)
        lines.append(line)


def main() -> int:
    ap = argparse.ArgumentParser(description="CUDA llmy native inference CLI")
    ap.add_argument("-m", "--model", "--checkpoint", type=Path)
    ap.add_argument("--checkpoint-dir", action="append", type=Path)
    ap.add_argument("-p", "--prompt")
    ap.add_argument("-n", "--tokens", type=int, default=32)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--rep", type=float, default=1.1)
    ap.add_argument("--loops", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B-Base")
    ap.add_argument("--engine", choices=("python", "native"), default="python",
                    help="python is the validated reference engine; native is experimental")
    ap.add_argument("--cuda-graph", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--greedy", action="store_true")
    args = ap.parse_args()

    if args.model:
        checkpoint = args.model
    else:
        roots = list(args.checkpoint_dir or [])
        if not roots:
            if os.environ.get("LLMY_CHECKPOINT_DIR"):
                roots.append(Path(os.environ["LLMY_CHECKPOINT_DIR"]))
            roots.append(Path.cwd())
        checkpoint = find_latest(roots)
        print(f"[checkpoint] newest: {checkpoint}", flush=True)
    selected_checkpoint = checkpoint.resolve()
    checkpoint = prepare_checkpoint(checkpoint) if args.engine == "native" else selected_checkpoint
    exe = find_exe() if args.engine == "native" else None
    tokenizer = load_tokenizer(args.tokenizer) if args.engine == "native" else None
    distillation_prompt = (
        is_distillation_checkpoint(selected_checkpoint)
        if args.engine == "native" else False
    )
    if distillation_prompt:
        print("[tokenizer] distillation checkpoint: using the standalone chat template", flush=True)

    def run(prompt: str) -> int:
        if args.engine == "python":
            root = Path(__file__).resolve().parent.parent
            source = selected_checkpoint
            if source.suffix.lower() == ".llmy":
                sibling = source.with_suffix(".pt")
                if sibling.exists():
                    source = sibling
            cmd = [sys.executable, "-u", str(root / "sample_standalone.py"),
                   "--checkpoint", str(source), "--prompt", prompt,
                   "--max-tokens", str(args.tokens),
                   "--device", args.device, "--temp", str(args.temp),
                   "--top-k", str(args.top_k), "--top-p", str(args.top_p),
                   "--rep", str(args.rep), "--seed", str(args.seed)]
            if args.loops > 0:
                cmd.extend(["--fixed-loops", str(args.loops)])
            if args.cuda_graph:
                cmd.append("--cuda-graph")
            if args.greedy:
                cmd.append("--greedy")
            return subprocess.run(cmd).returncode

        if exe is None:
            raise RuntimeError("The native inference executable was not found.")
        if distillation_prompt:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
            if hasattr(rendered, "__contains__") and "input_ids" in rendered:
                rendered = rendered["input_ids"]
            if hasattr(rendered, "tolist"):
                rendered = rendered.tolist()
            if rendered and isinstance(rendered[0], list):
                rendered = rendered[0]
            ids = [int(token) for token in rendered]
        else:
            ids = tokenizer.encode(prompt, add_special_tokens=False)
        if not ids:
            print("[error] prompt produced no tokens", file=sys.stderr)
            return 1
        id_csv = ",".join(str(int(x)) for x in ids)
        cmd = [str(exe), "-m", str(checkpoint), "--token-ids", id_csv,
               "-n", str(args.tokens), "--temp", str(args.temp),
               "--top-k", str(args.top_k), "--top-p", str(args.top_p),
               "--rep", str(args.rep), "--loops", str(args.loops),
               "--seed", str(args.seed)]
        if args.greedy:
            cmd.append("--greedy")
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        if result.returncode == 0:
            match = re.search(r"GENERATED_IDS:\s*(.*)", result.stdout)
            if match:
                generated = [int(x) for x in match.group(1).split()]
                print("TEXT:")
                print(tokenizer.decode(generated, skip_special_tokens=True))
        return result.returncode

    if args.prompt is not None:
        return run(args.prompt)
    while True:
        prompt = multiline_prompt()
        if prompt is None:
            return 0
        if run(prompt):
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
