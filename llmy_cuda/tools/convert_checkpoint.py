#!/usr/bin/env python3
"""Convert a PyTorch .pt checkpoint into the llmy flat binary format.

The output is a single file that the C++ engine mmap's read-only. Format:

    offset  field
    ------  -------------------------------------------------------------
    0       magic       "LLMY" (4 bytes)
    4       version     u32, currently 1
    8       cfg_len     u32
    12      config      cfg_len bytes of UTF-8 text
    ...     n_tensors   u32
    ...     index       n_tensors * 96 bytes
    4096    data        padded to 4096 alignment

Index entry (96 bytes):

    u32  name_len
    char name[64]        NUL-padded
    u32  dtype           0=f32, 1=f16, 2=bf16
    u32  ndim            <= 4
    u32  shape[4]        unused trailing dims = 0
    u64  offset          from data_start (== 4096)
    u64  nbytes

Config text format: one `key=value` per line, list values comma-separated.
See `flatten_config` below.

Usage:
    python tools/convert_checkpoint.py input.pt output.llmy
    python tools/convert_checkpoint.py input.pt output.llmy --keep-fp32
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import torch

MAGIC = b"LLMY"
VERSION = 1
# 4-byte name length + 64-byte name + 4-byte dtype + 4-byte ndim +
# 16-byte shape + 8-byte offset + 8-byte payload size.
INDEX_ENTRY_SIZE = 108
DATA_ALIGN = 4096

DTYPE_F32 = 0
DTYPE_F16 = 1
DTYPE_BF16 = 2


def dtype_code(t: torch.Tensor) -> int:
    if t.dtype == torch.float32:
        return DTYPE_F32
    if t.dtype == torch.float16:
        return DTYPE_F16
    if t.dtype == torch.bfloat16:
        return DTYPE_BF16
    raise ValueError(f"unsupported dtype {t.dtype}")


# ---------------------------------------------------------------------------
# Config flattening.
#
# The checkpoint embeds a nested dict under the "config" key. We flatten it
# to `a.b.c=value` pairs. List values become comma-separated strings.
# ---------------------------------------------------------------------------

# Every top-level key we care about. Anything else is dropped so the C++
# parser has a known, stable surface.
_ALLOWED = {
    "vocab_size", "d_model", "num_loops",
    "physical_blocks", "stem_blocks", "recurrent_blocks", "tail_blocks",
    "recurrent_layout", "max_position_embeddings",
    "mhc", "gdn", "moe", "mla", "router", "csa", "hca", "dense_ffn",
    "loop_attnres", "adaptive_loops",
}


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return v
    if isinstance(v, (list, tuple)):
        return ",".join(str(x) for x in v)
    return str(v)


def _flatten_into(prefix: str, value, out: dict[str, str]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            key = f"{prefix}.{k}" if prefix else k
            _flatten_into(key, v, out)
    else:
        out[prefix] = _fmt(value)


def flatten_config(cfg: dict) -> str:
    # Config may be either {"model": {...}} or a flat dict. Normalize.
    if "model" in cfg and isinstance(cfg["model"], dict):
        cfg = cfg["model"]
    flat: dict[str, str] = {}
    for k, v in cfg.items():
        if k not in _ALLOWED:
            continue
        if isinstance(v, (dict, list, tuple, str, int, float, bool)):
            _flatten_into(k, v, flat)
    # Deterministic ordering helps diffing.
    lines = [f"{k}={flat[k]}" for k in sorted(flat)]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--keep-fp32", action="store_true",
                    help="store all tensors as float32 (default: keep input dtype)")
    ap.add_argument("--strict", action="store_true",
                    help="fail if any tensor is not float32 after conversion")
    args = ap.parse_args()

    if not args.input.is_file():
        print(f"error: {args.input} is not a file", file=sys.stderr)
        return 1

    print(f"loading {args.input} ...")
    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        print("error: checkpoint is not a dict", file=sys.stderr)
        return 1

    cfg = ckpt.get("config")
    if cfg is None:
        print("error: checkpoint has no embedded config", file=sys.stderr)
        return 1
    if not isinstance(cfg, dict):
        print(f"error: config is {type(cfg).__name__}, expected dict", file=sys.stderr)
        return 1

    state = ckpt.get("model", ckpt)
    if not isinstance(state, dict):
        print("error: state dict is not a dict", file=sys.stderr)
        return 1

    # Strip torch.compile wrappers.
    tensors: list[tuple[str, torch.Tensor]] = []
    for name, t in state.items():
        if not isinstance(t, torch.Tensor):
            continue
        clean = name.replace("_orig_mod.", "")
        t = t.detach().contiguous()
        if args.keep_fp32:
            t = t.float()
        if args.strict and t.dtype != torch.float32:
            print(f"error: tensor {clean} is {t.dtype}", file=sys.stderr)
            return 1
        # Collapse expert ModuleLists into single stacked tensors so the C++
        # side gets [E, H, D] instead of E separate [H, D] tensors.
        tensors.append((clean, t))

    # Detect and stack expert / shared expert lists.
    tensors = _stack_module_lists(tensors)

    tensors.sort(key=lambda kv: kv[0])

    config_text = flatten_config(cfg)
    config_bytes = config_text.encode("utf-8")

    print(f"writing {args.output} ...")
    with args.output.open("wb") as f:
        # Header
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", len(config_bytes)))
        f.write(config_bytes)

        # Tensor count
        f.write(struct.pack("<I", len(tensors)))

        # Reserve the index
        index_start = f.tell()
        f.write(b"\0" * (INDEX_ENTRY_SIZE * len(tensors)))
        index_end = f.tell()

        # Pad to DATA_ALIGN
        pad = (DATA_ALIGN - (index_end % DATA_ALIGN)) % DATA_ALIGN
        f.write(b"\0" * pad)
        data_start = f.tell()
        assert data_start % DATA_ALIGN == 0, (
            f"data_start {data_start} is not {DATA_ALIGN}-byte aligned"
        )

        # Write payloads, remembering offsets
        offsets: list[int] = []
        sizes: list[int] = []
        for _, t in tensors:
            offsets.append(f.tell() - data_start)
            blob = _to_bytes(t)
            sizes.append(len(blob))
            f.write(blob)

        # Backfill index
        f.seek(index_start)
        for (name, t), off, nbytes in zip(tensors, offsets, sizes):
            name_b = name.encode("utf-8")
            if len(name_b) > 63:
                raise ValueError(f"tensor name too long: {name}")
            shape = list(t.shape)
            if len(shape) > 4:
                raise ValueError(f"tensor {name} has {len(shape)} dims (max 4)")
            shape_padded = shape + [0] * (4 - len(shape))
            entry = struct.pack(
                "<I 64s I I 4I Q Q",
                len(name_b),
                name_b.ljust(64, b"\0"),
                dtype_code(t),
                len(shape),
                *shape_padded,
                off,
                nbytes,
            )
            assert len(entry) == INDEX_ENTRY_SIZE, len(entry)
            f.write(entry)

    print(f"done: {args.output.stat().st_size / 1e9:.3f} GB, "
          f"{len(tensors)} tensors")
    return 0


def _to_bytes(t: torch.Tensor) -> bytes:
    """Fast path for fp32/fp16/bf16 contiguous tensors via numpy."""
    if t.dtype == torch.bfloat16:
        # numpy has no bf16; use torch's own byte view.
        return t.view(torch.uint8).numpy().tobytes()
    return t.numpy().tobytes()


# ---------------------------------------------------------------------------
# Stack expert ModuleList entries.
#
# PyTorch's state_dict for `experts: nn.ModuleList([...])` produces entries
# like `...experts.0.gate_proj.weight`, `...experts.1.gate_proj.weight`, etc.
# We collapse these into `...experts.gate_proj.weight` with shape [E, H, D]
# so the C++ MoE kernel gets a single contiguous pointer.
# ---------------------------------------------------------------------------

import re

_EXPERT_RE = re.compile(
    r"^(?P<prefix>.+\.(?:experts|shared))\.(?P<idx>\d+)\.(?P<rest>.+)$"
)


def _stack_module_lists(
    tensors: list[tuple[str, torch.Tensor]],
) -> list[tuple[str, torch.Tensor]]:
    groups: dict[tuple[str, str], list[tuple[int, torch.Tensor]]] = {}
    passthrough: list[tuple[str, torch.Tensor]] = []

    for name, t in tensors:
        m = _EXPERT_RE.match(name)
        if m is None:
            passthrough.append((name, t))
            continue
        prefix = m.group("prefix")
        idx = int(m.group("idx"))
        rest = m.group("rest")
        key = (prefix, rest)
        groups.setdefault(key, []).append((idx, t))

    for (prefix, rest), items in groups.items():
        items.sort(key=lambda kv: kv[0])
        # Verify all experts have the same shape
        first_shape = items[0][1].shape
        for _, t in items[1:]:
            if t.shape != first_shape:
                raise ValueError(
                    f"expert shapes mismatch under {prefix}.{rest}: "
                    f"{first_shape} vs {t.shape}"
                )
        stacked = torch.stack([t for _, t in items], dim=0).contiguous()
        passthrough.append((f"{prefix}.{rest}", stacked))

    return passthrough


if __name__ == "__main__":
    sys.exit(main())
