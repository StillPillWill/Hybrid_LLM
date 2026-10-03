"""Pack tokenized documents into causal windows with reset and segment masks."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .tokenize import TokenizedDocument


@dataclass
class PackedBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    segment_ids: torch.Tensor
    reset_mask: torch.Tensor
    document_ids: list[list[str]]
    provenance: dict | None = None


def pack_tokenized_documents(documents: list[TokenizedDocument], *, context_length: int = 4096,
                             pad_token_id: int = 0, drop_remainder: bool = True) -> PackedBatch:
    if context_length < 2:
        raise ValueError("context_length must be at least 2")
    rows: list[list[int]] = []
    segments: list[list[int]] = []
    row_docs: list[list[str]] = []
    current: list[int] = []
    current_segments: list[int] = []
    current_docs: list[str] = []
    segment = 0
    for document in documents:
        if not document.token_ids:
            continue
        cursor = 0
        while cursor < len(document.token_ids):
            available = context_length - len(current)
            take = min(available, len(document.token_ids) - cursor)
            current.extend(document.token_ids[cursor:cursor + take])
            current_segments.extend([segment] * take)
            if document.document_id not in current_docs:
                current_docs.append(document.document_id)
            cursor += take
            if len(current) == context_length:
                rows.append(current)
                segments.append(current_segments)
                row_docs.append(current_docs)
                current, current_segments, current_docs = [], [], []
                segment += 1
    if current and not drop_remainder:
        pad = context_length - len(current)
        rows.append(current + [pad_token_id] * pad)
        segments.append(current_segments + [-1] * pad)
        row_docs.append(current_docs)
    if not rows:
        raise ValueError("no complete packed windows were produced")
    inputs = torch.tensor(rows, dtype=torch.long)
    labels = inputs.clone()
    segment_ids = torch.tensor(segments, dtype=torch.long)
    labels[segment_ids < 0] = -100
    reset = torch.zeros_like(segment_ids, dtype=torch.bool)
    reset[:, 0] = True
    reset[:, 1:] = segment_ids[:, 1:] != segment_ids[:, :-1]
    reset &= segment_ids >= 0
    return PackedBatch(inputs, labels, segment_ids, reset, row_docs)
