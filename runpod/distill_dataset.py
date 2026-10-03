"""Streaming loader for assistant-only distillation examples.

The dataset's structured ``sft`` view contains JSON-encoded messages and tool
definitions.  This module deliberately renders the complete conversation but
labels only assistant spans.  User, system, and tool content is therefore
available as context without becoming a target the student is trained to
imitate as its own role.
"""
from __future__ import annotations

import difflib
import json
import random
import time
from collections.abc import Mapping
from typing import Any, Iterator

import torch


def _tools(value: Any) -> Any:
    if value in (None, "", []):
        return None
    tools = json.loads(value) if isinstance(value, str) else json.loads(json.dumps(value))
    for tool in tools:
        function = tool.get("function") or {}
        encoded = function.pop("parameters_json", None)
        if encoded is not None:
            function["parameters"] = json.loads(encoded)
    return tools


def _messages(value: Any, tokenizer: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        value = json.loads(value)
    messages = json.loads(json.dumps(value))
    idmap: dict[str, str] = {}
    mistral = "mistral" in str(getattr(tokenizer, "name_or_path", "")).lower()
    if mistral and messages and messages[0].get("role") == "system":
        system = str(messages.pop(0).get("content", "")).strip()
        if not messages or messages[0].get("role") != "user":
            raise ValueError("Mistral adapter requires a user turn after system")
        messages[0]["content"] = (
            f"[SYSTEM]\n{system}\n[/SYSTEM]\n\n"
            f"{str(messages[0].get('content', ''))}"
        ).strip()
    for message in messages:
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    function["arguments"] = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            if mistral:
                old = str(call.get("id", ""))
                new = idmap.setdefault(old, f"call{len(idmap):05d}")
                call["id"] = new
        if mistral and message.get("tool_call_id") in idmap:
            message["tool_call_id"] = idmap[message["tool_call_id"]]
    return messages


def _ids(rendered: Any) -> list[int]:
    if isinstance(rendered, Mapping):
        rendered = rendered["input_ids"]
    if rendered and isinstance(rendered[0], list):
        if len(rendered) != 1:
            raise ValueError("unexpected batched chat-template output")
        rendered = rendered[0]
    return list(rendered)


def _render(tokenizer: Any, messages: list[dict[str, Any]], tools: Any) -> list[int]:
    kwargs: dict[str, Any] = {"tokenize": True, "add_generation_prompt": False}
    if tools:
        kwargs["tools"] = tools
    return _ids(tokenizer.apply_chat_template(messages, **kwargs))


def _assistant_stop_ids(tokenizer: Any) -> set[int]:
    """Return chat-template termination IDs that can end an assistant turn.

    The Qwen template used for distillation emits ``<|im_end|>``.  Other
    ChatML-compatible tokenizers may use a different named terminator or only
    expose ``eos_token_id``.  Avoid treating an unknown-token ID as a stop ID
    when a named token is absent.
    """
    stop_ids: set[int] = set()
    vocab = {}
    try:
        vocab = tokenizer.get_vocab()
    except Exception:
        pass
    for token in ("<|im_end|>", "<|eot_id|>", "<|end|>", "</s>"):
        token_id = vocab.get(token)
        if token_id is None:
            continue
        stop_ids.add(int(token_id))
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, int):
        stop_ids.add(int(eos_token_id))
    elif isinstance(eos_token_id, (list, tuple)):
        stop_ids.update(int(token_id) for token_id in eos_token_id)
    return stop_ids


def _prefix_labels(tokenizer: Any, messages: list[dict[str, Any]], tools: Any):
    previous: list[int] = []
    labels: list[int] = []
    for index, message in enumerate(messages):
        # Qwen's template rejects a system-only prefix (it requires a user
        # query), even though the complete system+user+assistant conversation
        # is valid.  There are no trainable tokens in that prefix, so defer the
        # first render until a user-containing prefix exists.
        if index == 0 and message.get("role") == "system":
            continue
        try:
            current = _render(tokenizer, messages[: index + 1], tools)
        except Exception as exc:
            raise ValueError(f"chat template cannot render prefix at message {index}: {exc}") from exc
        if current[: len(previous)] != previous:
            raise ValueError(f"chat template is not prefix preserving at message {index}")
        suffix = current[len(previous) :]
        # ``trainable`` is metadata, but role is the hard safety boundary:
        # never label user/system/tool messages even if malformed metadata says
        # otherwise.  Assistant turns are expected to be trainable in this view.
        is_assistant = message.get("role") == "assistant" and message.get("trainable", True) is not False
        labels.extend(suffix if is_assistant else [-100] * len(suffix))
        previous = current
    return previous, labels, "prefix-differential"


def _redaction_labels(tokenizer: Any, messages: list[dict[str, Any]], tools: Any):
    full = _render(tokenizer, messages, tools)
    mask = [False] * len(full)
    stop_ids = _assistant_stop_ids(tokenizer)
    assistant_indices = [
        index for index, message in enumerate(messages)
        if message.get("role") == "assistant" and message.get("trainable", True) is not False
    ]
    for index in assistant_indices:
        redacted = json.loads(json.dumps(messages))
        redacted[index]["content"] = f"__REDACTED_ASSISTANT_{index}__"
        redacted[index].pop("reasoning_content", None)
        redacted[index].pop("tool_calls", None)
        changed_positions: list[int] = []
        other = _render(tokenizer, redacted, tools)
        for tag, a0, a1, _, _ in difflib.SequenceMatcher(
            a=full, b=other, autojunk=False
        ).get_opcodes():
            if tag in {"replace", "delete"}:
                for position in range(a0, a1):
                    mask[position] = True
                    changed_positions.append(position)

        # A response-only mask must include the assistant turn terminator, or
        # generation has no supervised signal for when to stop.  Prefix
        # differential labeling gets this token naturally, but the fallback
        # path above only sees the content that changed under redaction.  Find
        # the first known terminator after the changed assistant payload.
        if changed_positions and stop_ids:
            after_payload = max(changed_positions)
            stop_position = next(
                (
                    position
                    for position in range(after_payload + 1, len(full))
                    if full[position] in stop_ids
                ),
                None,
            )
            if stop_position is None:
                stop_position = next(
                    (
                        position
                        for position in range(min(changed_positions), len(full))
                        if full[position] in stop_ids
                    ),
                    None,
                )
            if stop_position is not None:
                mask[stop_position] = True
    labels = [token if marked else -100 for token, marked in zip(full, mask)]
    method = "structural-redaction+assistant-stop" if stop_ids else "structural-redaction"
    return full, labels, method


def semantic_labels(tokenizer: Any, messages: Any, tools: Any = None,
                    max_length: int | None = None) -> dict[str, Any]:
    messages = _messages(messages, tokenizer)
    tools = _tools(tools)
    try:
        input_ids, labels, method = _prefix_labels(tokenizer, messages, tools)
        payload = "\n".join(
            str(message.get("reasoning_content", ""))
            + str(message.get("content", ""))
            + json.dumps(message.get("tool_calls") or [], sort_keys=True)
            for message in messages
            if message.get("role") == "assistant"
            and message.get("trainable", True) is not False
        )
        payload_ids = _ids(tokenizer(payload, add_special_tokens=False)) if payload else []
        if sum(label != -100 for label in labels) < max(1, int(0.15 * len(payload_ids))):
            raise ValueError("prefix differential did not expose assistant payload")
    except ValueError:
        input_ids, labels, method = _redaction_labels(tokenizer, messages, tools)
    if max_length is not None and len(input_ids) > max_length:
        raise ValueError(
            f"row requires {len(input_ids)} tokens; max_length={max_length}; truncation prohibited"
        )
    if len(input_ids) != len(labels):
        raise AssertionError("input/label length mismatch")
    if not any(label != -100 for label in labels):
        raise ValueError("row has no proven assistant-labeled tokens")
    return {
        "input_ids": input_ids,
        "labels": labels,
        "mask_method": method,
    }


class DistillationStreamingLoader:
    """Bounded streaming loader with deterministic example-level shuffling.

    Batch size one is intentional: examples have variable lengths and may be
    as long as ``max_length``.  This avoids padding a short example to a rare
    worst-case 64k row and keeps the memory bound honest.
    """

    variable_length = True

    def __init__(
        self,
        *,
        dataset_name: str,
        dataset_config: str | None,
        split: str = "train",
        tokenizer: Any,
        max_length: int = 65536,
        seed: int = 1337,
        state: dict[str, Any] | None = None,
        shuffle_buffer: int = 10000,
        fallback_datasets: list[dict[str, Any]] | None = None,
    ) -> None:
        if max_length < 2:
            raise ValueError("max_length must be at least 2")
        self.shard_dir = None
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        primary = {
            "source_name": "distillation_sft",
            "dataset_name": str(dataset_name),
            "dataset_config": dataset_config,
            "split": str(split),
            "seed": int(seed),
            "shuffle_buffer": max(1, int(shuffle_buffer)),
        }
        self._sources = [primary]
        for fallback in fallback_datasets or []:
            if not isinstance(fallback, dict):
                raise ValueError("fallback_datasets entries must be mappings")
            name = fallback.get("dataset_name")
            if not name:
                raise ValueError("fallback dataset is missing dataset_name")
            fallback_name = str(fallback.get("source_name", "")).strip()
            if not fallback_name:
                fallback_name = "distillation_" + str(name).replace("/", "_")
            self._sources.append({
                "source_name": fallback_name,
                "dataset_name": str(name),
                "dataset_config": fallback.get("dataset_config"),
                "split": str(fallback.get("split", split)),
                "seed": int(fallback.get("seed", seed)),
                "shuffle_buffer": max(
                    1, int(fallback.get("shuffle_buffer", shuffle_buffer))
                ),
            })

        saved = state or {}
        saved_states = saved.get("dataset_states")
        if not isinstance(saved_states, list):
            saved_states = []
        self._dataset_states: list[dict[str, int]] = []
        for index in range(len(self._sources)):
            source_state = (
                dict(saved_states[index])
                if index < len(saved_states) and isinstance(saved_states[index], dict)
                else {}
            )
            # Backward-compatible migration from the original single-source
            # checkpoint state format.
            if index == 0:
                source_state.setdefault("examples_seen", int(saved.get("examples_seen", 0)))
                source_state.setdefault("tokens_seen", int(saved.get("tokens_seen", 0)))
            self._dataset_states.append({
                "examples_seen": int(source_state.get("examples_seen", 0)),
                "tokens_seen": int(source_state.get("tokens_seen", 0)),
            })

        self.dataset_index = min(
            max(0, int(saved.get("dataset_index", 0))), len(self._sources)
        )
        saved_dataset_index = self.dataset_index
        # Keep an exhausted-chain index representable in checkpoints while
        # exposing the last source's metadata to the dashboard.
        self._activate_source(min(self.dataset_index, len(self._sources) - 1))
        self.dataset_index = saved_dataset_index
        self.rows_emitted = 0
        self.total_downloaded_shards = 0
        self.is_downloading = True
        self._stop = False
        self._dataset = None

    def _activate_source(self, index: int) -> None:
        source = self._sources[index]
        self.dataset_index = int(index)
        self.dataset_name = str(source["dataset_name"])
        self.dataset_config = source.get("dataset_config")
        self.split = str(source["split"])
        self.seed = int(source["seed"])
        self.shuffle_buffer = max(1, int(source["shuffle_buffer"]))
        current = self._dataset_states[index]
        self.examples_seen = int(current["examples_seen"])
        self.tokens_seen = int(current["tokens_seen"])
        self.source_name = str(source["source_name"])

    def _stream(self):
        from datasets import load_dataset

        kwargs = {
            "path": self.dataset_name,
            "split": self.split,
            "streaming": True,
        }
        # Some datasets have no named configuration.  Passing name="None"
        # or name="default" is not equivalent to omitting it.
        if self.dataset_config not in (None, "", "default"):
            kwargs["name"] = str(self.dataset_config)
        dataset = load_dataset(**kwargs)
        # The shuffle buffer is finite and deterministic; no source-order
        # pattern from the Parquet shards is exposed to the optimizer.
        dataset = dataset.shuffle(buffer_size=self.shuffle_buffer, seed=self.seed)
        return dataset

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        while self.dataset_index < len(self._sources):
            index = self.dataset_index
            self._activate_source(index)
            self.is_downloading = True
            dataset = self._stream()
            self._dataset = dataset
            # Freeze the resume offset for this pass.  Comparing against the
            # live examples_seen counter would make each yielded row increase
            # the skip target and silently discard fresh rows from the current
            # stream.
            resume_examples = int(self.examples_seen)
            skipped = 0
            try:
                for row in dataset:
                    if self._stop:
                        return
                    if skipped < resume_examples:
                        skipped += 1
                        continue
                    # Tokenize without rejecting an overlong conversation;
                    # keep the most recent context window so assistant-labeled
                    # completion tokens remain available while the trainer's
                    # memory bound stays enforceable.
                    encoded = semantic_labels(
                        self.tokenizer,
                        row.get("messages"),
                        row.get("tools"),
                        None,
                    )
                    if len(encoded["input_ids"]) > self.max_length:
                        trim_start = len(encoded["input_ids"]) - self.max_length
                        encoded["input_ids"] = encoded["input_ids"][trim_start:]
                        encoded["labels"] = encoded["labels"][trim_start:]
                        if not any(label != -100 for label in encoded["labels"]):
                            raise ValueError(
                                "overlong row has no assistant labels in retained context"
                            )
                    input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
                    labels = torch.tensor(encoded["labels"], dtype=torch.long)
                    if input_ids.numel() != labels.numel() or input_ids.numel() < 2:
                        raise ValueError("invalid distillation row after semantic labeling")
                    length = int(input_ids.numel())
                    segment_ids = torch.zeros(length, dtype=torch.int32)
                    reset_mask = torch.zeros(length, dtype=torch.bool)
                    reset_mask[0] = True
                    attention_mask = torch.ones(length, dtype=torch.long)
                    self.examples_seen += 1
                    self.rows_emitted += 1
                    self.tokens_seen += length
                    self._dataset_states[index]["examples_seen"] = self.examples_seen
                    self._dataset_states[index]["tokens_seen"] = self.tokens_seen
                    yield {
                        "input_ids": input_ids.unsqueeze(0),
                        "labels": labels.unsqueeze(0),
                        "segment_ids": segment_ids.unsqueeze(0),
                        "reset_mask": reset_mask.unsqueeze(0),
                        "attention_mask": attention_mask.unsqueeze(0),
                        "_has_reset": True,
                        "_source_name": self.source_name,
                        "_source_names": [self.source_name],
                        "_distill_mask_method": encoded["mask_method"],
                    }
            finally:
                self.is_downloading = False

            if self._stop:
                return
            print(
                f"[Data] Exhausted distillation source {self.source_name} "
                f"({self.dataset_name}); advancing without repetition.",
                flush=True,
            )
            self.dataset_index += 1

        raise RuntimeError(
            "all configured distillation databases are exhausted; refusing to repeat data"
        )

    def state_dict(self) -> dict[str, Any]:
        current_index = min(self.dataset_index, len(self._dataset_states) - 1)
        current = self._dataset_states[current_index]
        return {
            "loader": "distillation",
            "dataset": self.dataset_name,
            "dataset_config": self.dataset_config,
            "split": self.split,
            "seed": self.seed,
            "max_length": self.max_length,
            "dataset_index": self.dataset_index,
            "dataset_states": [dict(item) for item in self._dataset_states],
            "examples_seen": int(current["examples_seen"]),
            "tokens_seen": int(current["tokens_seen"]),
            "datasets": [dict(source) for source in self._sources],
        }

    def stop(self) -> None:
        self._stop = True
