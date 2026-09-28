"""Test helpers. Nothing in here implements learner exercises."""

import contextlib

import torch
import torch.nn as nn

from tiny_vllm.request import Request


def random_tokens(n: int, vocab_size: int, seed: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    # Avoid the special ids 0..3 so prompts never accidentally contain EOS/BOS.
    return torch.randint(4, vocab_size, (n,), generator=g).tolist()


class RecordingModel(nn.Module):
    """Wraps a TinyLlama and records the arguments of every forward call.

    ``calls[i]`` is a dict with keys ``input_ids``, ``positions``, ``kv_cache``, ``attn_metadata``.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.inner = model
        self.calls: list[dict] = []

    @property
    def config(self):
        return self.inner.config

    @property
    def device(self):
        return self.inner.device

    @property
    def dtype(self):
        return self.inner.dtype

    @property
    def layers(self):
        return self.inner.layers

    def forward(self, *args, **kwargs):
        names = ["input_ids", "positions", "kv_cache", "attn_metadata"]
        call = {name: None for name in names}
        for name, value in zip(names, args):
            call[name] = value
        call.update(kwargs)
        self.calls.append(
            {
                "input_ids": call["input_ids"].clone(),
                "positions": call["positions"].clone(),
                "kv_cache": call["kv_cache"],
                "attn_metadata": call["attn_metadata"],
            }
        )
        return self.inner(*args, **kwargs)

    def input_lengths(self) -> list[int]:
        return [c["input_ids"].shape[1] for c in self.calls]

    def batch_sizes(self) -> list[int]:
        return [c["input_ids"].shape[0] for c in self.calls]


class ScriptedModel(nn.Module):
    """A fake model whose greedy next token after position ``p`` is ``script[p]``.

    Ignores the token ids and any KV cache. It lets us test control flow
    (stopping rules, call shapes) independently of real model weights.
    """

    def __init__(self, config, script: list[int]):
        super().__init__()
        self.config = config
        self.script = list(script)
        self._anchor = nn.Parameter(torch.zeros(1), requires_grad=False)

    @property
    def device(self):
        return self._anchor.device

    @property
    def dtype(self):
        return self._anchor.dtype

    def forward(self, input_ids, positions, kv_cache=None, attn_metadata=None):
        assert input_ids.shape == positions.shape, "input_ids and positions must have the same shape"
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, self.config.vocab_size)
        for b in range(batch):
            for s in range(seq_len):
                p = int(positions[b, s])
                token = self.script[p] if p < len(self.script) else 0
                logits[b, s, token] = 10.0
        return logits


@contextlib.contextmanager
def capture_io(module: nn.Module):
    """Record (args, kwargs, output) of every forward call of ``module`` via hooks."""
    records: list[tuple[tuple, dict, object]] = []

    def hook(_module, args, kwargs, output):
        records.append((args, kwargs, output))

    handle = module.register_forward_hook(hook, with_kwargs=True)
    try:
        yield records
    finally:
        handle.remove()


def assert_block_tables_disjoint(requests: list[Request]) -> None:
    seen: dict[int, int] = {}
    for req in requests:
        for block in req.block_table:
            assert block not in seen, (
                f"physical block {block} owned by both request {seen[block]} and request {req.request_id}"
            )
            seen[block] = req.request_id
        assert len(set(req.block_table)) == len(req.block_table), (
            f"request {req.request_id} lists a block twice: {req.block_table}"
        )
