"""Single-request generation loops.

Milestone 2 (naive and contiguous-KV-cache generation) and Milestone 4.5
(generation over the paged KV cache).

All functions take a prompt as a list of token ids and return ONLY the newly
generated token ids (the EOS token is included if it was produced). Greedy
decoding is used everywhere, so different strategies must produce identical
tokens for the same prompt.

Call the model as ``model(input_ids, positions, kv_cache=..., attn_metadata=...)``
so that the test-suite can observe every forward pass.

    input_ids: [batch, seq_len]   positions: [batch, seq_len]   logits: [batch, seq_len, vocab]
"""

import torch

from .block_pool import BlockPool
from .kv_cache import ContiguousKVCache, PagedKVCache
from .model import TinyLlama

# ---------------------------------------------------------------------------
# Milestone 2
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_naive(
    model: TinyLlama,
    prompt_tokens: list[int],
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> list[int]:
    """Deliberately inefficient autoregressive generation (checkpoint 2.1).

    Every step re-runs the model on the WHOLE sequence (prompt + everything generated
    so far) with ``kv_cache=None`` and takes the argmax of the last position's logits.

        step 0:  A B C      -> D
        step 1:  A B C D    -> E
        step 2:  A B C D E  -> F

    Stop after ``max_new_tokens`` tokens, or as soon as ``eos_token_id`` is produced
    (the EOS token is included in the returned list).
    """
    # TODO(student): Milestone 2
    raise NotImplementedError("Milestone 2 (2.1): implement generate_naive()")


@torch.no_grad()
def prefill(model: TinyLlama, kv_cache: ContiguousKVCache, prompt_tokens: list[int]) -> torch.Tensor:
    """Run the whole prompt once, filling the KV cache for positions 0..len-1 (checkpoint 2.3).

    Returns:
        logits of the LAST prompt position: [vocab_size]
    """
    # TODO(student): Milestone 2
    raise NotImplementedError("Milestone 2 (2.3): implement prefill()")


@torch.no_grad()
def decode_step(model: TinyLlama, kv_cache: ContiguousKVCache, token: int, position: int) -> torch.Tensor:
    """Run ONE new token at ``position`` against the cached history (checkpoint 2.4).

    The model must only see ``input_ids`` of shape [1, 1].

    Returns:
        logits for the next token: [vocab_size]
    """
    # TODO(student): Milestone 2
    raise NotImplementedError("Milestone 2 (2.4): implement decode_step()")


@torch.no_grad()
def generate_with_kv_cache(
    model: TinyLlama,
    prompt_tokens: list[int],
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> list[int]:
    """Prefill once, then decode one token at a time (checkpoint 2.5).

    Allocate a ``ContiguousKVCache`` large enough for ``len(prompt_tokens) + max_new_tokens``
    tokens, call ``prefill`` once and ``decode_step`` for every following token.
    Same stopping rules and return value as ``generate_naive``.
    """
    # TODO(student): Milestone 2
    raise NotImplementedError("Milestone 2 (2.5): implement generate_with_kv_cache()")


# ---------------------------------------------------------------------------
# Milestone 4.5
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_paged(
    model: TinyLlama,
    kv_cache: PagedKVCache,
    block_pool: BlockPool,
    prompt_tokens: list[int],
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> list[int]:
    """Single-request generation over the block-based KV cache (checkpoint 4.5).

    1. Allocate blocks for the prompt (``ensure_block_capacity``), build the slot mapping
       and an ``AttentionMetadata``, run the prompt through the model.
    2. For every new token: make sure its position has a block, build the metadata for
       that one token, run the model with ``input_ids`` of shape [1, 1].
    3. When generation stops, return ALL of the request's blocks to ``block_pool``
       (also if an exception is raised).

    Same stopping rules and return value as ``generate_naive``.
    """
    # TODO(student): Milestone 4
    raise NotImplementedError("Milestone 4 (4.5): implement generate_paged()")
