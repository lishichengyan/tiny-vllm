"""Self-attention: plain causal attention (M1), attention over a contiguous KV cache (M2),
and paged attention over block-scattered KV (M3/M4/M5).

The same ``Attention`` module is used in every milestone. Its ``forward`` dispatches on
the kind of KV cache it receives:

    kv_cache is None               -> Milestone 1: attend over the tokens in this forward pass
    kv_cache is ContiguousKVCache  -> Milestone 2: write new K/V, attend over cache[0:seq_len]
    kv_cache is PagedKVCache       -> Milestone 3/4/5: write via slot mapping, attend via block table

Shape conventions:

    hidden_states: [batch, seq_len, hidden_size]
    q:             [batch, seq_len, num_heads,    head_dim]
    k, v:          [batch, seq_len, num_kv_heads, head_dim]
    attention out: [batch, seq_len, num_heads,    head_dim]  -> reshaped to [batch, seq_len, hidden_size]
"""

import math

import torch
import torch.nn as nn

from .config import TinyLlamaConfig
from .kv_cache import AttentionMetadata, ContiguousKVCache, PagedKVCache
from .layers import apply_rope  # noqa: F401  (used by your project_qkv)


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Scaled dot-product attention with a causal mask (checkpoint 1.3).

    Args:
        q: [batch, num_queries, num_heads,    head_dim]
        k: [batch, num_keys,    num_kv_heads, head_dim]
        v: [batch, num_keys,    num_kv_heads, head_dim]
        with ``num_keys >= num_queries``.

    The queries are the LAST ``num_queries`` positions of the key sequence:

        keys:     0 1 2 3 4 5 6 7            (num_keys = 8)
        queries:            5 6 7            (num_queries = 3)

    so query ``i`` (0-based) sits at absolute position ``num_keys - num_queries + i`` and may
    attend to keys ``0 .. num_keys - num_queries + i`` inclusive. With ``num_keys == num_queries``
    this is ordinary causal self-attention; with ``num_queries == 1`` it is a decode step.

    Grouped-query attention: query head ``h`` uses K/V head ``h // (num_heads // num_kv_heads)``.

    Scores are scaled by ``1 / sqrt(head_dim)``; compute the softmax in float32.

    Returns:
        [batch, num_queries, num_heads, head_dim]

    Raises:
        ValueError: if ``num_keys < num_queries``.
    """
    # TODO(student): Milestone 1
    raise NotImplementedError("Milestone 1 (1.3): implement causal_attention()")


class Attention(nn.Module):
    def __init__(self, config: TinyLlamaConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    # ------------------------------------------------------------------ M1
    def project_qkv(
        self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project hidden states to Q, K, V and apply RoPE to Q and K (checkpoint 1.3).

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            cos, sin:      [batch, seq_len, head_dim]
        Returns:
            q: [batch, seq_len, num_heads,    head_dim]   (rotated)
            k: [batch, seq_len, num_kv_heads, head_dim]   (rotated)
            v: [batch, seq_len, num_kv_heads, head_dim]
        """
        # TODO(student): Milestone 1
        raise NotImplementedError("Milestone 1 (1.3): implement Attention.project_qkv()")

    # ------------------------------------------------------------------ M2
    def attend_contiguous(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: ContiguousKVCache,
    ) -> torch.Tensor:
        """Attention with a per-request contiguous KV cache (checkpoints 2.3, 2.4).

        1. Write the new K/V (``k``, ``v`` at ``positions``) into ``kv_cache`` for this layer.
        2. Read the K/V of all positions ``0 .. positions.max()`` back.
        3. Run ``causal_attention`` with the new tokens as queries and the whole
           history as keys/values.

        Only batch size 1 is supported in this mode.
        Returns: [1, seq_len, num_heads, head_dim]
        """
        # TODO(student): Milestone 2
        raise NotImplementedError("Milestone 2 (2.3/2.4): implement Attention.attend_contiguous()")

    # ------------------------------------------------------------ M3/M4/M5
    def attend_paged(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: PagedKVCache,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """Attention with the global paged KV cache (checkpoints 3.6, 4.5, 4.6, 5.5).

        1. Write the new K/V into their physical slots (``attn_metadata.slot_mapping``).
        2. For every sequence ``b`` in the batch run ``paged_attention`` with that
           sequence's own block table and sequence length.
        3. (4.6) When the pass is a decode step -- ``seq_len == 1``, one new token per
           sequence -- use ``paged_attention_decode`` instead: it reads the blocks in place
           without materialising the history. Prefill keeps using ``paged_attention``.

        Args:
            q:    [batch, seq_len, num_heads,    head_dim]
            k, v: [batch, seq_len, num_kv_heads, head_dim]
        Returns:
            [batch, seq_len, num_heads, head_dim]
        """
        # TODO(student): Milestone 3 (write) / Milestone 4 (single sequence) / Milestone 5 (batch)
        raise NotImplementedError("Milestone 3/4/5: implement Attention.attend_paged()")

    # ------------------------------------------------------------ dispatch
    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: ContiguousKVCache | PagedKVCache | None = None,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        """hidden_states: [batch, seq_len, hidden_size] -> [batch, seq_len, hidden_size]"""
        batch, seq_len, _ = hidden_states.shape
        q, k, v = self.project_qkv(hidden_states, cos, sin)

        if kv_cache is None:
            out = causal_attention(q, k, v)
        elif isinstance(kv_cache, ContiguousKVCache):
            out = self.attend_contiguous(q, k, v, positions, kv_cache)
        elif isinstance(kv_cache, PagedKVCache):
            if attn_metadata is None:
                raise ValueError("attn_metadata is required when using a PagedKVCache")
            out = self.attend_paged(q, k, v, kv_cache, attn_metadata)
        else:
            raise TypeError(f"unsupported kv_cache type: {type(kv_cache).__name__}")

        # out: [batch, seq_len, num_heads, head_dim] -> [batch, seq_len, hidden_size]
        return self.o_proj(out.reshape(batch, seq_len, self.num_heads * self.head_dim))


# ---------------------------------------------------------------------------
# Milestone 4: paged attention
# ---------------------------------------------------------------------------


def gather_kv(cache: torch.Tensor, block_table: list[int], seq_len: int) -> torch.Tensor:
    """Rebuild the logically ordered K (or V) of one sequence from scattered physical blocks
    (checkpoints 4.1, 4.2, 4.4).

    Args:
        cache:       [num_blocks, block_size, num_kv_heads, head_dim] -- one layer of the pool
                     (``kv_cache.k_cache[layer_idx]`` or ``kv_cache.v_cache[layer_idx]``).
        block_table: physical block id of each logical block of the sequence. May contain
                     more blocks than ``seq_len`` needs; extra blocks must be ignored.
        seq_len:     number of valid tokens. The last logical block may be only partially
                     filled; slots at or beyond ``seq_len`` must NOT appear in the result.

    Returns:
        [seq_len, num_kv_heads, head_dim] in logical token order.
    """
    # TODO(student): Milestone 4
    raise NotImplementedError("Milestone 4 (4.1/4.2/4.4): implement gather_kv()")


def paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: list[int],
    seq_len: int,
) -> torch.Tensor:
    """Causal attention for one sequence whose K/V live in scattered physical blocks
    (checkpoint 4.3).

    Args:
        q:           [num_queries, num_heads, head_dim] -- the queries of the LAST
                     ``num_queries`` tokens of the sequence (all tokens for prefill,
                     one token for decode). Their K/V must already be in the cache.
        k_cache:     [num_blocks, block_size, num_kv_heads, head_dim] for this layer.
        v_cache:     same shape as ``k_cache``.
        block_table: the sequence's logical -> physical block mapping.
        seq_len:     total tokens in the sequence (including the query tokens).

    Returns:
        [num_queries, num_heads, head_dim]

    This is a *semantic* implementation: gather K/V into logical order, then run
    ordinary causal attention. Real vLLM kernels read the blocks directly without
    materialising a contiguous copy; the maths is identical.
    """
    # TODO(student): Milestone 4
    raise NotImplementedError("Milestone 4 (4.3): implement paged_attention()")


def paged_attention_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: list[int],
    seq_len: int,
) -> torch.Tensor:
    """Decode attention that walks the blocks in place -- no gather (checkpoint 4.6).

    This is the algorithm inside real paged-attention kernels (vLLM's original
    ``paged_attention_v1`` was decode-only, exactly like this function).

    Args:
        q:           [num_heads, head_dim] -- the single query of the newest token, whose K/V
                     are already in the cache at logical position ``seq_len - 1``.
        k_cache:     [num_blocks, block_size, num_kv_heads, head_dim] for this layer.
        v_cache:     same shape as ``k_cache``.
        block_table: the sequence's logical -> physical block mapping.
        seq_len:     total tokens in the sequence (including the query token).

    Returns:
        [num_heads, head_dim] -- bit-for-bit the same maths as
        ``paged_attention(q[None], ...)[0]`` up to floating-point rounding.

    Requirements:
        * Visit the logical blocks one at a time; never build a ``[seq_len, ...]`` K or V
          tensor and never call ``gather_kv`` / ``paged_attention``.
        * Keep three running quantities per head: the max score ``m``, the softmax
          denominator ``l`` and the weighted-V accumulator ``acc``. When a block raises the
          max, rescale ``l`` and ``acc`` before adding the block's contribution
          ("online softmax"; the recurrence is in docs/04-paged-attention.md).
        * The last block may be partial: only its first ``seq_len % block_size`` slots (or all,
          if that is 0) are valid.
        * Scale scores by ``1 / sqrt(head_dim)``; keep ``m``, ``l``, ``acc`` in float32.
        * GQA: query head ``h`` reads K/V head ``h // (num_heads // num_kv_heads)``.
    """
    # TODO(student): Milestone 4
    raise NotImplementedError("Milestone 4 (4.6): implement paged_attention_decode()")
