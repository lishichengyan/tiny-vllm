"""KV cache storage: contiguous (Milestone 2) and block-based / paged (Milestone 3).

Two different ways of storing the K and V vectors of past tokens:

    Milestone 2 -- ContiguousKVCache (one request, one big tensor per layer)

        k_cache[layer]:  pos 0   pos 1   pos 2   pos 3   pos 4 ...
                        [ K0  ][ K1  ][ K2  ][ K3  ][ K4  ] ...

    Milestone 3 -- PagedKVCache (global pool shared by all requests, split into blocks)

        k_cache[layer]:  block 0        block 1        block 2        block 3
                        [ ][ ][ ][ ]   [ ][ ][ ][ ]   [ ][ ][ ][ ]   [ ][ ][ ][ ] ...
                                        \\_______ block_size slots ______/

    A request owns a *block table*: logical block i -> physical block id.
    A *slot mapping* says where each new token's K/V is written:

        slot = physical_block * block_size + offset

Both caches store K and V separately, with identical shapes.
"""

from dataclasses import dataclass

import torch

from .config import TinyLlamaConfig

# ---------------------------------------------------------------------------
# Milestone 2: contiguous KV cache
# ---------------------------------------------------------------------------


class ContiguousKVCache:
    """Per-request KV cache stored contiguously by token position (checkpoint 2.2).

    Attributes (both must exist after construction):
        k_cache: [num_layers, max_seq_len, num_kv_heads, head_dim]
        v_cache: [num_layers, max_seq_len, num_kv_heads, head_dim]

    Token position ``p`` of layer ``l`` lives at ``k_cache[l, p]``.
    """

    def __init__(
        self,
        config: TinyLlamaConfig,
        max_seq_len: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ):
        self.num_layers = config.num_layers
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.device = torch.device(device)
        # TODO(student): Milestone 2 -- allocate self.k_cache and self.v_cache.
        raise NotImplementedError("Milestone 2 (2.2): allocate ContiguousKVCache tensors")

    def write(self, layer_idx: int, positions: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        """Store the K/V of new tokens at their absolute positions.

        Args:
            layer_idx: which decoder layer these K/V belong to.
            positions: [num_tokens] integer tensor of absolute positions.
            k, v:      [num_tokens, num_kv_heads, head_dim]
        Writing to a position >= max_seq_len is an error (let indexing raise).
        """
        # TODO(student): Milestone 2
        raise NotImplementedError("Milestone 2 (2.2): implement ContiguousKVCache.write()")

    def read(self, layer_idx: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the K and V of positions ``0 .. seq_len-1`` for one layer.

        Returns:
            (k, v), each [seq_len, num_kv_heads, head_dim]
        """
        # TODO(student): Milestone 2
        raise NotImplementedError("Milestone 2 (2.2): implement ContiguousKVCache.read()")


# ---------------------------------------------------------------------------
# Milestone 3: block-based (paged) KV cache
# ---------------------------------------------------------------------------


class PagedKVCache:
    """Global block-based KV pool shared by every request (checkpoints 3.1, 3.6).

    Attributes (both must exist after construction):
        k_cache: [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        v_cache: [num_layers, num_blocks, block_size, num_kv_heads, head_dim]

    Layout rationale: ``num_layers`` first so that ``k_cache[layer]`` is the whole
    pool of one layer; ``num_blocks`` and ``block_size`` next so that
    ``k_cache[layer, block]`` is one physical block and ``k_cache[layer, block, offset]``
    is one token slot. Every layer has the same number of blocks, and a request's
    block table is shared by all layers.

    This class only owns the memory. Which blocks are free is tracked by
    ``BlockPool``; which blocks belong to a request is the request's block table.
    """

    def __init__(
        self,
        config: TinyLlamaConfig,
        num_blocks: int,
        block_size: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ):
        if num_blocks < 1 or block_size < 1:
            raise ValueError("num_blocks and block_size must be >= 1")
        self.num_layers = config.num_layers
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.dtype = dtype
        self.device = torch.device(device)
        # TODO(student): Milestone 3 -- allocate self.k_cache and self.v_cache.
        raise NotImplementedError("Milestone 3 (3.1): allocate PagedKVCache tensors")

    @property
    def num_slots(self) -> int:
        """Total number of token slots per layer."""
        return self.num_blocks * self.block_size

    def write(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, slot_mapping: torch.Tensor) -> None:
        """Write the K/V of new tokens into their physical slots (checkpoint 3.6).

        Args:
            layer_idx:    which decoder layer these K/V belong to.
            k, v:         [num_tokens, num_kv_heads, head_dim]
            slot_mapping: [num_tokens] integer tensor. Entry ``t`` is the flat slot
                          ``physical_block * block_size + offset`` for token ``t``
                          (see ``compute_slot_mapping``).
        Only the addressed slots may change; every other slot of every layer must be untouched.
        """
        # TODO(student): Milestone 3
        raise NotImplementedError("Milestone 3 (3.6): implement PagedKVCache.write()")


def blocks_needed(num_tokens: int, block_size: int) -> int:
    """Number of blocks required to hold ``num_tokens`` tokens (checkpoint 3.4).

    Examples with block_size = 4:  0 -> 0,  1 -> 1,  4 -> 1,  5 -> 2.
    """
    # TODO(student): Milestone 3
    raise NotImplementedError("Milestone 3 (3.4): implement blocks_needed()")


def logical_to_physical(block_table: list[int], token_position: int, block_size: int) -> tuple[int, int]:
    """Translate a logical token position into (physical_block, offset) (checkpoint 3.4).

    Example:
        block_size = 4, block_table = [7, 2], token_position = 5
        -> logical block 1, offset 1 -> physical block 2 -> returns (2, 1)

    Raises:
        IndexError if the position is not covered by the block table.
    """
    # TODO(student): Milestone 3
    raise NotImplementedError("Milestone 3 (3.4): implement logical_to_physical()")


def ensure_block_capacity(
    block_table: list[int],
    block_pool,
    num_tokens: int,
    block_size: int,
) -> None:
    """Grow ``block_table`` in place until it can hold ``num_tokens`` tokens (checkpoint 3.4).

    New physical blocks are taken from ``block_pool`` (a ``BlockPool``) and appended
    to the end of the table. Never shrinks the table; does nothing if the table
    already has enough blocks. Propagates ``OutOfBlocksError`` if the pool is empty.
    """
    # TODO(student): Milestone 3
    raise NotImplementedError("Milestone 3 (3.4): implement ensure_block_capacity()")


def compute_slot_mapping(block_table: list[int], positions: torch.Tensor, block_size: int) -> torch.Tensor:
    """Map logical token positions to flat physical slots (checkpoint 3.5).

    Args:
        block_table: the request's logical -> physical block mapping.
        positions:   [num_tokens] integer tensor of absolute token positions.
        block_size:  tokens per block.

    Returns:
        [num_tokens] int64 tensor; entry ``t`` is ``physical_block * block_size + offset``
        for ``positions[t]``.

    Example:
        block_size = 4, block_table = [7, 2], positions = [5]  ->  [9]
    """
    # TODO(student): Milestone 3
    raise NotImplementedError("Milestone 3 (3.5): implement compute_slot_mapping()")


# ---------------------------------------------------------------------------
# Metadata handed to the attention layers in paged mode (Milestones 4-6)
# ---------------------------------------------------------------------------


@dataclass
class AttentionMetadata:
    """Everything attention needs to read/write the paged KV cache for one forward pass.

    A forward pass processes ``batch`` sequences with ``seq_len`` new tokens each
    (``seq_len`` is the prompt length during prefill and 1 during decode).

    Attributes:
        slot_mapping: [batch * seq_len] int64 tensor. Flat physical slot for every new
                      token, in the same order as ``input_ids.reshape(-1)``.
        block_tables: one block table (list of physical block ids) per sequence.
        seq_lens:     one int per sequence: the total number of tokens in that sequence
                      *including* the new tokens of this forward pass. Attention for
                      sequence ``b`` covers logical positions ``0 .. seq_lens[b]-1``.
    """

    slot_mapping: torch.Tensor
    block_tables: list[list[int]]
    seq_lens: list[int]

    def __post_init__(self) -> None:
        if len(self.block_tables) != len(self.seq_lens):
            raise ValueError("block_tables and seq_lens must have one entry per sequence")
        if self.slot_mapping.dim() != 1:
            raise ValueError("slot_mapping must be a flat 1-D tensor")
