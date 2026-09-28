"""KV cache storage.

Milestone 2 -- checkpoint 2.2 (ContiguousKVCache).
Milestone 3 -- checkpoints 3.1 (global pool), 3.4 (block table helpers), 3.5 (slot mapping),
               3.6 (writing K/V into physical blocks).
"""

import pytest
import torch

from tiny_vllm.block_pool import BlockPool, OutOfBlocksError
from tiny_vllm.kv_cache import (
    AttentionMetadata,
    ContiguousKVCache,
    PagedKVCache,
    blocks_needed,
    compute_slot_mapping,
    ensure_block_capacity,
    logical_to_physical,
)

# ---------------------------------------------------------------------------
# 2.2 contiguous KV cache
# ---------------------------------------------------------------------------


@pytest.mark.m2
@pytest.mark.checkpoint("2.2")
class TestContiguousKVCache:
    def test_tensor_shapes_dtype_device(self, tiny_config):
        cache = ContiguousKVCache(tiny_config, max_seq_len=32)
        expected = (tiny_config.num_layers, 32, tiny_config.num_kv_heads, tiny_config.head_dim)
        assert tuple(cache.k_cache.shape) == expected
        assert tuple(cache.v_cache.shape) == expected
        assert cache.k_cache.dtype == torch.float32 and cache.v_cache.dtype == torch.float32
        assert cache.k_cache.device.type == "cpu"
        assert cache.k_cache.data_ptr() != cache.v_cache.data_ptr()

    def test_custom_dtype(self, tiny_config):
        cache = ContiguousKVCache(tiny_config, max_seq_len=8, dtype=torch.float16)
        assert cache.k_cache.dtype == torch.float16

    def test_write_then_read_roundtrip(self, tiny_config):
        cache = ContiguousKVCache(tiny_config, max_seq_len=8)
        k = torch.randn(3, tiny_config.num_kv_heads, tiny_config.head_dim)
        v = torch.randn(3, tiny_config.num_kv_heads, tiny_config.head_dim)
        cache.write(0, torch.tensor([0, 1, 2]), k, v)
        k_out, v_out = cache.read(0, 3)
        assert k_out.shape == (3, tiny_config.num_kv_heads, tiny_config.head_dim)
        torch.testing.assert_close(k_out, k)
        torch.testing.assert_close(v_out, v)
        # a shorter read is a prefix
        k_prefix, _ = cache.read(0, 2)
        torch.testing.assert_close(k_prefix, k[:2])

    def test_read_length(self, tiny_config):
        cache = ContiguousKVCache(tiny_config, max_seq_len=8)
        for n in (0, 1, 5, 8):
            k, v = cache.read(1, n)
            assert k.shape[0] == n and v.shape[0] == n

    def test_writes_land_at_the_given_positions(self, tiny_config):
        cache = ContiguousKVCache(tiny_config, max_seq_len=8)
        cache.k_cache.fill_(float("nan"))
        cache.v_cache.fill_(float("nan"))
        k = torch.randn(2, tiny_config.num_kv_heads, tiny_config.head_dim)
        v = torch.randn(2, tiny_config.num_kv_heads, tiny_config.head_dim)
        cache.write(1, torch.tensor([5, 2]), k, v)  # positions need not be sorted
        torch.testing.assert_close(cache.k_cache[1, 5], k[0])
        torch.testing.assert_close(cache.k_cache[1, 2], k[1])
        torch.testing.assert_close(cache.v_cache[1, 5], v[0])
        torch.testing.assert_close(cache.v_cache[1, 2], v[1])
        untouched = torch.tensor([0, 1, 3, 4, 6, 7])
        assert torch.isnan(cache.k_cache[1, untouched]).all()
        assert torch.isnan(cache.v_cache[1, untouched]).all()

    def test_layers_are_isolated(self, tiny_config):
        cache = ContiguousKVCache(tiny_config, max_seq_len=8)
        cache.k_cache.fill_(float("nan"))
        cache.v_cache.fill_(float("nan"))
        k = torch.randn(4, tiny_config.num_kv_heads, tiny_config.head_dim)
        cache.write(1, torch.arange(4), k, k)
        assert torch.isnan(cache.k_cache[0]).all() and torch.isnan(cache.v_cache[0]).all()
        assert torch.isfinite(cache.k_cache[1, :4]).all()

    def test_decode_style_append(self, tiny_config):
        cache = ContiguousKVCache(tiny_config, max_seq_len=8)
        shape = (tiny_config.num_kv_heads, tiny_config.head_dim)
        k_prompt = torch.randn(3, *shape)
        cache.write(0, torch.arange(3), k_prompt, k_prompt)
        k_new = torch.randn(1, *shape)
        cache.write(0, torch.tensor([3]), k_new, k_new)
        k_all, _ = cache.read(0, 4)
        torch.testing.assert_close(k_all[:3], k_prompt)
        torch.testing.assert_close(k_all[3], k_new[0])

    def test_overwrite_replaces(self, tiny_config):
        cache = ContiguousKVCache(tiny_config, max_seq_len=8)
        shape = (1, tiny_config.num_kv_heads, tiny_config.head_dim)
        cache.write(0, torch.tensor([2]), torch.ones(shape), torch.ones(shape))
        cache.write(0, torch.tensor([2]), torch.zeros(shape), torch.zeros(shape))
        k, _ = cache.read(0, 3)
        torch.testing.assert_close(k[2], torch.zeros(shape[1:]))

    def test_write_beyond_capacity_raises(self, tiny_config):
        cache = ContiguousKVCache(tiny_config, max_seq_len=4)
        k = torch.randn(1, tiny_config.num_kv_heads, tiny_config.head_dim)
        with pytest.raises((IndexError, RuntimeError)):
            cache.write(0, torch.tensor([4]), k, k)


# ---------------------------------------------------------------------------
# 3.1 global paged KV pool
# ---------------------------------------------------------------------------


@pytest.mark.m3
@pytest.mark.checkpoint("3.1")
class TestPagedKVCacheStorage:
    def test_tensor_shapes_dtype_device(self, tiny_config):
        cache = PagedKVCache(tiny_config, num_blocks=8, block_size=4)
        expected = (tiny_config.num_layers, 8, 4, tiny_config.num_kv_heads, tiny_config.head_dim)
        assert tuple(cache.k_cache.shape) == expected
        assert tuple(cache.v_cache.shape) == expected
        assert cache.k_cache.dtype == torch.float32
        assert cache.k_cache.device.type == "cpu"
        assert cache.k_cache.data_ptr() != cache.v_cache.data_ptr()
        assert cache.num_blocks == 8 and cache.block_size == 4 and cache.num_slots == 32

    def test_custom_dtype(self, tiny_config):
        cache = PagedKVCache(tiny_config, num_blocks=2, block_size=4, dtype=torch.float16)
        assert cache.k_cache.dtype == torch.float16 and cache.v_cache.dtype == torch.float16

    def test_invalid_sizes_raise(self, tiny_config):
        with pytest.raises(ValueError):
            PagedKVCache(tiny_config, num_blocks=0, block_size=4)
        with pytest.raises(ValueError):
            PagedKVCache(tiny_config, num_blocks=4, block_size=0)

    def test_one_pool_serves_all_layers(self, tiny_config):
        cache = PagedKVCache(tiny_config, num_blocks=5, block_size=2)
        for layer in range(tiny_config.num_layers):
            assert cache.k_cache[layer].shape[0] == 5


# ---------------------------------------------------------------------------
# 3.4 block table helpers
# ---------------------------------------------------------------------------


@pytest.mark.m3
@pytest.mark.checkpoint("3.4")
class TestBlockTableHelpers:
    @pytest.mark.parametrize(
        "num_tokens,block_size,expected",
        [
            (0, 4, 0),
            (1, 4, 1),
            (3, 4, 1),
            (4, 4, 1),
            (5, 4, 2),
            (8, 4, 2),
            (9, 4, 3),
            (7, 1, 7),
            (16, 16, 1),
            (17, 16, 2),
        ],
    )
    def test_blocks_needed(self, num_tokens, block_size, expected):
        assert blocks_needed(num_tokens, block_size) == expected

    def test_logical_to_physical_plan_example(self):
        assert logical_to_physical([7, 2], token_position=5, block_size=4) == (2, 1)

    def test_logical_to_physical_covers_every_position(self):
        table, block_size = [7, 2, 11], 4
        expected = [
            (7, 0),
            (7, 1),
            (7, 2),
            (7, 3),
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 3),
            (11, 0),
            (11, 1),
            (11, 2),
            (11, 3),
        ]
        assert [logical_to_physical(table, p, block_size) for p in range(12)] == expected

    def test_logical_to_physical_block_size_one(self):
        assert logical_to_physical([9, 4, 6], token_position=2, block_size=1) == (6, 0)

    def test_logical_to_physical_out_of_range_raises(self):
        with pytest.raises(IndexError):
            logical_to_physical([7, 2], token_position=8, block_size=4)
        with pytest.raises(IndexError):
            logical_to_physical([], token_position=0, block_size=4)

    def test_ensure_block_capacity_grows_to_exactly_what_is_needed(self):
        pool = BlockPool(8)
        table: list[int] = []
        ensure_block_capacity(table, pool, num_tokens=5, block_size=4)
        assert len(table) == 2
        assert pool.num_free_blocks == 6
        assert len(set(table)) == 2 and all(0 <= b < 8 for b in table)

    def test_ensure_block_capacity_is_incremental_and_never_shrinks(self):
        pool = BlockPool(8)
        table: list[int] = []
        ensure_block_capacity(table, pool, 4, 4)
        first = list(table)
        ensure_block_capacity(table, pool, 4, 4)  # already enough: no-op
        assert table == first and pool.num_free_blocks == 7
        ensure_block_capacity(table, pool, 5, 4)  # crossing a block boundary: one more block
        assert table[:1] == first and len(table) == 2 and pool.num_free_blocks == 6
        ensure_block_capacity(table, pool, 1, 4)  # fewer tokens: still no shrinking
        assert len(table) == 2 and pool.num_free_blocks == 6

    def test_ensure_block_capacity_zero_tokens(self):
        pool = BlockPool(4)
        table: list[int] = []
        ensure_block_capacity(table, pool, 0, 4)
        assert table == [] and pool.num_free_blocks == 4

    def test_ensure_block_capacity_exhaustion(self):
        pool = BlockPool(2)
        table: list[int] = []
        with pytest.raises(OutOfBlocksError):
            ensure_block_capacity(table, pool, num_tokens=12, block_size=4)


# ---------------------------------------------------------------------------
# 3.5 slot mapping
# ---------------------------------------------------------------------------


@pytest.mark.m3
@pytest.mark.checkpoint("3.5")
class TestSlotMapping:
    def test_plan_example(self):
        slots = compute_slot_mapping([7, 2], torch.tensor([5]), block_size=4)
        assert slots.tolist() == [9]
        assert slots.dtype == torch.int64

    def test_prompt_spanning_two_blocks(self):
        slots = compute_slot_mapping([7, 2], torch.arange(8), block_size=4)
        assert slots.tolist() == [28, 29, 30, 31, 8, 9, 10, 11]

    def test_scrambled_three_block_table(self):
        slots = compute_slot_mapping([7, 2, 11], torch.arange(10), block_size=4)
        assert slots.tolist() == [28, 29, 30, 31, 8, 9, 10, 11, 44, 45]

    def test_single_decode_position_crossing_boundary(self):
        # position 4 is the first token of logical block 1 -> physical 2, offset 0
        assert compute_slot_mapping([7, 2], torch.tensor([4]), 4).tolist() == [8]
        # position 3 is the last token of logical block 0 -> physical 7, offset 3
        assert compute_slot_mapping([7, 2], torch.tensor([3]), 4).tolist() == [31]

    def test_block_size_one(self):
        assert compute_slot_mapping([9, 4, 6], torch.tensor([0, 1, 2]), 1).tolist() == [9, 4, 6]

    def test_empty_positions(self):
        slots = compute_slot_mapping([7, 2], torch.tensor([], dtype=torch.long), 4)
        assert slots.shape == (0,) and slots.dtype == torch.int64

    def test_unsorted_positions_keep_order(self):
        assert compute_slot_mapping([7, 2], torch.tensor([5, 0, 7]), 4).tolist() == [9, 28, 11]

    def test_slots_are_unique_and_in_range(self):
        table, block_size, num_blocks = [3, 0, 5, 1], 4, 6
        slots = compute_slot_mapping(table, torch.arange(16), block_size)
        assert len(set(slots.tolist())) == 16
        assert slots.min() >= 0 and slots.max() < num_blocks * block_size

    def test_position_outside_table_raises(self):
        with pytest.raises(IndexError):
            compute_slot_mapping([7, 2], torch.tensor([8]), 4)


# ---------------------------------------------------------------------------
# 3.6 writing K/V into physical blocks
# ---------------------------------------------------------------------------


def _nan_fill(cache: PagedKVCache) -> None:
    cache.k_cache.fill_(float("nan"))
    cache.v_cache.fill_(float("nan"))


@pytest.mark.m3
@pytest.mark.checkpoint("3.6")
class TestPagedKVCacheWrite:
    def test_write_lands_in_the_addressed_slots(self, tiny_config):
        cache = PagedKVCache(tiny_config, num_blocks=8, block_size=4)
        _nan_fill(cache)
        shape = (tiny_config.num_kv_heads, tiny_config.head_dim)
        k = torch.randn(3, *shape)
        v = torch.randn(3, *shape)
        # slots 9, 28, 11  ->  (block 2, offset 1), (block 7, offset 0), (block 2, offset 3)
        cache.write(0, k, v, torch.tensor([9, 28, 11]))
        torch.testing.assert_close(cache.k_cache[0, 2, 1], k[0])
        torch.testing.assert_close(cache.k_cache[0, 7, 0], k[1])
        torch.testing.assert_close(cache.k_cache[0, 2, 3], k[2])
        torch.testing.assert_close(cache.v_cache[0, 2, 1], v[0])
        torch.testing.assert_close(cache.v_cache[0, 7, 0], v[1])
        torch.testing.assert_close(cache.v_cache[0, 2, 3], v[2])

    def test_nothing_else_changes(self, tiny_config):
        cache = PagedKVCache(tiny_config, num_blocks=8, block_size=4)
        _nan_fill(cache)
        shape = (tiny_config.num_kv_heads, tiny_config.head_dim)
        cache.write(1, torch.randn(2, *shape), torch.randn(2, *shape), torch.tensor([9, 28]))
        finite_k = (
            torch.isfinite(cache.k_cache).reshape(tiny_config.num_layers, -1, *shape).all(dim=-1).all(dim=-1)
        )
        finite_v = (
            torch.isfinite(cache.v_cache).reshape(tiny_config.num_layers, -1, *shape).all(dim=-1).all(dim=-1)
        )
        # exactly two slots of layer 1 are finite, none of layer 0
        assert finite_k[0].sum() == 0 and finite_v[0].sum() == 0
        assert finite_k[1].sum() == 2 and finite_v[1].sum() == 2
        assert finite_k[1, 9] and finite_k[1, 28]

    def test_layers_are_isolated(self, tiny_config):
        cache = PagedKVCache(tiny_config, num_blocks=4, block_size=2)
        shape = (tiny_config.num_kv_heads, tiny_config.head_dim)
        slots = torch.tensor([0, 1])
        for layer in range(tiny_config.num_layers):
            cache.write(
                layer,
                torch.full((2, *shape), float(layer + 1)),
                torch.full((2, *shape), -float(layer + 1)),
                slots,
            )
        for layer in range(tiny_config.num_layers):
            # slots 0 and 1 are the two offsets of physical block 0
            torch.testing.assert_close(cache.k_cache[layer, 0], torch.full((2, *shape), float(layer + 1)))
            torch.testing.assert_close(cache.v_cache[layer, 0], torch.full((2, *shape), -float(layer + 1)))

    def test_write_via_slot_mapping_across_block_boundary(self, tiny_config):
        """Write a 6-token prompt through block_table [3, 0] and check where it landed."""
        cache = PagedKVCache(tiny_config, num_blocks=4, block_size=4)
        _nan_fill(cache)
        shape = (tiny_config.num_kv_heads, tiny_config.head_dim)
        k = torch.randn(6, *shape)
        v = torch.randn(6, *shape)
        slots = compute_slot_mapping([3, 0], torch.arange(6), block_size=4)
        cache.write(0, k, v, slots)
        torch.testing.assert_close(cache.k_cache[0, 3], k[0:4])  # logical block 0 -> physical 3
        torch.testing.assert_close(cache.k_cache[0, 0, :2], k[4:6])  # logical block 1 -> physical 0, partial
        torch.testing.assert_close(cache.v_cache[0, 3], v[0:4])
        torch.testing.assert_close(cache.v_cache[0, 0, :2], v[4:6])
        assert torch.isnan(cache.k_cache[0, 0, 2:]).all()  # rest of the partial block untouched
        assert torch.isnan(cache.k_cache[0, 1]).all() and torch.isnan(cache.k_cache[0, 2]).all()

    def test_decode_write_of_single_token(self, tiny_config):
        cache = PagedKVCache(tiny_config, num_blocks=4, block_size=4)
        _nan_fill(cache)
        shape = (tiny_config.num_kv_heads, tiny_config.head_dim)
        k = torch.randn(1, *shape)
        cache.write(0, k, k, compute_slot_mapping([3, 0], torch.tensor([4]), 4))
        torch.testing.assert_close(cache.k_cache[0, 0, 0], k[0])
        assert torch.isfinite(cache.k_cache[0]).reshape(-1, *shape).all(-1).all(-1).sum() == 1

    def test_overwrite_replaces(self, tiny_config):
        cache = PagedKVCache(tiny_config, num_blocks=2, block_size=2)
        shape = (1, tiny_config.num_kv_heads, tiny_config.head_dim)
        cache.write(0, torch.ones(shape), torch.ones(shape), torch.tensor([3]))
        cache.write(0, torch.zeros(shape), torch.zeros(shape), torch.tensor([3]))
        torch.testing.assert_close(cache.k_cache[0, 1, 1], torch.zeros(shape[1:]))

    def test_slot_out_of_range_raises(self, tiny_config):
        cache = PagedKVCache(tiny_config, num_blocks=2, block_size=2)
        k = torch.randn(1, tiny_config.num_kv_heads, tiny_config.head_dim)
        with pytest.raises((IndexError, RuntimeError)):
            cache.write(0, k, k, torch.tensor([4]))

    def test_two_requests_share_the_pool_without_overlap(self, tiny_config):
        cache = PagedKVCache(tiny_config, num_blocks=8, block_size=4)
        _nan_fill(cache)
        shape = (tiny_config.num_kv_heads, tiny_config.head_dim)
        k_a, k_b = torch.randn(6, *shape), torch.randn(3, *shape)
        cache.write(0, k_a, k_a, compute_slot_mapping([7, 2], torch.arange(6), 4))
        cache.write(0, k_b, k_b, compute_slot_mapping([5], torch.arange(3), 4))
        torch.testing.assert_close(cache.k_cache[0, 7], k_a[:4])
        torch.testing.assert_close(cache.k_cache[0, 2, :2], k_a[4:])
        torch.testing.assert_close(cache.k_cache[0, 5, :3], k_b)
        for free_block in (0, 1, 3, 4, 6):
            assert torch.isnan(cache.k_cache[0, free_block]).all()


# ---------------------------------------------------------------------------
# AttentionMetadata (provided) sanity
# ---------------------------------------------------------------------------


@pytest.mark.m3
def test_attention_metadata_validation():
    AttentionMetadata(slot_mapping=torch.tensor([1, 2]), block_tables=[[0]], seq_lens=[2])
    with pytest.raises(ValueError):
        AttentionMetadata(slot_mapping=torch.tensor([1, 2]), block_tables=[[0], [1]], seq_lens=[2])
    with pytest.raises(ValueError):
        AttentionMetadata(slot_mapping=torch.tensor([[1, 2]]), block_tables=[[0]], seq_lens=[2])
