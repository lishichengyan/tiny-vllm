"""Milestone 3 -- checkpoints 3.2 (allocation), 3.3 (free + reuse), the fragmentation test,
the visualization helper and the milestone integration test."""

import pytest
import torch

from tiny_vllm.block_pool import BlockPool, OutOfBlocksError
from tiny_vllm.kv_cache import PagedKVCache, compute_slot_mapping, ensure_block_capacity
from tiny_vllm.visualize import format_kv_pool

pytestmark = pytest.mark.m3


# ---------------------------------------------------------------------------
# 3.2 allocation
# ---------------------------------------------------------------------------


@pytest.mark.checkpoint("3.2")
class TestAllocate:
    def test_fresh_pool_is_fully_free(self):
        pool = BlockPool(8)
        assert pool.num_blocks == 8
        assert pool.num_free_blocks == 8

    def test_invalid_size_raises(self):
        with pytest.raises(ValueError):
            BlockPool(0)

    def test_allocate_returns_distinct_in_range_ids(self):
        pool = BlockPool(8)
        ids = [pool.allocate() for _ in range(8)]
        assert all(isinstance(i, int) for i in ids)
        assert sorted(ids) == list(range(8))  # every block handed out exactly once

    def test_free_count_decreases(self):
        pool = BlockPool(5)
        for expected_free in (4, 3, 2, 1, 0):
            pool.allocate()
            assert pool.num_free_blocks == expected_free

    def test_exhaustion_is_explicit(self):
        pool = BlockPool(3)
        for _ in range(3):
            pool.allocate()
        with pytest.raises(OutOfBlocksError):
            pool.allocate()
        assert pool.num_free_blocks == 0  # a failed allocation changes nothing
        assert isinstance(OutOfBlocksError(), RuntimeError)

    def test_single_block_pool(self):
        pool = BlockPool(1)
        assert pool.allocate() == 0
        with pytest.raises(OutOfBlocksError):
            pool.allocate()


# ---------------------------------------------------------------------------
# 3.3 free + reuse
# ---------------------------------------------------------------------------


@pytest.mark.checkpoint("3.3")
class TestFreeAndReuse:
    def test_free_returns_block_to_pool(self):
        pool = BlockPool(4)
        a = pool.allocate()
        assert pool.num_free_blocks == 3
        pool.free(a)
        assert pool.num_free_blocks == 4

    def test_freed_block_is_reused(self):
        pool = BlockPool(1)
        a = pool.allocate()
        pool.free(a)
        assert pool.allocate() == a

    def test_freed_blocks_are_the_only_ones_handed_out_again(self):
        pool = BlockPool(6)
        held = [pool.allocate() for _ in range(6)]
        released = {held[1], held[4]}
        for b in released:
            pool.free(b)
        assert {pool.allocate(), pool.allocate()} == released

    def test_double_free_raises(self):
        pool = BlockPool(4)
        a = pool.allocate()
        pool.free(a)
        with pytest.raises(ValueError):
            pool.free(a)
        assert pool.num_free_blocks == 4

    def test_free_never_allocated_raises(self):
        pool = BlockPool(4)
        with pytest.raises(ValueError):
            pool.free(2)

    def test_free_out_of_range_raises(self):
        pool = BlockPool(4)
        with pytest.raises(ValueError):
            pool.free(4)
        with pytest.raises(ValueError):
            pool.free(-1)

    def test_interleaved_allocate_free_keeps_invariants(self):
        """Randomised allocate/free sequence: live blocks are unique, counts add up."""
        pool = BlockPool(10)
        live: set[int] = set()
        g = torch.Generator().manual_seed(123)
        for _ in range(400):
            do_alloc = live == set() or (len(live) < 10 and torch.rand(1, generator=g).item() < 0.55)
            if do_alloc:
                b = pool.allocate()
                assert 0 <= b < 10
                assert b not in live, "allocated a block that is still live"
                live.add(b)
            else:
                victim = sorted(live)[int(torch.randint(len(live), (1,), generator=g))]
                pool.free(victim)
                live.remove(victim)
            assert pool.num_free_blocks == 10 - len(live)
        # drain
        while len(live) < 10:
            live.add(pool.allocate())
        with pytest.raises(OutOfBlocksError):
            pool.allocate()


# ---------------------------------------------------------------------------
# Fragmentation (required by the plan)
# ---------------------------------------------------------------------------


@pytest.mark.checkpoint("3.2", "3.3")
def test_fragmented_pool_serves_non_contiguous_blocks():
    """
    physical blocks:
    0     1     2     3     4     5     6     7
    USED  FREE  USED  FREE  USED  FREE  USED  FREE
    A request needing four blocks must get {1, 3, 5, 7} -- no contiguity required.
    """
    pool = BlockPool(8)
    ids = [pool.allocate() for _ in range(8)]
    assert sorted(ids) == list(range(8))
    for b in (1, 3, 5, 7):
        pool.free(b)
    assert pool.num_free_blocks == 4

    block_table: list[int] = []
    ensure_block_capacity(block_table, pool, num_tokens=16, block_size=4)
    assert sorted(block_table) == [1, 3, 5, 7]
    assert pool.num_free_blocks == 0

    # and the table addresses the right slots
    slots = compute_slot_mapping(block_table, torch.arange(16), block_size=4)
    expected = [b * 4 + off for b in block_table for off in range(4)]
    assert slots.tolist() == expected


# ---------------------------------------------------------------------------
# Visualization helper (provided infrastructure)
# ---------------------------------------------------------------------------


class TestVisualize:
    def test_plan_example(self):
        text = format_kv_pool(8, 4, {"A": [7, 2], "B": [5]}, {"A": 8, "B": 4})
        lines = text.splitlines()
        assert "Request A" in lines and "block_table = [7, 2]" in lines
        assert "Request B" in lines and "block_table = [5]" in lines
        assert "0 FREE" in lines and "1 FREE" in lines and "3 FREE" in lines
        assert "2 A [tokens 4-7]" in lines
        assert "5 B [tokens 0-3]" in lines
        assert "7 A [tokens 0-3]" in lines
        assert lines[-1] == "free = [0, 1, 3, 4, 6]"

    def test_partial_and_empty_blocks(self):
        text = format_kv_pool(4, 4, {"A": [1, 3, 0]}, {"A": 6})
        assert "1 A [tokens 0-3]" in text
        assert "3 A [tokens 4-5]" in text
        assert "0 A [empty]" in text

    def test_without_seq_lens(self):
        text = format_kv_pool(3, 2, {"X": [2]})
        assert "2 X" in text.splitlines()

    def test_detects_double_ownership(self):
        with pytest.raises(ValueError):
            format_kv_pool(4, 2, {"A": [1], "B": [1]})

    def test_detects_out_of_range(self):
        with pytest.raises(ValueError):
            format_kv_pool(4, 2, {"A": [4]})


# ---------------------------------------------------------------------------
# Milestone 3 integration
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_m3_integration_two_requests_share_a_fragmented_pool(tiny_config):
    """Allocate for two requests in a fragmented pool, write their K/V through slot mappings,
    verify isolation, free one request, and reuse its blocks for a third request."""
    block_size, num_blocks = 4, 8
    pool = BlockPool(num_blocks)
    cache = PagedKVCache(tiny_config, num_blocks, block_size)
    cache.k_cache.fill_(float("nan"))
    cache.v_cache.fill_(float("nan"))
    shape = (tiny_config.num_kv_heads, tiny_config.head_dim)

    # Fragment the pool: blocks {0, 2, 4, 6} stay reserved by "someone else".
    reserved = [pool.allocate() for _ in range(num_blocks)]
    for b in reserved:
        if b % 2 == 1:
            pool.free(b)
    assert pool.num_free_blocks == 4

    # Request A: 6 tokens -> 2 blocks.  Request B: 3 tokens -> 1 block.
    table_a: list[int] = []
    table_b: list[int] = []
    ensure_block_capacity(table_a, pool, 6, block_size)
    ensure_block_capacity(table_b, pool, 3, block_size)
    assert len(table_a) == 2 and len(table_b) == 1
    assert set(table_a).isdisjoint(table_b)
    assert set(table_a + table_b) <= {1, 3, 5, 7}
    assert pool.num_free_blocks == 1

    k_a = torch.randn(6, *shape)
    k_b = torch.randn(3, *shape)
    for layer in range(tiny_config.num_layers):
        cache.write(
            layer, k_a + layer, -(k_a + layer), compute_slot_mapping(table_a, torch.arange(6), block_size)
        )
        cache.write(
            layer, k_b + layer, -(k_b + layer), compute_slot_mapping(table_b, torch.arange(3), block_size)
        )

    for layer in range(tiny_config.num_layers):
        torch.testing.assert_close(cache.k_cache[layer, table_a[0]], k_a[:4] + layer)
        torch.testing.assert_close(cache.k_cache[layer, table_a[1], :2], k_a[4:] + layer)
        torch.testing.assert_close(cache.v_cache[layer, table_b[0], :3], -(k_b + layer))
        assert torch.isnan(cache.k_cache[layer, table_a[1], 2:]).all()
        for b in (0, 2, 4, 6):
            assert torch.isnan(cache.k_cache[layer, b]).all()

    picture = format_kv_pool(num_blocks, block_size, {"A": table_a, "B": table_b}, {"A": 6, "B": 3})
    assert f"{table_a[0]} A [tokens 0-3]" in picture
    assert f"{table_b[0]} B [tokens 0-2]" in picture

    # A finishes: its blocks go back and a new request C can take them.
    for b in table_a:
        pool.free(b)
    assert pool.num_free_blocks == 3
    table_c: list[int] = []
    ensure_block_capacity(table_c, pool, 12, block_size)
    assert set(table_a) <= set(table_c)
    assert set(table_c).isdisjoint(table_b)
    assert pool.num_free_blocks == 0
    with pytest.raises(OutOfBlocksError):
        pool.allocate()
