"""Milestone 4 -- checkpoints 4.1/4.2 (gather through the block table), 4.3 (attention),
4.4 (partial final block), 4.5 (TinyLlama integration, generate_paged)."""

import pytest
import torch

from helpers import RecordingModel, random_tokens
from tiny_vllm.attention import causal_attention, gather_kv, paged_attention, paged_attention_decode
from tiny_vllm.block_pool import BlockPool, OutOfBlocksError
from tiny_vllm.generation import generate_naive, generate_paged, generate_with_kv_cache, prefill
from tiny_vllm.kv_cache import (
    AttentionMetadata,
    ContiguousKVCache,
    PagedKVCache,
    blocks_needed,
    compute_slot_mapping,
)

pytestmark = pytest.mark.m4

TOL = dict(rtol=1e-4, atol=1e-4)
BLOCK_SIZE = 4
NUM_BLOCKS = 12
SCRAMBLED = [7, 2, 11]  # the plan's deliberately non-contiguous block table


def _pool_with_logical_kv(config, k_logical, v_logical, block_table, layer=0):
    """Place logically ordered K/V into physical blocks using the (already tested) M3 write path."""
    cache = PagedKVCache(config, NUM_BLOCKS, BLOCK_SIZE)
    cache.k_cache.fill_(float("nan"))  # anything not written is poison
    cache.v_cache.fill_(float("nan"))
    seq_len = k_logical.shape[0]
    slots = compute_slot_mapping(block_table, torch.arange(seq_len), BLOCK_SIZE)
    cache.write(layer, k_logical, v_logical, slots)
    return cache


# ---------------------------------------------------------------------------
# 4.1 / 4.2 / 4.4 gather_kv
# ---------------------------------------------------------------------------


@pytest.mark.checkpoint("4.1", "4.2")
class TestGatherKV:
    def test_scrambled_blocks_come_back_in_logical_order(self, tiny_config):
        shape = (tiny_config.num_kv_heads, tiny_config.head_dim)
        k = torch.randn(12, *shape)
        v = torch.randn(12, *shape)
        cache = _pool_with_logical_kv(tiny_config, k, v, SCRAMBLED)
        out_k = gather_kv(cache.k_cache[0], SCRAMBLED, seq_len=12)
        out_v = gather_kv(cache.v_cache[0], SCRAMBLED, seq_len=12)
        assert out_k.shape == (12, *shape)
        torch.testing.assert_close(out_k, k)
        torch.testing.assert_close(out_v, v)

    def test_physical_layout_does_not_matter(self, tiny_config):
        shape = (tiny_config.num_kv_heads, tiny_config.head_dim)
        k = torch.randn(9, *shape)
        a = gather_kv(_pool_with_logical_kv(tiny_config, k, k, [0, 1, 2]).k_cache[0], [0, 1, 2], 9)
        b = gather_kv(_pool_with_logical_kv(tiny_config, k, k, SCRAMBLED).k_cache[0], SCRAMBLED, 9)
        c = gather_kv(_pool_with_logical_kv(tiny_config, k, k, [11, 0, 5]).k_cache[0], [11, 0, 5], 9)
        torch.testing.assert_close(a, k)
        torch.testing.assert_close(b, k)
        torch.testing.assert_close(c, k)

    def test_reads_the_correct_layer(self, tiny_config):
        shape = (tiny_config.num_kv_heads, tiny_config.head_dim)
        k0, k1 = torch.randn(5, *shape), torch.randn(5, *shape)
        cache = PagedKVCache(tiny_config, NUM_BLOCKS, BLOCK_SIZE)
        slots = compute_slot_mapping(SCRAMBLED, torch.arange(5), BLOCK_SIZE)
        cache.write(0, k0, k0, slots)
        cache.write(1, k1, k1, slots)
        torch.testing.assert_close(gather_kv(cache.k_cache[0], SCRAMBLED, 5), k0)
        torch.testing.assert_close(gather_kv(cache.k_cache[1], SCRAMBLED, 5), k1)

    def test_single_token(self, tiny_config):
        shape = (tiny_config.num_kv_heads, tiny_config.head_dim)
        k = torch.randn(1, *shape)
        cache = _pool_with_logical_kv(tiny_config, k, k, [9])
        torch.testing.assert_close(gather_kv(cache.k_cache[0], [9], 1), k)

    def test_exactly_full_blocks(self, tiny_config):
        shape = (tiny_config.num_kv_heads, tiny_config.head_dim)
        k = torch.randn(8, *shape)
        cache = _pool_with_logical_kv(tiny_config, k, k, [3, 1])
        torch.testing.assert_close(gather_kv(cache.k_cache[0], [3, 1], 8), k)


@pytest.mark.checkpoint("4.4")
class TestPartialFinalBlock:
    @pytest.mark.parametrize("seq_len", [5, 6, 7, 9, 10, 11])
    def test_slots_beyond_seq_len_are_excluded(self, tiny_config, seq_len):
        shape = (tiny_config.num_kv_heads, tiny_config.head_dim)
        k = torch.randn(seq_len, *shape)
        cache = _pool_with_logical_kv(tiny_config, k, k, SCRAMBLED)  # unwritten slots are NaN
        out = gather_kv(cache.k_cache[0], SCRAMBLED, seq_len)
        assert out.shape[0] == seq_len
        assert torch.isfinite(out).all(), "gathered a slot beyond seq_len (stale/poison data)"
        torch.testing.assert_close(out, k)

    def test_extra_blocks_in_table_are_ignored(self, tiny_config):
        shape = (tiny_config.num_kv_heads, tiny_config.head_dim)
        k = torch.randn(5, *shape)
        cache = _pool_with_logical_kv(tiny_config, k, k, [7, 2])
        # block 11 was allocated ahead but holds nothing (NaN)
        out = gather_kv(cache.k_cache[0], [7, 2, 11], 5)
        torch.testing.assert_close(out, k)

    def test_partial_block_attention_matches_contiguous(self, tiny_config):
        shape = (tiny_config.num_kv_heads, tiny_config.head_dim)
        seq_len = 10  # 4 + 4 + 2
        k, v = torch.randn(seq_len, *shape), torch.randn(seq_len, *shape)
        q = torch.randn(seq_len, tiny_config.num_heads, tiny_config.head_dim)
        cache = _pool_with_logical_kv(tiny_config, k, v, SCRAMBLED)
        expected = causal_attention(q[None], k[None], v[None])[0]
        actual = paged_attention(q, cache.k_cache[0], cache.v_cache[0], SCRAMBLED, seq_len)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, **TOL)


# ---------------------------------------------------------------------------
# 4.3 paged attention
# ---------------------------------------------------------------------------


@pytest.mark.checkpoint("4.3")
class TestPagedAttention:
    @pytest.mark.parametrize("num_queries", [12, 3, 1])
    def test_matches_contiguous_attention_with_scrambled_blocks(self, tiny_config, num_queries):
        """The critical test: reference = contiguous attention, paged = block_table [7, 2, 11]."""
        shape = (tiny_config.num_kv_heads, tiny_config.head_dim)
        seq_len = 12
        k, v = torch.randn(seq_len, *shape), torch.randn(seq_len, *shape)
        q = torch.randn(num_queries, tiny_config.num_heads, tiny_config.head_dim)
        cache = _pool_with_logical_kv(tiny_config, k, v, SCRAMBLED)

        reference = causal_attention(q[None], k[None], v[None])[0]
        paged = paged_attention(q, cache.k_cache[0], cache.v_cache[0], block_table=SCRAMBLED, seq_len=seq_len)
        assert paged.shape == (num_queries, tiny_config.num_heads, tiny_config.head_dim)
        torch.testing.assert_close(reference, paged, **TOL)

    def test_result_independent_of_physical_order(self, tiny_config):
        shape = (tiny_config.num_kv_heads, tiny_config.head_dim)
        seq_len = 11
        k, v = torch.randn(seq_len, *shape), torch.randn(seq_len, *shape)
        q = torch.randn(1, tiny_config.num_heads, tiny_config.head_dim)
        outs = []
        for table in ([0, 1, 2], SCRAMBLED, [10, 4, 0]):
            cache = _pool_with_logical_kv(tiny_config, k, v, table)
            outs.append(paged_attention(q, cache.k_cache[0], cache.v_cache[0], table, seq_len))
        torch.testing.assert_close(outs[0], outs[1], **TOL)
        torch.testing.assert_close(outs[0], outs[2], **TOL)

    def test_logical_order_is_what_matters(self, tiny_config):
        """Swapping two logical blocks (same physical set) must change the answer.

        Note: a lone decode query sees the *whole* history, and softmax attention is
        permutation-invariant over its keys, so the swap is only observable through the
        causal mask -- hence prefill-style queries here.
        """
        shape = (tiny_config.num_kv_heads, tiny_config.head_dim)
        seq_len = 8
        k, v = torch.randn(seq_len, *shape), torch.randn(seq_len, *shape)
        q = torch.randn(seq_len, tiny_config.num_heads, tiny_config.head_dim)
        cache = _pool_with_logical_kv(tiny_config, k, v, [7, 2])
        correct = paged_attention(q, cache.k_cache[0], cache.v_cache[0], [7, 2], seq_len)
        swapped = paged_attention(q, cache.k_cache[0], cache.v_cache[0], [2, 7], seq_len)
        assert not torch.allclose(correct[:4], swapped[:4], atol=1e-3)  # early queries see different keys
        torch.testing.assert_close(
            correct[-1], swapped[-1], **TOL
        )  # the last query sees everything either way

    def test_decode_query_sees_whole_history(self, tiny_config):
        shape = (tiny_config.num_kv_heads, tiny_config.head_dim)
        seq_len = 9
        k, v = torch.randn(seq_len, *shape), torch.randn(seq_len, *shape)
        q = torch.randn(1, tiny_config.num_heads, tiny_config.head_dim)
        cache = _pool_with_logical_kv(tiny_config, k, v, SCRAMBLED)
        full = paged_attention(q, cache.k_cache[0], cache.v_cache[0], SCRAMBLED, seq_len)
        # Change the earliest token's V: the decode output must change (it attends to everything).
        v2 = v.clone()
        v2[0] += 5.0
        cache2 = _pool_with_logical_kv(tiny_config, k, v2, SCRAMBLED)
        changed = paged_attention(q, cache2.k_cache[0], cache2.v_cache[0], SCRAMBLED, seq_len)
        assert not torch.allclose(full, changed, atol=1e-3)


# ---------------------------------------------------------------------------
# 4.5 integration with TinyLlama
# ---------------------------------------------------------------------------


def _prefill_metadata(block_table, seq_len):
    return AttentionMetadata(
        slot_mapping=compute_slot_mapping(block_table, torch.arange(seq_len), BLOCK_SIZE),
        block_tables=[list(block_table)],
        seq_lens=[seq_len],
    )


@pytest.mark.checkpoint("4.5")
class TestModelWithPagedKV:
    def test_paged_prefill_logits_match_uncached(self, tiny_model, tiny_config):
        tokens = random_tokens(10, tiny_config.vocab_size, seed=31)
        input_ids = torch.tensor([tokens])
        positions = torch.arange(10)[None]
        cache = PagedKVCache(tiny_config, NUM_BLOCKS, BLOCK_SIZE)
        with torch.no_grad():
            expected = tiny_model(input_ids, positions)
            actual = tiny_model(
                input_ids, positions, kv_cache=cache, attn_metadata=_prefill_metadata(SCRAMBLED, 10)
            )
        torch.testing.assert_close(actual, expected, **TOL)

    def test_paged_decode_logits_match_uncached(self, tiny_model, tiny_config):
        tokens = random_tokens(12, tiny_config.vocab_size, seed=32)
        cache = PagedKVCache(tiny_config, NUM_BLOCKS, BLOCK_SIZE)
        table = [7, 2]
        with torch.no_grad():
            tiny_model(
                torch.tensor([tokens[:6]]),
                torch.arange(6)[None],
                kv_cache=cache,
                attn_metadata=_prefill_metadata(table, 6),
            )
            for p in range(6, 12):
                if blocks_needed(p + 1, BLOCK_SIZE) > len(table):
                    table.append(11)  # crossing into logical block 2
                metadata = AttentionMetadata(
                    slot_mapping=compute_slot_mapping(table, torch.tensor([p]), BLOCK_SIZE),
                    block_tables=[list(table)],
                    seq_lens=[p + 1],
                )
                actual = tiny_model(
                    torch.tensor([[tokens[p]]]), torch.tensor([[p]]), kv_cache=cache, attn_metadata=metadata
                )
                expected = tiny_model(torch.tensor([tokens[: p + 1]]), torch.arange(p + 1)[None])[:, -1:]
                torch.testing.assert_close(actual, expected, **TOL, msg=f"position {p}")

    def test_every_layer_writes_the_same_kv_as_the_contiguous_cache(self, tiny_model, tiny_config):
        """Invariant 6: each Transformer layer uses its own, correct KV storage."""
        tokens = random_tokens(10, tiny_config.vocab_size, seed=33)
        input_ids = torch.tensor([tokens])
        positions = torch.arange(10)[None]
        paged = PagedKVCache(tiny_config, NUM_BLOCKS, BLOCK_SIZE)
        paged.k_cache.fill_(float("nan"))
        paged.v_cache.fill_(float("nan"))
        contiguous = ContiguousKVCache(tiny_config, max_seq_len=16)
        with torch.no_grad():
            tiny_model(input_ids, positions, kv_cache=paged, attn_metadata=_prefill_metadata(SCRAMBLED, 10))
        prefill(tiny_model, contiguous, tokens)
        for layer in range(tiny_config.num_layers):
            k_c, v_c = contiguous.read(layer, 10)
            torch.testing.assert_close(
                gather_kv(paged.k_cache[layer], SCRAMBLED, 10), k_c, **TOL, msg=f"K layer {layer}"
            )
            torch.testing.assert_close(
                gather_kv(paged.v_cache[layer], SCRAMBLED, 10), v_c, **TOL, msg=f"V layer {layer}"
            )
            # nothing was written outside the request's blocks
            for b in range(NUM_BLOCKS):
                if b not in SCRAMBLED:
                    assert torch.isnan(paged.k_cache[layer, b]).all()

    def test_missing_metadata_raises(self, tiny_model, tiny_config):
        cache = PagedKVCache(tiny_config, NUM_BLOCKS, BLOCK_SIZE)
        with pytest.raises(ValueError):
            tiny_model(torch.tensor([[5, 6]]), torch.arange(2)[None], kv_cache=cache)


@pytest.mark.checkpoint("4.5")
class TestGeneratePaged:
    @pytest.mark.parametrize("seed", [34, 35])
    def test_matches_naive_and_cached_generation(self, tiny_model, tiny_config, seed):
        prompt = random_tokens(7, tiny_config.vocab_size, seed=seed)
        cache = PagedKVCache(tiny_config, NUM_BLOCKS, BLOCK_SIZE)
        pool = BlockPool(NUM_BLOCKS)
        paged = generate_paged(tiny_model, cache, pool, prompt, max_new_tokens=9, eos_token_id=2)
        assert paged == generate_naive(tiny_model, prompt, 9, eos_token_id=2)
        assert paged == generate_with_kv_cache(tiny_model, prompt, 9, eos_token_id=2)

    def test_blocks_are_returned_to_the_pool(self, tiny_model, tiny_config):
        prompt = random_tokens(5, tiny_config.vocab_size, seed=36)
        cache = PagedKVCache(tiny_config, NUM_BLOCKS, BLOCK_SIZE)
        pool = BlockPool(NUM_BLOCKS)
        generate_paged(tiny_model, cache, pool, prompt, max_new_tokens=8)
        assert pool.num_free_blocks == NUM_BLOCKS

    def test_works_in_a_fragmented_pool(self, tiny_model, tiny_config):
        prompt = random_tokens(6, tiny_config.vocab_size, seed=37)
        cache = PagedKVCache(tiny_config, NUM_BLOCKS, BLOCK_SIZE)
        pool = BlockPool(NUM_BLOCKS)
        held = [pool.allocate() for _ in range(NUM_BLOCKS)]
        for b in held:
            if b % 2 == 0:
                pool.free(b)  # only even blocks are free: forced non-contiguous
        out = generate_paged(tiny_model, cache, pool, prompt, max_new_tokens=10)
        assert out == generate_naive(tiny_model, prompt, 10)
        assert pool.num_free_blocks == NUM_BLOCKS // 2

    def test_forward_shapes_and_metadata(self, tiny_model, tiny_config):
        model = RecordingModel(tiny_model)
        prompt = random_tokens(6, tiny_config.vocab_size, seed=38)
        cache = PagedKVCache(tiny_config, NUM_BLOCKS, BLOCK_SIZE)
        pool = BlockPool(NUM_BLOCKS)
        out = generate_paged(model, cache, pool, prompt, max_new_tokens=5)
        assert len(out) == 5
        assert model.input_lengths() == [6, 1, 1, 1, 1]
        for i, call in enumerate(model.calls):
            md = call["attn_metadata"]
            assert isinstance(md, AttentionMetadata)
            assert call["kv_cache"] is cache
            seq_len = 6 + i
            assert md.seq_lens == [seq_len]
            assert md.slot_mapping.shape == (call["input_ids"].numel(),)
            # blocks are allocated only when needed (no big up-front reservation)
            assert (
                blocks_needed(seq_len, BLOCK_SIZE)
                <= len(md.block_tables[0])
                <= blocks_needed(seq_len, BLOCK_SIZE) + 1
            )

    def test_stops_at_eos(self, tiny_model, tiny_config):
        prompt = random_tokens(5, tiny_config.vocab_size, seed=39)
        naive = generate_naive(tiny_model, prompt, 12)
        eos = naive[3]  # pretend the 4th generated token is EOS
        cache = PagedKVCache(tiny_config, NUM_BLOCKS, BLOCK_SIZE)
        pool = BlockPool(NUM_BLOCKS)
        assert generate_paged(tiny_model, cache, pool, prompt, 12, eos_token_id=eos) == naive[:4]

    def test_out_of_blocks_is_explicit_and_leaks_nothing(self, tiny_model, tiny_config):
        prompt = random_tokens(6, tiny_config.vocab_size, seed=40)
        cache = PagedKVCache(tiny_config, num_blocks=2, block_size=BLOCK_SIZE)
        pool = BlockPool(2)  # 8 slots: enough for the prompt, not for prompt + 6 new tokens
        with pytest.raises(OutOfBlocksError):
            generate_paged(tiny_model, cache, pool, prompt, max_new_tokens=6)
        assert pool.num_free_blocks == 2


# ---------------------------------------------------------------------------
# 4.6 blockwise decode attention (online softmax, no gather)
# ---------------------------------------------------------------------------


def _forbid_gather(monkeypatch):
    """Make the gather path unusable so a test can prove it was not taken."""
    import tiny_vllm.attention as attention_module

    def boom(*args, **kwargs):
        raise AssertionError("the blockwise decode path must not gather the sequence")

    monkeypatch.setattr(attention_module, "gather_kv", boom)
    monkeypatch.setattr(attention_module, "paged_attention", boom)


@pytest.mark.checkpoint("4.6")
class TestPagedAttentionDecode:
    @pytest.mark.parametrize("seq_len", [1, 3, 4, 5, 8, 9, 12])
    def test_matches_gather_based_attention(self, tiny_config, seq_len):
        shape = (tiny_config.num_kv_heads, tiny_config.head_dim)
        k, v = torch.randn(seq_len, *shape), torch.randn(seq_len, *shape)
        q = torch.randn(tiny_config.num_heads, tiny_config.head_dim)
        cache = _pool_with_logical_kv(tiny_config, k, v, SCRAMBLED)
        expected = paged_attention(q[None], cache.k_cache[0], cache.v_cache[0], SCRAMBLED, seq_len)[0]
        actual = paged_attention_decode(q, cache.k_cache[0], cache.v_cache[0], SCRAMBLED, seq_len)
        assert actual.shape == (tiny_config.num_heads, tiny_config.head_dim)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)

    def test_does_not_gather(self, tiny_config, monkeypatch):
        shape = (tiny_config.num_kv_heads, tiny_config.head_dim)
        k, v = torch.randn(10, *shape), torch.randn(10, *shape)
        q = torch.randn(tiny_config.num_heads, tiny_config.head_dim)
        cache = _pool_with_logical_kv(tiny_config, k, v, SCRAMBLED)
        expected = causal_attention(q[None, None], k[None], v[None])[0, 0]
        _forbid_gather(monkeypatch)
        actual = paged_attention_decode(q, cache.k_cache[0], cache.v_cache[0], SCRAMBLED, 10)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)

    def test_partial_last_block_ignores_poisoned_slots(self, tiny_config):
        shape = (tiny_config.num_kv_heads, tiny_config.head_dim)
        for seq_len in (5, 6, 7, 10, 11):
            k, v = torch.randn(seq_len, *shape), torch.randn(seq_len, *shape)
            q = torch.randn(tiny_config.num_heads, tiny_config.head_dim)
            cache = _pool_with_logical_kv(tiny_config, k, v, SCRAMBLED)  # unwritten slots are NaN
            out = paged_attention_decode(q, cache.k_cache[0], cache.v_cache[0], SCRAMBLED, seq_len)
            assert torch.isfinite(out).all(), f"seq_len={seq_len}: read a slot beyond seq_len"

    def test_extra_blocks_in_table_are_ignored(self, tiny_config):
        shape = (tiny_config.num_kv_heads, tiny_config.head_dim)
        k, v = torch.randn(5, *shape), torch.randn(5, *shape)
        q = torch.randn(tiny_config.num_heads, tiny_config.head_dim)
        cache = _pool_with_logical_kv(tiny_config, k, v, [7, 2])
        expected = causal_attention(q[None, None], k[None], v[None])[0, 0]
        out = paged_attention_decode(q, cache.k_cache[0], cache.v_cache[0], [7, 2, 11], 5)
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-5)

    def test_numerically_stable_across_blocks(self, tiny_config):
        """Scores differ by hundreds between blocks: a per-block softmax without the running
        max (or without rescaling) overflows or returns garbage. The correct answer is finite."""
        shape = (tiny_config.num_kv_heads, tiny_config.head_dim)
        seq_len = 12
        k = torch.randn(seq_len, *shape)
        v = torch.randn(seq_len, *shape)
        q = torch.randn(tiny_config.num_heads, tiny_config.head_dim)
        # Make one key in the LAST block align hugely with the query so the true max lives there,
        # while an earlier block contains a moderately large score that a naive method would
        # normalise on its own.
        rep = tiny_config.num_heads // tiny_config.num_kv_heads
        q_kv = q.view(tiny_config.num_kv_heads, rep, -1).mean(dim=1)  # [num_kv_heads, head_dim]
        k[1] = q_kv * 40.0
        k[11] = q_kv * 400.0
        cache = _pool_with_logical_kv(tiny_config, k, v, SCRAMBLED)
        expected = causal_attention(q[None, None], k[None], v[None])[0, 0]
        actual = paged_attention_decode(q, cache.k_cache[0], cache.v_cache[0], SCRAMBLED, seq_len)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)

    def test_block_size_one(self, tiny_config):
        shape = (tiny_config.num_kv_heads, tiny_config.head_dim)
        k, v = torch.randn(6, *shape), torch.randn(6, *shape)
        q = torch.randn(tiny_config.num_heads, tiny_config.head_dim)
        cache = PagedKVCache(tiny_config, num_blocks=8, block_size=1)
        table = [5, 0, 7, 2, 6, 1]
        cache.write(0, k, v, compute_slot_mapping(table, torch.arange(6), 1))
        expected = causal_attention(q[None, None], k[None], v[None])[0, 0]
        out = paged_attention_decode(q, cache.k_cache[0], cache.v_cache[0], table, 6)
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-5)

    def test_model_decode_uses_the_blockwise_path(self, tiny_model, tiny_config, monkeypatch):
        """After 4.6, a decode step through the model must not gather; prefill may."""
        tokens = random_tokens(9, tiny_config.vocab_size, seed=46)
        cache = PagedKVCache(tiny_config, NUM_BLOCKS, BLOCK_SIZE)
        table = [7, 2, 11]
        with torch.no_grad():
            tiny_model(
                torch.tensor([tokens[:8]]),
                torch.arange(8)[None],
                kv_cache=cache,
                attn_metadata=_prefill_metadata(table, 8),
            )
            expected = tiny_model(torch.tensor([tokens]), torch.arange(9)[None])[:, -1:]
            _forbid_gather(monkeypatch)
            metadata = AttentionMetadata(
                slot_mapping=compute_slot_mapping(table, torch.tensor([8]), BLOCK_SIZE),
                block_tables=[table],
                seq_lens=[9],
            )
            actual = tiny_model(
                torch.tensor([[tokens[8]]]), torch.tensor([[8]]), kv_cache=cache, attn_metadata=metadata
            )
        torch.testing.assert_close(actual, expected, **TOL)


@pytest.mark.integration
def test_m4_integration_paged_generation_over_scrambled_blocks(tiny_model, tiny_config):
    """generate_paged in a fragmented pool == naive generation, and the pool is clean afterwards."""
    cache = PagedKVCache(tiny_config, NUM_BLOCKS, BLOCK_SIZE)
    pool = BlockPool(NUM_BLOCKS)
    reserved = [pool.allocate() for _ in range(NUM_BLOCKS)]
    keep = {b for b in reserved if b in (0, 1, 3, 4, 6, 8, 9)}
    for b in reserved:
        if b not in keep:
            pool.free(b)  # free = {2, 5, 7, 10, 11}
    for seed in (41, 42, 43):
        prompt = random_tokens(5 + seed % 3, tiny_config.vocab_size, seed=seed)
        assert generate_paged(tiny_model, cache, pool, prompt, 10) == generate_naive(tiny_model, prompt, 10)
        assert pool.num_free_blocks == NUM_BLOCKS - len(keep)


@pytest.mark.integration
@pytest.mark.hf
def test_m4_integration_real_model(real_model_and_tokenizer):
    model, tokenizer = real_model_and_tokenizer
    prompt = tokenizer("Paged attention lets", return_tensors="pt").input_ids[0].tolist()
    cache = PagedKVCache(model.config, num_blocks=16, block_size=4)
    pool = BlockPool(16)
    for b in [pool.allocate() for _ in range(16)]:
        if b % 3 != 0:
            pool.free(b)
    paged = generate_paged(model, cache, pool, prompt, 6, eos_token_id=tokenizer.eos_token_id)
    assert paged == generate_naive(model, prompt, 6, eos_token_id=tokenizer.eos_token_id)
