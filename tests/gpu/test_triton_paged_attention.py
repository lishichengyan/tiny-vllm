"""GPU track -- Triton decode kernel vs the CPU reference (docs/08-gpu-track.md).

Skipped automatically unless CUDA and Triton are available:

    pytest tests/gpu -v
"""

import pytest
import torch

from tiny_vllm.kernels import gpu_track_available

if not gpu_track_available():
    pytest.skip("GPU track needs CUDA + Triton", allow_module_level=True)

from tiny_vllm.attention import causal_attention, paged_attention_decode  # noqa: E402
from tiny_vllm.config import TinyLlamaConfig  # noqa: E402
from tiny_vllm.kernels.paged_attention_triton import (  # noqa: E402
    build_block_tables_tensor,
    paged_attention_decode_triton,
)
from tiny_vllm.kv_cache import PagedKVCache, compute_slot_mapping  # noqa: E402

pytestmark = pytest.mark.gpu

DEVICE = torch.device("cuda")
NUM_BLOCKS = 32


def _config(num_heads=8, num_kv_heads=2, head_dim=64):
    return TinyLlamaConfig(
        vocab_size=128,
        hidden_size=num_heads * head_dim,
        intermediate_size=256,
        num_layers=1,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )


def _scenario(config, block_size, seq_lens, dtype, seed=0):
    """Random logical K/V per sequence, placed into a shared pool through disjoint scrambled tables."""
    g = torch.Generator().manual_seed(seed)
    cache = PagedKVCache(config, NUM_BLOCKS, block_size, dtype=dtype, device=DEVICE)
    cache.k_cache.fill_(float("nan"))
    cache.v_cache.fill_(float("nan"))
    perm = torch.randperm(NUM_BLOCKS, generator=g).tolist()
    tables, ks, vs, qs = [], [], [], []
    for n in seq_lens:
        needed = (n + block_size - 1) // block_size
        table, perm = perm[:needed], perm[needed:]
        k = torch.randn(n, config.num_kv_heads, config.head_dim, generator=g).to(dtype)
        v = torch.randn(n, config.num_kv_heads, config.head_dim, generator=g).to(dtype)
        cache.write(
            0, k.to(DEVICE), v.to(DEVICE), compute_slot_mapping(table, torch.arange(n), block_size).to(DEVICE)
        )
        tables.append(table)
        ks.append(k)
        vs.append(v)
        qs.append(torch.randn(config.num_heads, config.head_dim, generator=g).to(dtype))
    return cache, tables, ks, vs, torch.stack(qs)


def _reference(qs, ks, vs):
    """Plain causal attention on the CPU, one sequence at a time, in float32."""
    outs = []
    for q, k, v in zip(qs, ks, vs):
        outs.append(causal_attention(q[None, None].float(), k[None].float(), v[None].float())[0, 0])
    return torch.stack(outs)


def _run_triton(cache, tables, seq_lens, qs):
    bt = build_block_tables_tensor(tables, DEVICE)
    sl = torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE)
    return paged_attention_decode_triton(
        qs.to(DEVICE).contiguous(), cache.k_cache[0], cache.v_cache[0], bt, sl
    )


@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-4), (torch.float16, 2e-2)])
def test_matches_reference_scrambled_blocks(dtype, tol):
    config = _config()
    seq_lens = [37, 1, 16, 5, 64, 17]
    cache, tables, ks, vs, qs = _scenario(config, 16, seq_lens, dtype)
    out = _run_triton(cache, tables, seq_lens, qs)
    assert out.shape == qs.shape and out.dtype == dtype
    assert torch.isfinite(out).all(), "read a poisoned slot beyond seq_len or an unused block"
    torch.testing.assert_close(out.float().cpu(), _reference(qs, ks, vs), rtol=tol, atol=tol)


@pytest.mark.parametrize("block_size", [4, 16, 32])
def test_block_sizes(block_size):
    config = _config()
    seq_lens = [block_size - 1, block_size, block_size + 1, 3 * block_size + 2]
    cache, tables, ks, vs, qs = _scenario(config, block_size, seq_lens, torch.float32, seed=block_size)
    out = _run_triton(cache, tables, seq_lens, qs)
    torch.testing.assert_close(out.cpu(), _reference(qs, ks, vs), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("num_heads,num_kv_heads", [(8, 8), (8, 2), (9, 3), (4, 1)])
def test_grouped_query_attention(num_heads, num_kv_heads):
    config = _config(num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=64)
    seq_lens = [23, 40]
    cache, tables, ks, vs, qs = _scenario(config, 16, seq_lens, torch.float32, seed=7)
    out = _run_triton(cache, tables, seq_lens, qs)
    torch.testing.assert_close(out.cpu(), _reference(qs, ks, vs), rtol=1e-4, atol=1e-4)


def test_matches_cpu_blockwise_implementation():
    """The Triton kernel and checkpoint 4.6 are the same algorithm; they must agree."""
    config = _config(num_heads=4, num_kv_heads=2, head_dim=32)
    seq_lens = [10, 33]
    cache, tables, ks, vs, qs = _scenario(config, 8, seq_lens, torch.float32, seed=3)
    out = _run_triton(cache, tables, seq_lens, qs).cpu()
    k_cpu, v_cpu = cache.k_cache[0].cpu(), cache.v_cache[0].cpu()
    for b, (table, n) in enumerate(zip(tables, seq_lens)):
        cpu = paged_attention_decode(qs[b], k_cpu, v_cpu, table, n)
        torch.testing.assert_close(out[b], cpu, rtol=1e-4, atol=1e-4)


def test_numerically_stable_across_blocks():
    config = _config(num_heads=4, num_kv_heads=4, head_dim=64)
    n = 48
    cache, tables, ks, vs, qs = _scenario(config, 16, [n], torch.float32, seed=11)
    # make the true max live in the last block with a huge score
    q = qs[0]
    k = ks[0].clone()
    k[5] = q * 30.0
    k[n - 1] = q * 300.0
    cache.write(
        0, k.to(DEVICE), vs[0].to(DEVICE), compute_slot_mapping(tables[0], torch.arange(n), 16).to(DEVICE)
    )
    out = _run_triton(cache, tables, [n], qs)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.cpu(), _reference(qs, [k], vs), rtol=1e-4, atol=1e-4)


def test_padded_block_table_entries_are_never_read():
    config = _config()
    cache, tables, ks, vs, qs = _scenario(config, 16, [5, 70], torch.float32, seed=5)
    bt = build_block_tables_tensor(tables, DEVICE)
    bt[0, 1:] = 10_000  # garbage physical ids beyond the pool: must not be dereferenced
    sl = torch.tensor([5, 70], dtype=torch.int32, device=DEVICE)
    out = paged_attention_decode_triton(
        qs.to(DEVICE).contiguous(), cache.k_cache[0], cache.v_cache[0], bt, sl
    )
    torch.testing.assert_close(out.cpu(), _reference(qs, ks, vs), rtol=1e-4, atol=1e-4)


def test_rejects_cpu_tensors():
    config = _config()
    cache = PagedKVCache(config, 4, 16)
    q = torch.randn(1, config.num_heads, config.head_dim)
    with pytest.raises(RuntimeError):
        paged_attention_decode_triton(
            q,
            cache.k_cache[0],
            cache.v_cache[0],
            torch.zeros(1, 1, dtype=torch.int32),
            torch.tensor([1], dtype=torch.int32),
        )
