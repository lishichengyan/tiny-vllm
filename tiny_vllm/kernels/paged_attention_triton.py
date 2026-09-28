"""Triton paged-attention decode kernel (GPU track, docs/08-gpu-track.md).

This is ``tiny_vllm.attention.paged_attention_decode`` (checkpoint 4.6) moved onto the
GPU: the same online-softmax loop over blocks, but one *program* (a group of GPU
threads) per (sequence, query head), all running in parallel, and every block read
straight from the KV pool with pointer arithmetic.

    grid = (batch, num_heads)

    program (b, h):
        kv_head = h // NUM_QUERIES_PER_KV
        q       = load q[b, h, :]                                  [HEAD_DIM]
        m, l, acc = -inf, 0, zeros(HEAD_DIM)
        for logical block i in 0 .. ceil(seq_lens[b] / BLOCK_SIZE) - 1:
            physical = block_tables[b, i]
            K = load k_cache[physical, :, kv_head, :]              [BLOCK_SIZE, HEAD_DIM]
            V = load v_cache[physical, :, kv_head, :]              [BLOCK_SIZE, HEAD_DIM]
            mask slots >= seq_lens[b] - i * BLOCK_SIZE  (partial last block)
            s = (K · q) * sm_scale                                 [BLOCK_SIZE]
            online-softmax update of m, l, acc
        store out[b, h, :] = acc / l

The wrapper, the pointer/stride plumbing and the tests are provided. You write the
kernel body. The module imports cleanly without Triton or CUDA; the wrapper checks.

Layout requirements (asserted by the wrapper):
    q:            [batch, num_heads, head_dim]           contiguous, on CUDA
    k_cache:      [num_blocks, block_size, num_kv_heads, head_dim]   contiguous, on CUDA
    v_cache:      same shape and layout as k_cache
    block_tables: [batch, max_num_blocks] int32          padded rows are never read
    seq_lens:     [batch] int32
    head_dim and block_size must be powers of two (``tl.arange`` restriction).
"""

import math

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - CPU-only machines
    HAS_TRITON = False

# TODO(student, GPU track): set to True once the kernel body below is implemented.
KERNEL_IMPLEMENTED = False


if HAS_TRITON:

    @triton.jit
    def _paged_attention_decode_kernel(
        q_ptr,
        k_cache_ptr,
        v_cache_ptr,
        out_ptr,
        block_tables_ptr,
        seq_lens_ptr,
        stride_q_batch,
        stride_q_head,
        stride_k_block,
        stride_k_slot,
        stride_k_head,
        stride_bt_batch,
        stride_out_batch,
        stride_out_head,
        sm_scale,
        NUM_QUERIES_PER_KV: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        """One program per (sequence, query head). See the module docstring for the plan.

        Useful pieces:
            tl.program_id(0) / tl.program_id(1)         which (sequence, head) am I
            tl.arange(0, HEAD_DIM)                       offsets along head_dim
            tl.arange(0, BLOCK_SIZE)                     offsets along the slots of a block
            tl.load(ptr + offsets, mask=..., other=0.0)  masked loads for the partial last block
            tl.sum(x, axis=...)                          reductions
            tl.max(x, axis=0), tl.exp(x)                 the online-softmax update
            offsets[:, None] * stride + offsets[None, :] 2-D pointer grids for a [BLOCK_SIZE, HEAD_DIM] tile
        Keep m, l and acc in float32 regardless of the cache dtype.
        """
        # TODO(student, GPU track): implement the kernel body.
        pass


def build_block_tables_tensor(block_tables: list[list[int]], device) -> torch.Tensor:
    """Pad a list of block tables into a [batch, max_num_blocks] int32 tensor (pad value 0;
    padded entries are never dereferenced because the kernel stops at seq_len)."""
    max_len = max(1, max(len(t) for t in block_tables))
    out = torch.zeros(len(block_tables), max_len, dtype=torch.int32)
    for i, table in enumerate(block_tables):
        if table:
            out[i, : len(table)] = torch.tensor(table, dtype=torch.int32)
    return out.to(device)


def paged_attention_decode_triton(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Batched decode attention over the paged KV cache, on the GPU.

    Args:
        q:            [batch, num_heads, head_dim]
        k_cache:      [num_blocks, block_size, num_kv_heads, head_dim]  (one layer)
        v_cache:      same as k_cache
        block_tables: [batch, max_num_blocks] int32  (see ``build_block_tables_tensor``)
        seq_lens:     [batch] int32 -- total tokens per sequence incl. the query token
    Returns:
        [batch, num_heads, head_dim] in ``q.dtype``.
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is not installed; pip install -e '.[gpu]' on a CUDA machine")
    if not q.is_cuda:
        raise RuntimeError("paged_attention_decode_triton requires CUDA tensors")
    if not KERNEL_IMPLEMENTED:
        raise NotImplementedError(
            "GPU track: implement _paged_attention_decode_kernel in "
            "tiny_vllm/kernels/paged_attention_triton.py and set KERNEL_IMPLEMENTED = True"
        )

    batch, num_heads, head_dim = q.shape
    num_blocks, block_size, num_kv_heads, head_dim_k = k_cache.shape
    assert head_dim == head_dim_k, "q and cache head_dim differ"
    assert v_cache.shape == k_cache.shape
    assert q.is_contiguous() and k_cache.is_contiguous() and v_cache.is_contiguous()
    assert block_tables.dtype == torch.int32 and seq_lens.dtype == torch.int32
    assert block_tables.shape[0] == batch and seq_lens.shape[0] == batch
    assert head_dim & (head_dim - 1) == 0, "head_dim must be a power of two"
    assert block_size & (block_size - 1) == 0, "block_size must be a power of two"
    assert num_heads % num_kv_heads == 0

    out = torch.empty_like(q)
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    grid = (batch, num_heads)
    _paged_attention_decode_kernel[grid](
        q,
        k_cache,
        v_cache,
        out,
        block_tables,
        seq_lens,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        block_tables.stride(0),
        out.stride(0),
        out.stride(1),
        sm_scale,
        NUM_QUERIES_PER_KV=num_heads // num_kv_heads,
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
    )
    return out
