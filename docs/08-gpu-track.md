# GPU Track (optional) — Paged Attention in Triton + a Roofline Writeup

This track is optional and separate from the six milestones. Everything in
Milestones 1–6 runs on a CPU; nothing here is required to finish the course. Do it
after checkpoint 4.6, because the kernel you will write *is* checkpoint 4.6 moved
onto a GPU.

# Goal

1. Port `paged_attention_decode` (the online-softmax loop over blocks) to a Triton
   kernel that runs one program per (sequence, head) and reads blocks straight from
   the KV pool.
2. Measure it against the gather-based path and the PyTorch blockwise loop, and write
   a short roofline analysis explaining what you see.

# Why This Exists

Checkpoint 4.6 showed that the gather copy is unnecessary. But a Python `for` loop
over blocks is not fast either — on a GPU it launches dozens of tiny kernels per
sequence per layer and leaves the hardware idle between them. The real kernel does
the same loop *inside one launch*, in parallel across all sequences and heads, with
every block read exactly once from memory. This track is where "I understand the
data structure" becomes "I understand why the kernel is shaped the way it is".

It is also the first time in the project that hardware matters. Decode attention
reads a lot of memory (the whole KV history) and does almost no arithmetic per byte,
so it is **memory-bandwidth bound**. The roofline writeup makes you measure that
rather than take it on faith.

# Mental Model

```text
grid = (batch, num_heads)              one program per (sequence b, query head h)

program (b, h):
    kv_head = h // NUM_QUERIES_PER_KV
    q       = q[b, h, :]                                       [HEAD_DIM]      registers
    m, l    = -inf, 0                                          scalars         registers
    acc     = zeros[HEAD_DIM]                                                  registers

    for i in range(ceil(seq_lens[b] / BLOCK_SIZE)):
        physical = block_tables[b, i]                          one int load
        K = k_cache[physical, :, kv_head, :]                   [BLOCK_SIZE, HEAD_DIM]  masked tile load
        V = v_cache[physical, :, kv_head, :]                   [BLOCK_SIZE, HEAD_DIM]  masked tile load
        valid = slot < seq_lens[b] - i * BLOCK_SIZE            partial last block
        s = sum(K * q, axis=1) * sm_scale                      [BLOCK_SIZE]; -inf where not valid
        m_new = max(m, max(s)); scale = exp(m - m_new)
        l   = l * scale + sum(exp(s - m_new))
        acc = acc * scale + sum(exp(s - m_new)[:, None] * V, axis=0)
        m = m_new

    out[b, h, :] = acc / l
```

Compare with the PyTorch version in `attention.py`: the loop body is identical. What
changed is *who runs it* (thousands of programs at once) and *how memory is addressed*
(pointer + offsets × strides, with a mask for the partial block, instead of tensor
indexing).

## Pointer arithmetic

Triton has no tensors, only pointers. A `[BLOCK_SIZE, HEAD_DIM]` tile of K for one
physical block and one KV head is at

```text
k_cache_ptr
  + physical  * stride_k_block
  + slot      * stride_k_slot        slot     = tl.arange(0, BLOCK_SIZE)[:, None]
  + kv_head   * stride_k_head
  + d                                d        = tl.arange(0, HEAD_DIM)[None, :]
```

The wrapper passes all strides in elements; `head_dim` is contiguous so its stride is 1.

# Starting State

- `tiny_vllm/kernels/paged_attention_triton.py` has the wrapper (argument checks,
  strides, grid, launch), a helper to pad block tables into a tensor, and an empty
  `@triton.jit` kernel with the full argument list. It imports cleanly without Triton.
- `tests/gpu/test_triton_paged_attention.py` compares the kernel with CPU attention
  and with your `paged_attention_decode`, in float32 and float16, across block sizes
  and GQA ratios, with NaN-poisoned unused slots and garbage padding in the block table.
- `benchmarks/benchmark_attention_backends.py` times gather / blockwise / triton /
  contiguous and prints achieved GB/s.

# Setup

## Google Colab (free T4 works)

```python
!git clone <your fork> tiny-vllm
%cd tiny-vllm
!pip install -q -e ".[dev,gpu]"       # Triton ships with the Linux CUDA build of torch
!nvidia-smi --query-gpu=name,memory.total --format=csv
!pytest tests/gpu -v
```

Colab's default runtime already has a CUDA build of PyTorch with Triton; the `gpu`
extra only pins a minimum version. If `import triton` fails, restart the runtime after
the install. A T4 (320 GB/s) is enough for correctness and for a clear roofline; an
L4 or A100 gives cleaner numbers.

## Local CUDA machine

```bash
pip install -e ".[dev,gpu]"
pytest tests/gpu -v
```

On macOS or any machine without CUDA the GPU tests skip and the benchmark runs the
CPU backends only.

# Your Tasks

`tiny_vllm/kernels/paged_attention_triton.py`

- Implement the body of `_paged_attention_decode_kernel` following the mental model.
- Set `KERNEL_IMPLEMENTED = True`.

Then run `pytest tests/gpu -v` until green, and `python benchmarks/benchmark_attention_backends.py --device cuda --dtype float16 --batch 32 --seq-lens 128 512 2048 8192 --peak-gbps <your GPU>`.

`docs/roofline-writeup.md` (you create it; 1–2 pages)

See "Writeup" below.

# Checkpoints

```text
G.1 kernel matches CPU attention in float32            pytest tests/gpu -k "reference and float32"
G.2 float16, all block sizes, all GQA ratios           pytest tests/gpu -k "float16 or block_sizes or grouped"
G.3 partial blocks, padded tables, huge scores         pytest tests/gpu -k "stable or padded"
G.4 agrees with your 4.6 implementation                pytest tests/gpu -k cpu_blockwise
G.5 roofline writeup                                   (reviewed, not tested)
```

# Required Invariants

1. Bit-for-bit the same algorithm as 4.6: same recurrence, same result to rounding.
2. Slots at or beyond `seq_lens[b]` and padded block-table entries are never dereferenced
   (the tests poison them; out-of-bounds reads on a GPU may *not* crash, so the NaN
   poison is your only signal).
3. `m`, `l`, `acc` are float32 even when the cache is float16.
4. One launch per decode step for the whole batch and all heads.

# Writeup

Title it `docs/roofline-writeup.md`. Required contents:

1. **Setup.** GPU model, peak memory bandwidth (from the vendor spec sheet; cite it),
   dtype, batch, head configuration, block size.
2. **Traffic model.** Derive the bytes a decode step must read:
   `batch × seq_len × num_kv_heads × head_dim × 2 × bytes_per_element` — and state
   what it ignores (Q, output, block table, and any *re*-reads).
3. **Measurements.** The benchmark table for at least four sequence lengths and all
   four backends. Convert to achieved GB/s and % of peak.
4. **The roofline.** One plot: x = seq_len, y = achieved GB/s, one line per backend,
   horizontal line at peak. State which backends are memory-bound, which are launch-
   or Python-overhead-bound, and how you can tell from the *shape* of the curves
   (hint: overhead-bound curves rise with seq_len; bandwidth-bound ones plateau).
5. **Arithmetic intensity.** FLOPs per byte for decode attention. Show that it is far
   below any modern GPU's ridge point, i.e. why nobody optimises decode attention for
   compute.
6. **Two experiments of your choice**, e.g. block_size 4 vs 16 vs 64; batch 1 vs 32;
   float32 vs float16; `num_kv_heads` 1 vs 8 (why GQA is a *bandwidth* optimisation).
7. **Gap analysis.** Your kernel vs peak. List two concrete reasons it does not reach
   peak (e.g. one program per head re-reads the same KV block for every query head in
   a GQA group; no software pipelining of loads; the `[BLOCK_SIZE, HEAD_DIM]` tile
   shape) and what a production kernel does about each.

Reference peak bandwidths (check the spec sheet for your exact card): T4 320 GB/s,
L4 300 GB/s, A10G 600 GB/s, A100-40GB 1555 GB/s, A100-80GB 2039 GB/s, H100-SXM 3350 GB/s.

# Expected Observations

- `gather` and `blockwise` (PyTorch) reach a few percent of peak at best: their time is
  dominated by kernel launches and Python, and it grows roughly linearly in the
  number of blocks.
- `triton` plateaus at a substantial fraction of peak for long sequences and large
  batches; at short sequences launch latency dominates and GB/s looks low.
- `contiguous` (SDPA on already-gathered K/V) is an upper bound on what paging can
  cost you; a good paged kernel comes within a small factor of it.
- Halving `num_kv_heads` roughly halves bytes and, for the bandwidth-bound kernel,
  roughly halves time. That is GQA's entire value proposition for decode.

# Hints

- Start by making `test_matches_reference_scrambled_blocks[float32]` pass with
  `BLOCK_SIZE=16`; add the partial-block mask second; only then worry about float16
  (cast tiles to `tl.float32` after loading).
- Use `tl.load(..., mask=valid[:, None], other=0.0)` for K/V and set masked scores to
  `-inf` *before* the max, not after.
- `tl.sum(K * q[None, :], axis=1)` is the whole score computation for one block.
- If everything is `nan`, you divided by `l = 0` — the first block's scale is
  `exp(-inf - m_new)`; make sure `m` starts at `-inf` as a float32 and `l` at `0.0`.
- If results are right for `seq_len % BLOCK_SIZE == 0` and wrong otherwise, the
  partial-block mask is off by one.
- `TRITON_INTERPRET=1 pytest tests/gpu` runs kernels in the Python interpreter for
  debugging with `print` (slow, but you can see values).

# Questions You Should Be Able to Answer

1. Why is decode attention memory-bound, and what number tells you so?
2. What does the Python blockwise loop and the Triton kernel have in common, and what is the one thing that differs?
3. Why is one program per (sequence, head) a natural grid, and what does it waste under GQA?
4. Why does a partially filled last block need a mask rather than a shorter loop?
5. Why can the same kernel serve every layer, and what changes between layers?
6. If you doubled `block_size`, what happens to (a) the number of loop iterations, (b) wasted memory in partial blocks, (c) the tile size in registers?

# Optional Stretch Goal

Restructure the grid to one program per (sequence, **KV head**) that handles all
`NUM_QUERIES_PER_KV` query heads at once, so each KV block is read once per group
instead of once per query head. Measure the bandwidth gain. This is the trick that
makes GQA pay off inside the kernel, not just in the traffic model.
