# Milestone 4 — Paged Attention

# Goal

Perform attention correctly when the historical K/V of a sequence lives in
physically scattered blocks, and generate text with TinyLlama over the paged pool.

```text
contiguous KV attention  ≈  paged KV attention     with block_table = [7, 2, 11]
```

# Why This Exists

Milestone 3 scattered K/V across blocks. Attention, however, needs the keys of a
sequence in **logical order**: key 0, key 1, key 2, … The question this milestone
answers is

> How can Q attend to historical K/V if the KV cache is scattered across physical blocks?

The answer is the block table. Attention never cares about physical addresses; it
only needs, for each logical position, the vector stored there. The block table is
the only thing that knows where that is.

When this milestone is done you have built the whole "TinyPagedAttention" core —
before any scheduler or batching exists.

# Mental Model

```text
Request A logical KV:

K0 K1 K2 K3 | K4 K5 K6 K7 | K8 K9 ..
      │              │           │
      ▼              ▼           ▼
 physical 7      physical 2   physical 11        block_table = [7, 2, 11], seq_len = 10
```

Our implementation reproduces the semantics in the most explicit possible way:

```text
physical KV blocks  k_cache[layer]            [num_blocks, block_size, num_kv_heads, head_dim]
        │
        │  index with block_table             [num_logical_blocks, block_size, ...]
        ▼
gathered blocks
        │
        │  merge the two leading dims          [num_logical_blocks * block_size, ...]
        ▼
        │  cut to seq_len                      [seq_len, ...]   (last block is partial)
        ▼
logical contiguous K, V
        │
        ▼
causal_attention(q, K, V)                      the exact function from Milestone 1
```

The partial last block matters: with `seq_len = 10` and `block_size = 4`, physical
block 11 contains two valid slots and two slots of garbage (uninitialised, or stale
K/V from a previous owner of that block). Reading them would silently corrupt the
softmax.

## The gather is a crutch (checkpoint 4.6 removes it)

Real vLLM does **not** materialise the gathered tensor. Its attention kernels
(PagedAttention v1/v2, FlashAttention / FlashInfer with paged KV) take the block
table as an argument and fetch each block straight from the pool while computing
the softmax, so no copy is made and the whole thing runs in one GPU kernel. Our
gather-then-attend version (4.1–4.5) costs one extra copy of the sequence's K/V per
layer per step. It is *semantically identical*, which is what 4.1–4.5 are about.
Checkpoint 4.6 then asks: why copy at all?

Attention is a weighted average of V with softmax weights. You could visit each
block where it lives, score its keys, and accumulate — except softmax needs the
**global** maximum before it can normalise anything, and block 7 does not know what
scores block 11 will produce. The fix is the **online softmax**: keep a running max
and rescale what you have accumulated whenever the max moves.

For one query `q` (decode), per head, processing blocks in logical order:

```text
m   = -inf            running max of scores
l   = 0               running sum of exp(score - m)
acc = 0               running sum of exp(score - m) * v

for each logical block:
    s      = q · K_blockᵀ / sqrt(head_dim)             scores of the block's valid slots
    m_new  = max(m, max(s))
    scale  = exp(m - m_new)                            how much the OLD max was too small
    l      = l   * scale + sum(exp(s - m_new))
    acc    = acc * scale + exp(s - m_new) · V_block
    m      = m_new

out = acc / l
```

That is the entire kernel, minus the hardware. `exp(m − m_new)` is the single line
that makes it exact: it retroactively re-normalises everything you accumulated
under the old max, so the result equals a one-shot softmax to rounding error. It is
also what keeps `exp` from overflowing when scores are large.

Why decode-only: with one query the whole history is visible and there is no
causal mask inside the loop. vLLM's original `paged_attention_v1` kernel had the same
restriction and prefill used a separate contiguous-attention kernel — the split you
end up with here (`paged_attention` for prefill, `paged_attention_decode` for decode)
is that historical design. FlashAttention generalises the same recurrence to many
queries with a causal mask per tile; the GPU track (`08-gpu-track.md`) ports the
decode version to Triton.

## Plumbing

`Attention.forward` receives a `PagedKVCache` and an `AttentionMetadata`:

```text
AttentionMetadata
  slot_mapping   [batch * seq_len]   where the NEW tokens' K/V are written  (M3)
  block_tables   list per sequence    where the WHOLE history is read from   (M4)
  seq_lens       list per sequence    how many logical positions are valid
```

`attend_paged` = write new K/V (M3) → per sequence `paged_attention` (this
milestone). In this milestone the batch is always 1; Milestone 5 loops over it.

# Starting State

- `PagedKVCache`, `BlockPool` and the slot-mapping helpers work (M3 green).
- `gather_kv`, `paged_attention` and `generate_paged` raise `NotImplementedError`.
- `AttentionMetadata` validates its own consistency.

# Your Tasks

`tiny_vllm/attention.py`

- `gather_kv(cache, block_table, seq_len)` — `[num_blocks, block_size, H, D]` → `[seq_len, H, D]`.
- `paged_attention(q, k_cache, v_cache, block_table, seq_len)` — gather K and V, then `causal_attention`.
- `Attention.attend_paged(q, k, v, kv_cache, attn_metadata)` — write via slot mapping,
  then attend for the (single) sequence.
- `paged_attention_decode(q, k_cache, v_cache, block_table, seq_len)` — the online-softmax
  loop over blocks for one query; then route decode steps (`seq_len == 1` per sequence)
  in `attend_paged` through it. Prefill keeps using `paged_attention`.

`tiny_vllm/generation.py`

- `generate_paged(model, kv_cache, block_pool, prompt_tokens, max_new_tokens, eos_token_id)`
  — allocate blocks, build `AttentionMetadata` for prefill and for every decode
  step, free all blocks at the end (also on error).

# Checkpoints

```text
4.1 Resolve logical token → physical KV     pytest -m m4 -k GatherKV
4.2 Read K/V through the block table        pytest -m m4 -k GatherKV
4.3 Compute attention                       pytest -m m4 -k PagedAttention
4.4 Handle partial final block              pytest -m m4 -k PartialFinalBlock
4.5 Integrate with TinyLlama                pytest -m m4 -k "ModelWithPagedKV or GeneratePaged or integration"
4.6 Blockwise decode attention              pytest -m m4 -k PagedAttentionDecode
```

Do 4.6 after 4.5: the 4.5 tests pass with the gather path, and the last 4.6 test then
checks that a decode step through the model no longer gathers.

# Required Invariants

1. `paged_attention(q, ..., block_table=[7, 2, 11], seq_len)` equals `causal_attention`
   over the logically ordered K/V, for prefill (`num_queries == seq_len`), decode
   (`num_queries == 1`) and anything between.
2. The result is independent of *which* physical blocks are used, and dependent on the
   *order* of the block table.
3. Slots at or beyond `seq_len` never influence the output (the tests poison them with NaN).
4. Extra blocks in the table beyond what `seq_len` needs are ignored.
5. Every layer reads and writes its own slice of the pool: after a paged prefill the
   gathered K/V of layer `l` equals the contiguous cache of layer `l`.
6. `generate_paged` produces exactly the tokens of `generate_naive` and returns every
   block to the pool.
7. `paged_attention_decode` equals `paged_attention` for one query to rounding error,
   stays finite when scores differ by hundreds between blocks, and never calls
   `gather_kv` / `paged_attention` (the tests replace both with functions that raise).

# Tests

- `TestGatherKV` and `TestPartialFinalBlock` place logically ordered K/V into the pool
  through your own M3 write path, then read it back through the block table.
- `TestPagedAttention` is the critical test of the plan: reference vs paged with
  `block_table = [7, 2, 11]`.
- `TestModelWithPagedKV` runs the whole model in paged mode and compares logits with
  the cache-free forward, and per-layer pool contents with `ContiguousKVCache`.
- `TestGeneratePaged` runs generation in a *fragmented* pool (only even blocks free).

# Experiment

```bash
python benchmarks/run_all_experiments.py --only C
```

fills a pool through a scrambled block table on the real model and reports the
maximum difference between contiguous and paged logits, plus the generated text.

# Expected Observations

- Differences between contiguous and paged logits are at floating-point noise level
  (`≈1e-6`); the tokens are identical.
- Swapping two entries of the block table changes the output — logical order is
  what attention sees.
- The paged forward is slightly slower than the contiguous one because of the
  gather copy. That is the price of the semantic implementation, not of paging.
- **`paged_attention_decode` in PyTorch is *slower* than the gather path** — on a CPU
  by roughly 5–10× for the attention call, and the engine's decode step gets ~20%
  slower once it routes through it (`benchmarks/benchmark_attention_backends.py` shows
  the per-call numbers). A Python loop that launches a handful of small tensor ops per
  block per layer per sequence is dominated by interpreter and dispatch overhead. This
  is the point: 4.6 gives you the *algorithm* a kernel runs; the speed only appears when
  that loop is fused into one launch (the optional GPU track, `08-gpu-track.md`). Keep
  the routing anyway — it is what real engines do, and the tests rely on it.

# Hints

- Advanced indexing with a list or tensor of block ids gathers whole blocks in one
  operation and preserves the order you give.
- `flatten(0, 1)` merges the leading two dimensions.
- `paged_attention` works on one sequence without a batch dimension; add one
  (`[None]`) before calling `causal_attention` and remove it afterwards.
- In `generate_paged`, the sequence length seen by attention for a decode step at
  position `p` is `p + 1`, and the block table must cover `p + 1` tokens *before* the
  forward pass.
- `try/finally` is the simplest way to guarantee blocks are freed.
- For 4.6, work per head with `m`, `l` of shape `[num_heads]` and `acc` of shape
  `[num_heads, head_dim]`; expand the block's K/V heads for GQA with
  `repeat_interleave` just as in `causal_attention`. Start with `m = -inf`; `exp(-inf − x)`
  is 0, so the first block's rescale is harmless.
- The number of valid slots in logical block `i` is `min(block_size, seq_len − i·block_size)`.
- If `test_numerically_stable_across_blocks` fails with NaN or inf, you are exponentiating
  before subtracting the *new* max, or forgetting to rescale `acc`.

# Questions You Should Be Able to Answer

1. Why does attention care about logical token order but not physical memory order?
2. How does the block table recover logical ordering?
3. Why is our gather-based implementation slower than an optimised paged-attention kernel, and what does 4.6 remove?
4. What part of our implementation represents the semantics of PagedAttention?
5. Why can the same block table be used across all Transformer layers?
6. What goes wrong if `gather_kv` ignores `seq_len` and returns all slots of the last block?
7. In the online softmax, why must `acc` and `l` be multiplied by `exp(m_old − m_new)` and why is the final result nevertheless exact?
8. Why does the decode kernel not need a causal mask, and what would change for a prefill (many-query) version?

# Optional Stretch Goal

Generalise `paged_attention_decode` to `num_queries > 1` with a causal mask inside
the block loop (the FlashAttention structure), and verify it against `paged_attention`
for prefill. Then read `08-gpu-track.md` and port the decode version to Triton.
