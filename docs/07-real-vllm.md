# How Real vLLM Goes Beyond tiny-vllm

This document is explanatory only. It maps the concepts you implemented onto the
corresponding parts of vLLM (the V1 engine, `vllm/v1/`) and lists what vLLM adds
on top.

**Snapshot.** File paths and class names below were inspected against
[vllm-project/vllm](https://github.com/vllm-project/vllm) `main` at commit
[`004e37ece2fd3f49b5ba471e016668f23d0e7c8b`](https://github.com/vllm-project/vllm/commit/004e37ece2fd3f49b5ba471e016668f23d0e7c8b)
(2026-09-28). The latest tagged release at that time was
[`v0.30.0`](https://github.com/vllm-project/vllm/releases/tag/v0.30.0). Internals
move fast, so treat the paths as a starting point for reading, not as a
specification. If a path no longer exists, start from that commit rather than
from current `main`.

# Concept map

```text
tiny-vllm                          real vLLM (V1)

TinyLlama                          vllm/model_executor/models/llama.py
  layers.RMSNorm                     RMSNorm (fused CUDA kernel available)
  layers.apply_rope                  get_rope(...) rotary embedding layers
  attention.Attention                LlamaAttention with a fused QKVParallelLinear and an
                                     Attention layer that delegates to a pluggable backend

Request / RequestStatus            vllm/v1/request.py  (Request, RequestStatus with many more states:
                                     WAITING_FOR_*, PREEMPTED, FINISHED_STOPPED/LENGTH/ABORTED, ...)

Scheduler                          vllm/v1/core/sched/scheduler.py (Scheduler.schedule,
                                     update_from_output, add_request, finish_requests, _preempt_request)

PagedKVCache                       KV cache tensors allocated by the worker/model runner
                                     (vllm/v1/worker/gpu_model_runner.py) according to a
                                     KVCacheConfig / KVCacheSpec (vllm/v1/kv_cache_interface.py)

BlockPool                          vllm/v1/core/block_pool.py  (BlockPool.get_new_blocks, free_blocks,
                                     get_num_free_blocks; blocks are KVCacheBlock objects with
                                     ref counts and hashes, kept in a FreeKVCacheBlockQueue)

ensure_block_capacity              vllm/v1/core/kv_cache_manager.py  (KVCacheManager.allocate_slots,
  + the per-request block table       free, get_block_ids) on the scheduler side, and
                                     vllm/v1/worker/block_table.py (BlockTable.append_row / add_row)
                                     on the worker side

compute_slot_mapping               BlockTable.compute_slot_mapping (vllm/v1/worker/block_table.py)

AttentionMetadata                  per-backend metadata, e.g. FlashAttentionMetadata in
  slot_mapping / block_tables /      vllm/v1/attention/backends/flash_attn.py with fields
  seq_lens                           slot_mapping, block_table, seq_lens, query_start_loc, max_query_len, ...

paged_attention (gather + SDPA)    attention backends in vllm/v1/attention/backends/
                                     (flash_attn.py, flashinfer.py, triton_attn.py, cpu_attn.py, ...)
                                     whose kernels read blocks directly through the block table

LLMEngine.step                     vllm/v1/engine/core.py (EngineCore.step) driven by
                                     vllm/v1/engine/llm_engine.py; the scheduler and the model
                                     executor run in separate processes
```

Do not read this table as source-level equivalence. tiny-vllm merges things that
vLLM separates (scheduler-side block accounting vs. worker-side block tables), and
vLLM has abstractions tiny-vllm has no reason to have (KV cache *groups* for models
with several attention types, connectors for remote KV, and so on).

# What is the same

The four ideas you implemented are literally the ideas vLLM is built on:

1. **The KV cache is a pool of fixed-size blocks** shared by all requests, allocated
   at start-up, with per-layer tensors of shape roughly
   `[num_blocks, block_size, num_kv_heads, head_dim]` for K and V (vLLM lets the
   backend choose the exact layout; see `vllm/v1/kv_cache_layout.py`).
2. **A request owns a block table**, not memory. Growth is appending a block id;
   completion is returning the ids.
3. **New K/V are written through a slot mapping** computed from the block table
   (`slot = block_id * block_size + offset`, exactly as in Milestone 3).
4. **Attention reads history through the block table**, so the physical layout is
   irrelevant to the maths. The only difference is *how* it reads: fused kernels
   instead of gather-then-attend.

Continuous batching — re-deciding the batch every step, admitting new requests as
others finish — is also the same loop as your `LLMEngine.step`, just with many more
policies attached.

# What vLLM adds

## Optimised kernels

- Attention kernels (FlashAttention, FlashInfer, Triton, CPU/SDPA backends) take the
  block table and iterate over blocks inside the kernel; no gathered copy exists.
  This is why our `paged_attention` is "semantically right, physically slow".
- Fused ops: RMSNorm + residual, RoPE, SiLU-and-mul, quantised GEMMs.
- Packed ("varlen") attention: prefill and decode tokens of *all* requests are
  concatenated into one `[num_tokens, hidden]` tensor described by
  `query_start_loc`; one kernel call handles every sequence. tiny-vllm instead uses
  `[batch, seq_len]` with prefill one request at a time.
- CUDA graphs / `torch.compile` to remove launch overhead for decode steps.

## Scheduling

Our scheduler admits FIFO with a worst-case block budget and never runs out of
memory. vLLM's `Scheduler.schedule` instead:

- admits optimistically and **preempts** a running request (freeing its blocks and
  re-queuing it for recomputation) when blocks run out;
- supports **chunked prefill**: a long prompt is processed in pieces across steps so
  decode tokens of other requests are not starved by one huge prefill;
- enforces a token budget per step (`max_num_batched_tokens`) rather than only a
  sequence count;
- handles priorities, multiple KV-cache groups, encoder inputs, structured-output
  grammars, speculative tokens and asynchronous scheduling.

## Prefix caching

Because blocks are the unit of storage, two requests with the same prompt prefix
can *share* full blocks. vLLM hashes each full block by its token content
(`BlockHash`, `BlockPool.cache_full_blocks`), keeps reference counts on
`KVCacheBlock`, and `KVCacheManager.get_computed_blocks` skips recomputing
prefixes that are already cached. This is why vLLM's `BlockPool` is far larger than
ours — most of it is hashing and eviction (LRU via the free queue), not allocation.

## Distributed execution

Tensor parallelism (`QKVParallelLinear`, `RowParallelLinear` in `llama.py` split the
projections across GPUs), pipeline parallelism, expert parallelism for MoE, and
multi-node serving. The block table concept is unchanged; every rank holds its shard
of every block.

## Quantisation

FP8 / INT8 / INT4 weights and FP8 KV cache (`csrc/attention/dtype_fp8.cuh`), which
directly shrink the per-token cost you computed in Milestone 2.

## Speculative decoding

Draft models or n-gram proposers produce several candidate tokens per step; the
target model verifies them in one forward pass. This requires slot mappings for
multiple new positions per request per step — a generalisation of your decode path.

## Production serving

An OpenAI-compatible HTTP server, streaming, request abort, metrics, multi-process
engine core, CPU/GPU offload of KV blocks, disaggregated prefill/decode via KV
connectors, and support for hundreds of architectures including multimodal models.

# Where tiny-vllm deliberately simplified

| Topic | tiny-vllm | vLLM |
|-------|-----------|------|
| Token layout | `[batch, seq_len]`, prefill one request per pass | packed `[num_tokens]` with `query_start_loc` |
| Admission | worst-case budget, no preemption | optimistic + preemption/recompute |
| Prefill | whole prompt in one step | chunked |
| Block metadata | plain `list[int]` | `KVCacheBlock` with ref count + hash |
| Block reuse across requests | none | prefix caching |
| Attention | gather + SDPA | fused paged kernels |
| Model | one Llama-like architecture | hundreds |
| Sampling | greedy | full sampler (temperature, top-p/k, penalties, logprobs, ...) |

# Reading guide

If you want to read vLLM after finishing this course, a reasonable order is:

1. `vllm/v1/request.py` — recognise `Request`.
2. `vllm/v1/core/block_pool.py` and `vllm/v1/core/kv_cache_utils.py` — recognise `BlockPool`, then find the hashing.
3. `vllm/v1/core/kv_cache_manager.py` — `allocate_slots` is `ensure_block_capacity` plus prefix caching.
4. `vllm/v1/core/sched/scheduler.py` — `schedule()` first; note the token budget and preemption.
5. `vllm/v1/worker/block_table.py` — the worker-side block table and slot mapping.
6. `vllm/v1/attention/backends/flash_attn.py` — find `slot_mapping`, `block_table`, `seq_lens` in the metadata and follow them into the kernel call.
7. `vllm/model_executor/models/llama.py` — compare with `tiny_vllm/model.py`.
