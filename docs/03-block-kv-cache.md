# Milestone 3 — Block-Based KV Cache

# Goal

Replace the per-request contiguous cache with **one global pool of fixed-size
physical blocks** shared by every request, plus the two small data structures that
make it usable: the **block table** and the **slot mapping**.

# Why This Exists

`ContiguousKVCache` reserves `max_seq_len` positions per request up front. In a
server that is fatal:

- you do not know how long a request will be, so you reserve the maximum → most of
  the memory is reserved but never used (internal fragmentation);
- requests of different lengths come and go, leaving holes that are too small to be
  reused (external fragmentation);
- a request cannot grow beyond its reservation.

This is exactly the problem operating systems solved with virtual memory: cut memory
into fixed-size pages, hand them out on demand, and keep a per-process table that
translates logical addresses to physical ones. vLLM's insight is to apply the same
idea to the KV cache. This milestone builds the "memory management unit".

# Mental Model

Instead of

```text
Request A:  [K0 K1 K2 K3 K4 K5 K6 K7 ...]        one big tensor per request
```

use

```text
Global KV Pool (per layer)

Block 0 [ ][ ][ ][ ]
Block 1 [ ][ ][ ][ ]
Block 2 [ ][ ][ ][ ]
Block 3 [ ][ ][ ][ ]
...                                 block_size = 4 token slots per block
```

Four concepts, deliberately kept in four separate places:

```text
┌───────────────┐   owns the tensors
│  PagedKVCache │   k_cache, v_cache: [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
└───────────────┘

┌───────────────┐   knows which block ids are free
│   BlockPool   │   allocate() -> id,  free(id)
└───────────────┘

┌───────────────┐   per request: logical block i -> physical block id
│  Block Table  │   block_table = [7, 2, 11]
└───────────────┘

┌───────────────┐   per forward pass: where does each NEW token's K/V go
│ Slot Mapping  │   slot = physical_block * block_size + offset
└───────────────┘
```

## Block table

```text
Request A logical KV:   K0 K1 K2 K3 | K4 K5 K6 K7 | K8 K9
logical block:                 0            1          2
block_table = [7, 2, 11]       ▼            ▼          ▼
physical block:                7            2         11
```

Blocks belonging to one request do **not** have to be adjacent. A request grows by
appending one more physical block to its table whenever it crosses a block boundary.

## Slot mapping

Given a token position, which physical slot holds its K/V?

```text
block_size = 4
block_table = [7, 2]
token position = 5

logical_block  = 5 // 4 = 1
offset         = 5 %  4 = 1
physical_block = block_table[1] = 2

slot = physical_block * block_size + offset = 2 * 4 + 1 = 9
```

A slot is a flat index into `k_cache[layer].reshape(num_blocks * block_size, ...)`.
The block table answers "where is my history"; the slot mapping answers "where do I
put this new token". Keep them distinct — attention (M4) uses the table, the write
path (this milestone) uses the mapping.

## Tensor layout

```text
k_cache[layer]                      whole pool of one layer
k_cache[layer, block]               one physical block   [block_size, num_kv_heads, head_dim]
k_cache[layer, block, offset]       one token slot       [num_kv_heads, head_dim]
```

Every layer has the same number of blocks and a request's block table is shared by
all layers: block 7 of layer 0 and block 7 of layer 29 belong to the same request.

# Starting State

- `PagedKVCache.__init__` stores the sizes; tensors are not allocated.
- `BlockPool` has the interface and the error type (`OutOfBlocksError`).
- `AttentionMetadata` (slot mapping + block tables + sequence lengths) is provided;
  you will *build* it in M4.
- `visualize.format_kv_pool` draws the pool from block tables alone.

# Your Tasks

`tiny_vllm/kv_cache.py`

- `PagedKVCache.__init__` — allocate `k_cache`, `v_cache`.
- `PagedKVCache.write(layer_idx, k, v, slot_mapping)` — scatter new K/V into slots.
- `blocks_needed(num_tokens, block_size)`
- `logical_to_physical(block_table, token_position, block_size) -> (physical_block, offset)`
- `ensure_block_capacity(block_table, block_pool, num_tokens, block_size)` — grow a table.
- `compute_slot_mapping(block_table, positions, block_size)`

`tiny_vllm/block_pool.py`

- `BlockPool.__init__`, `num_free_blocks`, `allocate`, `free`.

`tiny_vllm/attention.py`

- The **write half** of `Attention.attend_paged`: `kv_cache.write(self.layer_idx, ...)`
  with the flattened new K/V and `attn_metadata.slot_mapping`. The attention half
  comes in M4; you may leave a `raise NotImplementedError("Milestone 4")` after the
  write for now.

# Checkpoints

```text
3.1 Global KV pool                pytest -m m3 -k PagedKVCacheStorage
3.2 Block allocation              pytest -m m3 -k Allocate
3.3 Block free + reuse            pytest -m m3 -k FreeAndReuse
3.4 Block table                   pytest -m m3 -k BlockTableHelpers
3.5 Slot mapping                  pytest -m m3 -k SlotMapping
3.6 Write K/V into blocks         pytest -m m3 -k PagedKVCacheWrite
    fragmentation + integration   pytest -m m3 -k "fragmented or integration"
```

# Required Invariants

1. A physical block never belongs to two live requests at the same time.
2. A freed block becomes allocatable again; a block is never handed out twice without being freed in between.
3. Blocks of one request need not be contiguous (the fragmentation test allocates `{1, 3, 5, 7}`).
4. Exhaustion is explicit: `allocate()` raises `OutOfBlocksError`; nothing is silently overwritten.
5. Crossing a logical block boundary produces the correct physical address (`position 4` with `[7, 2]` → slot 8).
6. Layer `l` writes only into `k_cache[l]` / `v_cache[l]`; other layers and other slots are untouched.
7. Double free and freeing an unknown id raise `ValueError`.

# Tests

- `test_blocks.py` — `BlockPool` behaviour, a randomised allocate/free stress test,
  the fragmentation scenario from the plan, and the visualizer.
- `test_kv_cache.py` — tensor shapes, the helpers (with the plan's `[7, 2]`, position 5 → 9
  example hard-coded), and NaN-poisoned pools that prove writes hit only the
  addressed slots.
- `test_m3_integration_two_requests_share_a_fragmented_pool` — two requests, a
  fragmented pool, isolation, free, reuse.

# Experiment

```bash
python examples/visualize_blocks.py
```

prints a pool picture like

```text
Request A
block_table = [7, 2]

Request B
block_table = [5]

Physical KV Pool (num_blocks=8, block_size=4)

0 FREE
1 FREE
2 A [tokens 4-7]
3 FREE
4 FREE
5 B [tokens 0-3]
6 FREE
7 A [tokens 0-3]

free = [0, 1, 3, 4, 6]
```

Try changing the block tables, sizes and sequence lengths in the script.

# Expected Observations

- The pool's memory is fixed at start-up: `num_layers × num_blocks × block_size × num_kv_heads × head_dim × 2 (K and V) × bytes`.
  For SmolLM2 with 256 blocks of 16 tokens in float32 this is about 190 MB — for
  4096 tokens shared by *any* number of requests.
- The waste per request is at most one partially filled block (`block_size − 1` slots).
- Nothing about the *contents* of the pool is "a request"; ownership exists only in
  block tables and the free list.

# Hints

- A free list can be a `list`, `set`, or `deque`; the tests only check behaviour.
- A slot is a flat index into the `num_blocks × block_size` token slots of one
  layer. Viewing a layer's pool with those two dimensions merged makes the slot
  mapping directly usable as an index — make sure you write through a *view* of
  the pool, not a copy.
- Keep a `set` of allocated ids if you want `free()` to detect double frees cheaply.
- `ensure_block_capacity` is called with the *total* number of tokens the request
  will have after this step; compare with `len(block_table)`.
- `compute_slot_mapping` should be vectorised over `positions` (integer tensor
  arithmetic works fine) or a simple loop — both are acceptable here.

# Questions You Should Be Able to Answer

1. Who owns the actual K/V tensor memory?
2. What exactly is a physical block?
3. Why do blocks assigned to one request not need to be contiguous?
4. What information is stored in a block table?
5. What is the difference between a block table and a slot mapping?
6. What happens to physical blocks when a request finishes?

# Optional Stretch Goal

Compute, for a pool of `N` blocks with `block_size = B`, the worst-case fraction of
memory wasted to partially filled blocks when `R` requests are live. Then argue why
vLLM chose 16 as its default block size rather than 1 or 256.
