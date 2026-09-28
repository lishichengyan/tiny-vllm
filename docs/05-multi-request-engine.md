# Milestone 5 — Multi-Request Engine

# Goal

Turn the single-sequence code into a minimal engine that serves several requests
at once: a `Request` object, a `Scheduler`, and an `LLMEngine.step()` that runs
one batched forward pass for every running request.

# Why This Exists

A server never has one request. With the pieces so far you could serve requests one
after another, but a decode step for a single sequence uses the hardware badly:
every step reads all the model weights to produce one token. Reading the same
weights once to produce a token for *each* of N sequences costs almost the same
time — that is the entire reason batching exists.

Batching sequences of different lengths is where the block-based cache pays off:
each request has its own block table and sequence length, so the shared pool holds
all of them without padding or copying.

# Mental Model

```text
                        add_request()
                             │
                             ▼
              ┌──────────── Scheduler ────────────┐
              │  waiting: [C, D]   running: [A, B] │
              └──────────────────┬─────────────────┘
                                 │ schedule()
                     ┌───────────┴───────────┐
                     ▼                       ▼
            prefill batch [C]         decode batch [A, B]
            (newly admitted)          (one new token each)
                     │                       │
                     ▼                       ▼
            _prefill(C)               _decode([A, B])
              allocate blocks           ensure block for new position
              slot mapping              slot mapping (one slot per request)
              forward [1, S]            forward [2, 1]
              sample 1 token            sample 2 tokens
```

## Request

```text
request_id
prompt_tokens        immutable
generated_tokens     grows by one per step
status               WAITING → RUNNING → FINISHED
max_tokens
block_table          filled and grown by the engine
```

Each request owns only *logical* state. Three things are shared by everybody:

```text
TinyLlama        the weights
PagedKVCache     the K/V pool
BlockPool        which blocks are free
```

## Scheduler

Intentionally simple. `schedule()` returns one of two kinds of batch:

- a **prefill batch** — the requests admitted by this call (they have no KV yet), or
- a **decode batch** — all running requests, one token each.

Admission is FIFO and needs both a seat (`max_num_seqs`) and a block budget that
covers the request's worst-case length; see the module docstring of
`scheduler.py` for the exact rule and why the budget subtracts blocks already
promised to running requests. With that rule the engine never runs out of blocks
mid-decode and needs no preemption.

## Batched decode

```text
requests   A (num_tokens 9)     B (num_tokens 5)
input_ids  [[a8], [b4]]                         [batch=2, seq_len=1]   the last token of each
positions  [[8],  [4]]                          its position
metadata   slot_mapping  [slot(A, 8), slot(B, 4)]
           block_tables  [A.block_table, B.block_table]
           seq_lens      [9, 5]
```

The model runs once. Inside `Attention.attend_paged` the new K/V of both requests
are written by one `kv_cache.write`, then attention runs per sequence with that
sequence's own block table and length — this is where you extend M4's
single-sequence code to a loop.

Why is the "last token" the one to feed? After prefill (or the previous decode) the
engine sampled a token and appended it, but nobody has computed its K/V yet. The
decode step processes exactly that token and yields the *next* one.

# Starting State

- `Request.__init__`, `RequestStatus`, `Scheduler.__init__` (with `waiting`/`running` lists),
  `LLMEngine.__init__` (creates the pool, cache and scheduler), `add_request`, `run`,
  `format_state` are provided.
- `Attention.attend_paged` handles one sequence (M4).

# Your Tasks

`tiny_vllm/request.py`

- `num_tokens`, `all_tokens`, `last_token`, `append_token`, `should_stop`.

`tiny_vllm/scheduler.py`

- `add_request`, `has_unfinished_requests`, `finish`
- `can_admit`, `schedule`

`tiny_vllm/attention.py`

- Extend `attend_paged` to a batch: one `paged_attention` call per sequence, stacked.

`tiny_vllm/engine.py`

- `_prefill(request)`
- `_decode(requests)`
- `step()` — schedule, execute, sample, mark finished requests (`scheduler.finish`).
  Block reclamation is Milestone 6.

# Checkpoints

```text
5.1 Request abstraction           pytest -m m5 -k TestRequest
5.2 Waiting / running state       pytest -m m5 -k RequestStates
5.3 Basic scheduler               pytest -m m5 -k SchedulePolicy
5.4 Batched model execution       pytest -m m5 -k BatchedExecution
5.5 Per-request KV mappings       pytest -m m5 -k "PerRequestKVMappings or integration"
```

# Required Invariants

1. Every request's output equals `generate_naive` for its prompt, regardless of which other requests are in the batch.
2. Status goes WAITING → RUNNING → FINISHED and never backwards; finished requests are not scheduled again.
3. A decode step is **one** forward pass with `batch == len(running)`.
4. Block tables of running requests are pairwise disjoint at every step.
5. The pool content addressed by a request's block table equals a private contiguous cache filled with that request's tokens (no cross-request reads or writes).
6. `num_free_blocks + Σ len(block_table) == num_blocks` at every step.
7. Blocks are allocated lazily: after prefill a request holds exactly `blocks_needed(len(prompt))` blocks.

# Tests

- `test_scheduler.py` covers `Request` and `Scheduler` without any model.
- `test_engine.py::TestBatchedExecution` uses `RecordingModel` to count forward passes and check batch shapes.
- `TestPerRequestKVMappings` gathers each running request's K/V out of the shared
  pool after every step and compares it with a `ContiguousKVCache` built for that
  request alone — the strongest possible isolation check.

# Experiment

```bash
python examples/multiple_requests.py
```

serves a few prompts of different lengths on the real model and prints the finished
texts together with the block-pool picture after each step.

# Expected Observations

- With `N` running requests a decode step takes barely longer than with one; the
  time per *token* drops roughly by `N` (until the CPU saturates).
- Prefill of a long prompt dominates the step in which it happens — a first hint at
  why real schedulers chunk prefills.

# Hints

- `positions` for decode is `torch.tensor([[r.num_tokens - 1] for r in requests])`.
- Build the slot mapping per request with `compute_slot_mapping` and concatenate.
- Sampling: `greedy_sample(logits[:, -1])` gives one token per row.
- `schedule()` must return a *copy* of `running` for a decode batch; the engine
  mutates `running` while iterating otherwise.
- Do not allocate anything in `schedule()`; the budget rule is what keeps it safe.

# Questions You Should Be Able to Answer

1. Why does batching decode steps improve throughput but not the latency of a single token?
2. What state must be per request and what can be shared? Why?
3. What could go wrong if two requests' slot mappings overlapped for one step?
4. Why does the admission rule subtract blocks already promised to running requests?
5. Why is prefill run one request at a time while decode is batched?

# Optional Stretch Goal

Prefill several admitted requests in one forward pass. What does the attention
layer need to know that it does not know today? (This is what "packed"/"varlen"
attention and `cu_seqlens` are about.)
