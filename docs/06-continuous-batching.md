# Milestone 6 — Continuous Batching

# Goal

Make the engine *continuous*: the moment a request finishes, its blocks return to
the pool and a waiting request takes its place — while the other requests keep
decoding without interruption.

# Why This Exists

Static batching runs a fixed group of requests until the *longest* one is done:

```text
A █████████████
B ███ · · · · · · · · · ·     <- finished, but its seat and blocks stay idle
C ███████ · · · · · ·
```

Requests have wildly different lengths, so most of the batch is idle most of the
time. Continuous batching (a.k.a. iteration-level scheduling) re-decides the batch
at **every step**:

```text
B finishes
    │
    ▼
B's blocks freed
    │
    ▼
Request D admitted
    │
    ▼
D immediately starts making progress
```

The block-based cache is what makes this cheap: freeing a request is `len(block_table)`
list operations, and the newcomer's blocks can be anywhere.

# Mental Model

```text
while engine.has_work():
    batch, is_prefill = scheduler.schedule()      # may admit waiting requests NOW
    if is_prefill:  prefill each new request      # allocate their prompt blocks
    else:           decode all running requests   # allocate a block only at a boundary
    sample, append tokens
    for finished requests:
        scheduler.finish(r)                       # leave the running set
        free r's blocks                           # back to the pool  <-- new in M6
```

Observed through the pool picture:

```text
t0     A block_table = [7, 2]      B block_table = [5]      C block_table = [1, 9]
       free = [0, 3, 4, 6, 8, ...]

t1     B finishes
       physical block 5 → FREE

t2     D admitted
       D block_table = [5]
```

Two kinds of dynamic allocation are happening:

- **admission**: prompt blocks for a new request, exactly `blocks_needed(len(prompt))`;
- **growth**: one more block for a running request each time its length crosses a
  multiple of `block_size`.

And one kind of reclamation: all blocks of a request, the step it finishes.

# Starting State

Everything from Milestone 5 works. `LLMEngine._free_request_blocks` raises
`NotImplementedError`, and `step()` does not yet call it.

# Your Tasks

`tiny_vllm/engine.py`

- `_free_request_blocks(request)` — return every block to `block_pool`, clear the table.
- Call it from `step()` for every request that finished in that step.

That is genuinely all the new code. The rest of the milestone is verifying that the
system you built behaves as a continuous-batching engine:

- requests added later or waiting for a seat/budget get admitted as soon as one frees up (6.2),
- block tables grow only at boundaries and never reserve the worst case (6.3),
- `num_free_blocks + Σ live blocks == num_blocks` at every step, finished requests hold nothing (6.4),
- a saturated pool recycles the same physical blocks through many requests (6.5),
- `engine.run()` yields requests in completion order and leaves a clean pool (6.6).

# Checkpoints

```text
6.1 Request completion            pytest -m m6 -k RequestCompletion
6.2 Dynamic request admission     pytest -m m6 -k DynamicAdmission
6.3 Dynamic block allocation      pytest -m m6 -k DynamicBlockAllocation
6.4 Block reclamation             pytest -m m6 -k BlockReclamation
6.5 Block reuse                   pytest -m m6 -k BlockReuse
6.6 Continuous batching           pytest -m m6 -k "ContinuousBatchingLoop or integration"
```

# Required Invariants

1. A request is finished exactly when `should_stop` becomes true; it is returned by `step()` once and never scheduled again.
2. A finished request's `block_table` is empty and its blocks are free in the same step.
3. `num_free_blocks + Σ_{running} len(block_table) == num_blocks` after every step.
4. Waiting requests are admitted as soon as `can_admit` holds — in the very next `step()`.
5. The pool is never exhausted (`OutOfBlocksError` never escapes) as long as `add_request` accepted the request.
6. Outputs are unchanged by any of this: every request still equals `generate_naive`.

# Tests

- `TestBlockReuse::test_new_request_receives_blocks_of_a_finished_one` leaves exactly
  two free blocks, lets one request fill both and finish, and checks that the next
  request can only have received those two.
- `TestDynamicAdmission::test_pool_is_never_exhausted` pushes 10 requests through a
  6-block pool.
- `test_m6_integration_continuous_batching` is the plan's t0/t1/t2 demonstration as a test.

# Experiment

```bash
python benchmarks/benchmark_serving.py
python benchmarks/run_all_experiments.py        # Experiments A-D in one report
```

`benchmark_serving.py` runs the same prompts (different `max_tokens`) two ways:

- **sequential** — one request at a time through the engine (`max_num_seqs=1`);
- **continuous batching** — all requests at once.

It prints total time, tokens per second, and — with `--trace` — the block-pool
picture after each step so you can watch blocks being freed and reused.

# Expected Observations

- Continuous batching finishes the same work in far fewer engine steps than
  sequential serving, and total wall time drops accordingly. On a laptop CPU the
  speed-up is modest (the CPU is already busy with one sequence); on a GPU the same
  code structure yields large gains. Do not overstate CPU numbers.
- In the trace, a freed block id reappears in a later request's table within a step
  or two.
- With decode routed through `paged_attention_decode` (4.6) the CPU numbers are a bit
  worse than with the gather path — that loop is slow in Python. The *step counts* are
  what continuous batching changes; see the note in `04-paged-attention.md`.
- `num_free_blocks` oscillates but never goes negative and always returns to
  `num_blocks` at the end.

# Hints

- Free blocks *after* `scheduler.finish`, and make sure you iterate over a copy of
  the batch — finishing modifies `running`.
- `format_state()` (or `log_blocks=True`) is the quickest way to see what happened.
- If `test_pool_is_never_exhausted` fails, the problem is almost always in
  `can_admit` (M5), not here.

# Questions You Should Be Able to Answer

1. Why does continuous batching increase throughput even though every individual forward pass does the same work?
2. What would happen to memory if finished requests did not release their blocks?
3. Why can a newly admitted request use the freed blocks immediately without any copying?
4. What is the worst case for our admission rule, and what does real vLLM do instead (preemption / recomputation / swapping)?
5. Where would "prefix caching" plug into this design?

# Optional Stretch Goal

Add a `--arrival` option to `benchmark_serving.py` that adds requests at different
steps instead of all up front, and measure the latency each request experiences
between admission and completion. Then read `07-real-vllm.md`.
