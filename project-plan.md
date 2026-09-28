# tiny-vllm — Build vLLM from First Principles

## 1. Objective

Create an educational project called **tiny-vllm**.

The goal is NOT to build a production inference engine or clone the entire vLLM codebase.

The goal is to learn the core architecture behind vLLM by implementing a minimal inference engine step by step.

The learner should eventually understand and implement this entire path:

```text
HuggingFace config + tokenizer + weights
                    │
                    ▼
                TinyLlama
                    │
                    ▼
                  Q K V
                    │
                    ▼
                KV Cache
                    │
                    ▼
          Physical KV Blocks
                    │
             ┌──────┴──────┐
             ▼             ▼
        Block Table    Slot Mapping
             │             │
             └──────┬──────┘
                    ▼
             Paged Attention
                    │
                    ▼
                  Logits
                    │
                    ▼
                Next Token
```

Then extend the single-request implementation into a minimal serving engine:

```text
                 Requests
                    │
                    ▼
                Scheduler
                    │
                    ▼
            Batched Execution
                    │
                    ▼
             KV Cache Manager
                    │
                    ▼
          Physical Block Pool
                    │
                    ▼
                TinyLlama
                    │
                    ▼
             Paged Attention
                    │
                    ▼
                  Tokens
```

The final learner-written core implementation should ideally remain around **1,000–2,000 lines of Python**.

---

# 2. Philosophy

Treat this repository like a university systems course project.

You are the **course designer and TA**, not the student.

You should create:

- repository structure
- interfaces
- fixtures
- test infrastructure
- HuggingFace reference implementations
- benchmark infrastructure
- visualization/debugging helpers
- milestone documentation
- non-core boilerplate

The learner should implement the important inference-engine logic.

Do NOT generate a complete tiny-vllm and then randomly delete functions.

Every missing implementation must correspond to an important concept the learner is supposed to understand.

The repository should guide the learner toward discovering why each abstraction exists.

---

# 3. Critical Rule: Do Not Leak Solutions

Do not provide the implementation of learner exercises elsewhere in the repository.

For example, if the learner must implement:

```python
BlockPool.allocate()
```

do not implement equivalent allocation logic inside:

- tests
- helpers
- benchmarks
- examples
- documentation
- reference code

Tests should describe expected behavior and invariants, not reveal the implementation.

Documentation may contain:

- conceptual explanations
- equations
- diagrams
- examples
- hints

but should NOT contain copy-pastable solutions.

Use TODOs such as:

```python
# TODO(student): Milestone 3
raise NotImplementedError("Milestone 3: implement BlockPool.allocate()")
```

---

# 4. Hardware / Environment

The project must work on a **MacBook without an NVIDIA GPU**.

Requirements:

- Python
- PyTorch
- HuggingFace Transformers
- pytest

CPU must always work.

Apple MPS may optionally work but must not be required.

Do NOT use:

- CUDA
- Triton
- custom GPU kernels

Paged Attention will be implemented for **correctness and semantics**, not GPU performance.

---

# 5. Model Scope

Support exactly ONE small decoder-only Llama-like model.

Choose a genuinely small HuggingFace model suitable for running tests on a MacBook CPU.

The model should expose the architectural concepts we care about:

- token embedding
- RMSNorm
- RoPE
- causal self-attention
- Q/K/V projections
- output projection
- gated MLP
- residual connections
- decoder layers
- LM head

HuggingFace should provide:

```text
config
tokenizer
weights
```

but the main tiny-vllm inference path must use our own model implementation.

HuggingFace's model implementation should only be used as a **reference / ground truth**.

---

# 6. Explicit Non-Goals

Do NOT implement:

- CUDA kernels
- Triton kernels
- tensor parallelism
- pipeline parallelism
- distributed inference
- multi-node inference
- quantization
- LoRA
- speculative decoding
- prefix caching
- chunked prefill
- multimodal inference
- beam search
- production API server
- OpenAI API compatibility
- multiple model architectures
- advanced production scheduling policies

These may be discussed in a final document explaining how real vLLM goes beyond tiny-vllm.

---

# 7. Milestone Structure

There are exactly **6 major milestones**.

Each milestone may contain several smaller checkpoints.

The milestones are:

```text
M1  TinyLlama
M2  Generation + KV Cache
M3  Block-Based KV Cache
M4  Paged Attention
M5  Multi-Request Engine
M6  Continuous Batching
```

The first four milestones should build Paged Attention as quickly as reasonably possible.

The final two milestones explain why these mechanisms matter in a real serving engine.

---

# 8. Milestone 1 — TinyLlama: Own the Model

## Goal

Implement our own minimal Llama-like inference model and load HuggingFace weights into it.

The learner must understand:

```text
HuggingFace
config + tokenizer + weights
              │
              ▼
         our TinyLlama
              │
              ▼
            Q K V
```

The important realization is:

> K/V do not need to be "extracted" from HuggingFace. Once we own the model implementation, Q/K/V are produced directly inside our own attention forward pass.

## Learner Implements

Important pieces such as:

- RMSNorm
- RoPE
- self-attention
- Q projection
- K projection
- V projection
- output projection
- gated MLP
- decoder layer
- full model forward
- meaningful parts of weight loading

Do not over-abstract these pieces.

Tensor shapes should be explicitly documented.

For example:

```python
# hidden_states: [batch, seq_len, hidden_size]
# q: [batch, seq_len, num_heads, head_dim]
# k: [batch, seq_len, num_kv_heads, head_dim]
```

## Checkpoints

```text
1.1 RMSNorm
1.2 RoPE
1.3 Self Attention
1.4 MLP + DecoderLayer
1.5 HuggingFace weight loading
1.6 Full TinyLlama forward
```

## Tests

Compare against HuggingFace.

The most important final assertion should conceptually be:

```python
torch.testing.assert_close(
    tiny_logits,
    hf_logits,
    rtol=...,
    atol=...,
)
```

Use sensible numerical tolerances.

Tests should make debugging individual layers possible instead of only providing one giant end-to-end failure.

## Completion Criteria

Given the same input tokens:

```text
HuggingFace logits
        ≈
TinyLlama logits
```

---

# 9. Milestone 2 — Generation + KV Cache

## Goal

Understand autoregressive generation and why KV caching exists.

Start with deliberately inefficient generation:

```text
ABC
 │
 ▼
 D

ABCD
 │
 ▼
 E

ABCDE
 │
 ▼
 F
```

Every generation step recomputes previous tokens.

Then implement a traditional **contiguous KV cache**.

## Prefill

```text
A B C
│ │ │
▼ ▼ ▼
K K K
V V V
  │
  ▼
KV Cache
```

## Decode

For new token D:

```text
D
│
├── Q_D
├── K_D ──→ cache
└── V_D ──→ cache
```

Then:

```text
Q_D
 │
 ▼
K_A K_B K_C K_D
 │
 ▼
Attention
```

Only the newest token should need model computation during decode.

## Learner Implements

- naive autoregressive generation
- greedy sampling
- EOS handling
- contiguous KV cache
- KV writes
- KV reads
- prefill
- decode
- cached generation

## Checkpoints

```text
2.1 Naive autoregressive generation
2.2 Contiguous KV cache
2.3 Prefill
2.4 Decode
2.5 Cached generation
```

## Correctness Tests

Verify:

```text
uncached logits
      ≈
cached logits
```

Verify that decode processes only the newest token where appropriate.

Verify KV cache contents correspond to expected token positions.

## Experiment

Benchmark:

```text
No KV cache
vs
KV cache
```

Focus primarily on:

- decode latency
- TPOT

Do not incorrectly claim that ordinary KV caching automatically improves TTFT across unrelated requests.

---

# 10. Milestone 3 — Block-Based KV Cache

## Goal

Replace per-request contiguous KV storage with a global block-based KV memory pool.

This is where the project begins implementing the core memory-management idea associated with vLLM.

Instead of:

```text
Request A:

[K0 K1 K2 K3 K4 K5 K6 K7 ...]
```

use:

```text
Global KV Pool

Block 0 [ ][ ][ ][ ]
Block 1 [ ][ ][ ][ ]
Block 2 [ ][ ][ ][ ]
Block 3 [ ][ ][ ][ ]
...
```

## KV Cache

Preallocate global K/V tensors conceptually similar to:

```python
k_cache = torch.empty(
    num_layers,
    num_blocks,
    block_size,
    num_kv_heads,
    head_dim,
)

v_cache = torch.empty(
    num_layers,
    num_blocks,
    block_size,
    num_kv_heads,
    head_dim,
)
```

The exact tensor ordering may differ if there is a clear reason, but it must be documented.

## Separate Four Concepts

### 1. KVCache

Owns the actual K/V tensors.

### 2. BlockPool

Manages physical block availability.

A physical block may simply be represented by an integer ID.

Example:

```text
0 FREE
1 USED
2 USED
3 FREE
...
```

### 3. Block Table

Maps a request's logical blocks to physical KV blocks.

Example:

```python
block_table = [7, 2, 11]
```

means:

```text
logical block 0 → physical block 7
logical block 1 → physical block 2
logical block 2 → physical block 11
```

### 4. Slot Mapping

Identifies where newly produced K/V should be written.

Example:

```text
block_size = 4
block_table = [7, 2]

token position = 5

logical_block = 5 // 4 = 1
offset        = 5 % 4  = 1

physical_block = block_table[1]
               = 2

slot = physical_block * block_size + offset
     = 9
```

Preserve the conceptual distinction:

```text
Block Table
    │
    └── logical blocks → physical blocks

Slot Mapping
    │
    └── new K/V → physical write location
```

## Learner Implements

- global KV allocation
- block allocation
- block freeing
- block reuse
- block tables
- logical-to-physical translation
- slot mapping
- writing new K/V into physical blocks

## Checkpoints

```text
3.1 Global KV pool
3.2 Block allocation
3.3 Block free + reuse
3.4 Block table
3.5 Slot mapping
3.6 Write K/V into physical blocks
```

## Required Invariants

1. A physical block cannot independently belong to two live requests.
2. Freed blocks become reusable.
3. Physical blocks assigned to one request do NOT need to be contiguous.
4. Block exhaustion must be handled explicitly.
5. Crossing a logical block boundary must produce the correct physical address.
6. Every Transformer layer should use the correct KV storage.

## Fragmentation Test

Deliberately create:

```text
physical blocks:

0     1     2     3     4     5     6     7
USED  FREE  USED  FREE  USED  FREE  USED  FREE
```

A request requiring four blocks should be able to receive something equivalent to:

```python
block_table = [1, 3, 5, 7]
```

No contiguous physical allocation should be required.

## Visualization

Provide a helper that can display something like:

```text
Request A
block_table = [7, 2]

Request B
block_table = [5]

Physical KV Pool

0 FREE
1 FREE
2 A [tokens 4-7]
3 FREE
4 FREE
5 B [tokens 0-3]
6 FREE
7 A [tokens 0-3]
```

The visualization helper may be provided by the coding agent because it is educational infrastructure, not the core exercise.

---

# 11. Milestone 4 — Paged Attention

## Goal

Perform attention correctly when historical K/V is physically non-contiguous.

Example:

```text
Request A logical KV:

K0 K1 K2 K3 | K4 K5 K6 K7 | K8 ...
      │              │          │
      ▼              ▼          ▼
 physical 7      physical 2  physical 11
```

The question this milestone answers is:

> How can Q attend to historical K/V if the KV cache is scattered across physical blocks?

## Learner Implements

A semantic PyTorch implementation conceptually similar to:

```python
paged_attention(
    query,
    kv_cache,
    block_table,
    sequence_length,
)
```

The implementation should:

1. resolve logical KV positions using the block table
2. read K/V from the correct physical blocks
3. preserve logical token ordering
4. perform causal attention correctly
5. handle partially filled final blocks

## Important Performance Rule

Do NOT attempt to reproduce real vLLM CUDA/Triton performance.

For this project it is acceptable to do:

```text
physical KV blocks
        │
        ▼
PyTorch gather
        │
        ▼
logical contiguous K/V
        │
        ▼
normal attention
```

This implementation reproduces the **semantics** of paged attention.

Real vLLM uses optimized attention backends/kernels that can operate much more efficiently over paged KV layouts.

Explain this distinction clearly.

## Checkpoints

```text
4.1 Resolve logical token → physical KV
4.2 Read K/V through block table
4.3 Compute attention
4.4 Handle partial final block
4.5 Integrate with TinyLlama
```

## Critical Correctness Test

Deliberately scramble physical storage:

```python
block_table = [7, 2, 11]
```

Then compare:

```python
reference = contiguous_attention(...)

paged = paged_attention(
    ...,
    block_table=[7, 2, 11],
)

torch.testing.assert_close(
    reference,
    paged,
)
```

The physical KV layout must be non-contiguous for this test.

## Milestone Completion

At this point the learner has built:

```text
HF weights
    │
    ▼
TinyLlama
    │
    ▼
Q / K / V
    │
    ▼
KV Cache
    │
    ▼
Physical Blocks
    │
    ▼
Block Table + Slot Mapping
    │
    ▼
Paged Attention
```

This is an important project checkpoint.

**TinyPagedAttention is now complete even before implementing a full multi-request serving engine.**

---

# 12. Milestone 5 — Multi-Request Engine

## Goal

Extend the single-sequence inference system into a minimal inference engine that handles multiple requests.

Introduce three important concepts:

```text
Request
Engine
Scheduler
```

## Request

A request should contain only the state actually required by the project.

Conceptually:

```text
request_id
prompt_tokens
generated_tokens
status
max_tokens
sequence_length
block_table
```

Avoid bloated abstractions.

Possible states:

```text
WAITING
   │
   ▼
RUNNING
   │
   ▼
FINISHED
```

## Shared Resources

Each request has its own logical state:

```text
sequence length
generated tokens
block table
```

All requests share:

```text
TinyLlama
Global KV Cache
BlockPool
```

## Scheduler

Keep the scheduler intentionally simple.

Do NOT reproduce real vLLM scheduling policies.

It only needs to decide which currently runnable requests participate in the next engine step.

## Learner Implements

- Request abstraction
- waiting requests
- running requests
- request admission
- request completion
- simple scheduling
- batched model execution
- per-request block mappings

## Checkpoints

```text
5.1 Request abstraction
5.2 Waiting / running state
5.3 Basic scheduler
5.4 Batched model execution
5.5 Per-request KV mappings
```

## Tests

Verify:

- multiple requests generate correct outputs
- request states transition correctly
- each request uses its own block table
- one request cannot accidentally read another request's KV
- one request cannot accidentally overwrite another request's KV
- shared KV pool remains consistent

---

# 13. Milestone 6 — Continuous Batching

## Goal

Turn the multi-request engine into a minimal continuous-batching serving engine.

Suppose:

```text
A █████████████
B ███
C ███████
```

B finishes much earlier.

We do NOT want its execution capacity and KV blocks to remain unused until A and C finish.

Instead:

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

## Learner Implements

- detecting completed requests
- dynamic request admission
- dynamic block allocation
- block reclamation
- block reuse
- continuous batching engine loop

The final loop should conceptually resemble:

```python
while engine.has_work():
    batch = scheduler.schedule()

    metadata = kv_cache_manager.prepare(batch)

    logits = model.forward(
        ...,
        block_tables=metadata.block_tables,
        slot_mapping=metadata.slot_mapping,
    )

    tokens = sampler(logits)

    scheduler.update(tokens)

    kv_cache_manager.free_finished_requests()
```

Do not require this exact API if a simpler architecture is cleaner.

## Checkpoints

```text
6.1 Request completion
6.2 Dynamic request admission
6.3 Dynamic block allocation
6.4 Block reclamation
6.5 Block reuse
6.6 Continuous batching
```

## Required Demonstration

The learner should be able to observe something similar to:

```text
t0

A block_table = [7, 2]
B block_table = [5]
C block_table = [1, 9]

free = [0, 3, 4, 6, 8, ...]


t1

B finishes

physical block 5 → FREE


t2

D admitted

D block_table = [5]
```

This should be observable through debug output or the provided visualization infrastructure.

---

# 14. Final Experiments

Do NOT create a separate experiment milestone.

Attach experiments to the relevant milestones and provide a final script that summarizes the important results.

The final project should demonstrate four things.

## Experiment A — Model Correctness

Compare:

```text
HuggingFace
    ≈
TinyLlama
```

## Experiment B — KV Cache

Compare:

```text
uncached generation
vs
cached generation
```

Measure:

- decode latency
- TPOT

## Experiment C — Paged KV Correctness

Compare:

```text
contiguous KV attention
        ≈
paged KV attention
```

while intentionally using a fragmented mapping such as:

```python
block_table = [7, 2, 11]
```

## Experiment D — Serving / Block Reuse

Run requests with different lengths.

Observe:

```text
request finishes
       │
       ▼
blocks released
       │
       ▼
new request admitted
       │
       ▼
old blocks reused
```

Optionally compare sequential serving with continuous batching throughput.

Do not overstate performance results on CPU.

---

# 15. Testing Philosophy

Testing is a first-class part of this project.

Every checkpoint should have focused tests.

Every milestone should have an integration test.

Later milestones must preserve earlier correctness.

Tests should verify **behavior and invariants**, not arbitrary implementation details.

GOOD:

```text
A freed physical block can later be reused.
```

BAD:

```text
free_blocks must be implemented using collections.deque.
```

GOOD:

```text
Paged attention produces the same result when physical blocks are reordered.
```

BAD:

```text
paged_attention must use torch.stack internally.
```

Allow the learner to choose implementation details when they do not affect the architecture being taught.

---

# 16. Documentation Format

Create one main document for each milestone:

```text
docs/
├── 01-tinyllama.md
├── 02-kv-cache.md
├── 03-block-kv-cache.md
├── 04-paged-attention.md
├── 05-multi-request-engine.md
└── 06-continuous-batching.md
```

Every document should use the same structure:

```text
# Goal

# Why This Exists

# Mental Model

# Starting State

# Your Tasks

# Checkpoints

# Required Invariants

# Tests

# Experiment

# Expected Observations

# Hints

# Questions You Should Be Able to Answer

# Optional Stretch Goal
```

Use ASCII diagrams heavily.

Do not provide solutions.

---

# 17. Concept Questions

Every milestone should end with 3–6 conceptual questions.

For example, after Milestone 3:

1. Who owns the actual K/V tensor memory?
2. What exactly is a physical block?
3. Why do blocks assigned to one request not need to be contiguous?
4. What information is stored in a block table?
5. What is the difference between a block table and a slot mapping?
6. What happens to physical blocks when a request finishes?

After Milestone 4:

1. Why does attention care about logical token order but not physical memory order?
2. How does the block table recover logical ordering?
3. Why is our PyTorch implementation slower than an optimized paged-attention kernel?
4. What part of our implementation represents the semantics of PagedAttention?
5. Why can the same block table concept be used across Transformer layers?

These questions are part of the learning experience.

---

# 18. Suggested Repository Structure

Keep the repository intentionally small.

A reasonable starting structure is:

```text
tiny-vllm/
│
├── README.md
├── pyproject.toml
│
├── tiny-vllm/
│   ├── __init__.py
│   ├── config.py
│   │
│   ├── model.py
│   ├── layers.py
│   ├── attention.py
│   ├── loader.py
│   │
│   ├── kv_cache.py
│   ├── block_pool.py
│   │
│   ├── request.py
│   ├── scheduler.py
│   │
│   ├── sampler.py
│   └── engine.py
│
├── tests/
│   ├── test_layers.py
│   ├── test_model.py
│   ├── test_generation.py
│   ├── test_kv_cache.py
│   ├── test_blocks.py
│   ├── test_paged_attention.py
│   ├── test_scheduler.py
│   └── test_engine.py
│
├── benchmarks/
│   ├── benchmark_kv_cache.py
│   └── benchmark_serving.py
│
├── examples/
│   ├── hf_reference.py
│   ├── generate.py
│   ├── multiple_requests.py
│   └── visualize_blocks.py
│
└── docs/
    ├── 00-overview.md
    ├── 01-tinyllama.md
    ├── 02-kv-cache.md
    ├── 03-block-kv-cache.md
    ├── 04-paged-attention.md
    ├── 05-multi-request-engine.md
    ├── 06-continuous-batching.md
    └── 07-real-vllm.md
```

Adjust only when there is a clear reason.

Do not create unnecessary architecture layers.

---

# 19. Code Style

Prefer boring, explicit code.

For example, prefer:

```python
logical_block = token_position // block_size
offset = token_position % block_size
physical_block = block_table[logical_block]
```

over clever abstractions hiding these operations.

Important tensors should have shape comments.

Example:

```python
# q: [num_tokens, num_heads, head_dim]
# k: [num_tokens, num_kv_heads, head_dim]
```

Avoid:

- unnecessary inheritance
- excessive dataclasses
- deep abstraction hierarchies
- factories
- registries
- plugin systems
- premature optimization

The learner should eventually be able to read the entire core codebase.

---

# 20. Progress Experience

Each milestone should provide visible progress.

For example:

```text
========================================
MILESTONE 3 COMPLETE
Block-Based KV Cache
========================================

✓ global KV pool works
✓ block allocation works
✓ block reclamation works
✓ non-contiguous allocation works
✓ block table works
✓ slot mapping works
✓ physical KV writes are correct

Next:
Use the block table to perform attention
over physically scattered KV.
```

Provide a simple command for finding remaining learner work:

```bash
grep -R "TODO(student)" tiny-vllm/
```

---

# 21. Development Strategy

Do NOT fully scaffold all six milestones with detailed future implementations at once if doing so requires introducing abstractions the learner has not encountered yet.

The repository should evolve naturally.

However, the full roadmap and documentation outline may exist from the beginning.

Prefer:

```text
Milestone 1
    ↓
learner implementation
    ↓
tests green
    ↓
Milestone 2
    ↓
learner implementation
    ↓
...
```

Future tests may be skipped until their milestone becomes active.

Earlier milestones must not require implementing future concepts.

---

# 22. Comparison With Real vLLM

Create:

```text
docs/07-real-vllm.md
```

This document should explain how the completed tiny-vllm concepts correspond conceptually to real vLLM.

Discuss concepts such as:

```text
tiny-vllm                    real vLLM

TinyLlama                   vLLM model implementation
Request                     request state
Scheduler                   scheduler
KVCache                     KV cache tensors
BlockPool                   KV cache block management
BlockTable                  block tables
SlotMapping                 KV write locations
paged_attention             optimized attention backend
Engine                      vLLM engine
```

Do NOT claim exact source-level equivalence unless verified.

When discussing current vLLM implementation details, inspect the current vLLM source code rather than relying on old blog posts or tutorials.

Also explain what real vLLM adds beyond this project:

- optimized GPU kernels
- advanced scheduling
- prefix caching
- chunked prefill
- distributed execution
- quantization
- speculative decoding
- production serving
- broader model support

This document is explanatory only.

---

# 23. Final Architecture

At completion, the learner should understand every arrow in:

```text
                 User Requests
                       │
                       ▼
                    Engine
                       │
                       ▼
                   Scheduler
                       │
                       ▼
               Active Requests
                       │
                       ▼
               KV Cache Manager
                       │
              ┌────────┴────────┐
              ▼                 ▼
         Block Pool        Block Tables
                                │
                         Slot Mappings
              └────────┬────────┘
                       ▼
                   TinyLlama
                       │
                       ▼
                    Q K V
                  ↙       ↘
         write new KV     read old KV
                  ↘       ↙
                KV Cache Pool
                       │
                       ▼
                Paged Attention
                       │
                       ▼
                    Logits
                       │
                       ▼
                    Sampler
                       │
                       ▼
                  Next Tokens
```

---

# 24. Definition of Done

The final project should support something conceptually similar to:

```python
engine = LLMEngine(...)

engine.add_request("Explain virtual memory in one sentence.")

engine.add_request("What is paged attention?")

for output in engine.run():
    print(output)
```

Internally it must actually use:

```text
multiple requests
        ↓
scheduler
        ↓
block allocation
        ↓
block tables
        ↓
slot mappings
        ↓
TinyLlama
        ↓
Q/K/V
        ↓
paged KV cache
        ↓
paged attention
        ↓
sampling
        ↓
generated tokens
```

The primary success criterion is NOT performance.

The primary success criterion is:

> The learner can explain and has personally implemented every important transition from HuggingFace weights to a minimal continuous-batching inference engine with block-based KV caching and paged attention.

---

# 25. Your Immediate Task

Do NOT implement the entire project now.

First:

1. Review this plan for technical feasibility.
2. Select an appropriate tiny Llama-like HuggingFace model for CPU testing.
3. Propose the exact repository structure.
4. Identify which code should be provided versus left as `TODO(student)`.
5. Design the test strategy for all six milestones.
6. Point out any dependency between milestones that would make the proposed order impractical.
7. Keep the architecture as simple as possible.
8. Do NOT add features outside the scope above.
9. Do NOT write the learner's core implementations yet.

After presenting that design, wait for approval before generating the scaffold.
