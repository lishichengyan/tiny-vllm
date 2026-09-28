# Milestone 2 — Generation + KV Cache

# Goal

Generate text autoregressively — first the deliberately wasteful way, then with a
contiguous KV cache so that each decode step runs the model on **one** token.

```text
uncached logits  ≈  cached logits
```

# Why This Exists

A language model produces one token per forward pass. Producing 100 tokens means
100 forward passes, each conditioned on everything before it:

```text
ABC     ─►  D
ABCD    ─►  E
ABCDE   ─►  F
```

The naive loop re-computes the K and V of `A`, `B`, `C` at every step. But K and V
of a token depend only on that token and the tokens *before* it (causality + RoPE
relative encoding from Milestone 1). Once computed they never change. Storing them
is the KV cache. It is the single most important optimisation in LLM inference and
it is also what will eat all your memory in Milestone 3.

# Mental Model

## Prefill

```text
tokens   A B C
         │ │ │           one forward pass, seq_len = 3
         ▼ ▼ ▼
layer l: K K V  ──► kv_cache.k_cache[l, 0:3]
         V V V  ──► kv_cache.v_cache[l, 0:3]
                 logits[-1] ──► next token D
```

## Decode

```text
token D at position 3        one forward pass, seq_len = 1
   │
   ├── Q_D
   ├── K_D ──► k_cache[l, 3]
   └── V_D ──► v_cache[l, 3]

Q_D  attends to  K_A K_B K_C K_D   (= k_cache[l, 0:4])
                 ──► logits ──► next token E
```

The cache is indexed by **absolute position**, and every layer has its own slice:

```text
k_cache: [num_layers, max_seq_len, num_kv_heads, head_dim]
                        ▲
                        └── position p of layer l lives at k_cache[l, p]
```

Where in the code does this happen? Inside `Attention.forward`, which already
dispatches on the cache type:

```text
kv_cache is None               ─► causal_attention(q, k, v)            (M1)
kv_cache is ContiguousKVCache  ─► self.attend_contiguous(...)          (M2)  ◄── you
kv_cache is PagedKVCache       ─► self.attend_paged(...)               (M3+)
```

`attend_contiguous` writes the new K/V at their positions, reads back the whole
history, and calls the same `causal_attention` you wrote in M1 — with
`num_keys > num_queries`.

# Starting State

- `TinyLlama.forward(input_ids, positions, kv_cache=None, attn_metadata=None)` works without a cache.
- `ContiguousKVCache.__init__` stores the sizes; the tensors are not allocated.
- `generation.py` has the four generation functions with full docstrings.
- `sampler.greedy_sample` is empty.

# Your Tasks

`tiny_vllm/sampler.py`

- `greedy_sample(logits)` — argmax over the last dim.

`tiny_vllm/generation.py`

- `generate_naive(model, prompt_tokens, max_new_tokens, eos_token_id)` — whole
  sequence every step, `kv_cache=None`.

`tiny_vllm/kv_cache.py`

- `ContiguousKVCache.__init__` — allocate `k_cache`, `v_cache`.
- `ContiguousKVCache.write(layer_idx, positions, k, v)`
- `ContiguousKVCache.read(layer_idx, seq_len)`

`tiny_vllm/attention.py`

- `Attention.attend_contiguous(q, k, v, positions, kv_cache)`

`tiny_vllm/generation.py`

- `prefill(model, kv_cache, prompt_tokens)` — one forward over the prompt, return the last position's logits.
- `decode_step(model, kv_cache, token, position)` — one forward with `input_ids` of shape `[1, 1]`.
- `generate_with_kv_cache(model, prompt_tokens, max_new_tokens, eos_token_id)`

All generation functions return only the *new* tokens (EOS included if produced).

# Checkpoints

```text
2.1 Naive generation + greedy sampling    pytest -m m2 -k "GreedySample or GenerateNaive"
2.2 Contiguous KV cache                   pytest -m m2 -k ContiguousKVCache
2.3 Prefill                               pytest -m m2 -k Prefill
2.4 Decode                                pytest -m m2 -k DecodeStep
2.5 Cached generation                     pytest -m m2 -k "GenerateWithKVCache or integration"
```

# Required Invariants

1. `generate_with_kv_cache(...) == generate_naive(...)` token for token.
2. `decode_step` logits equal the last-position logits of an uncached forward over the whole sequence.
3. After prefill, every decode forward sees `input_ids` of shape `[1, 1]`.
4. Cache position `p` holds the K/V of token `p` for *every* layer; positions ≥ the current length are untouched.
5. `prefill(ABC) + decode(D) + decode(E)` leaves exactly the same cache as `prefill(ABCDE)`.

# Tests

- `ScriptedModel` (in `tests/helpers.py`) is a fake model whose greedy next token
  after position `p` is `script[p]`. It checks stopping rules and call shapes
  without depending on real weights.
- `RecordingModel` wraps the real model and records every forward call — this is
  how the suite verifies "decode processes only the newest token".
- `test_generation.py` compares naive, cached and HuggingFace `generate()`.
- `test_kv_cache.py::TestContiguousKVCache` uses NaN-filled caches to prove that
  writes land *only* where they should.

# Experiment

```bash
python benchmarks/benchmark_kv_cache.py
```

runs naive vs cached generation on the real model and reports

- **TTFT** (time to first token = the prefill),
- **TPOT** (time per output token = mean decode latency),
- total time.

# Expected Observations

- TPOT for naive generation grows with the sequence length; TPOT with the cache is
  roughly constant. The gap widens with `max_new_tokens`.
- TTFT is essentially identical: both variants run one forward over the prompt.
  A KV cache does **not** make the first token faster — it only avoids
  recomputing the past. Do not over-claim.
- On CPU with a 135M model the decode step is dominated by weight reads, so the
  cached TPOT is a few tens of milliseconds regardless of sequence length.

# Hints

- `positions` for a decode step is `torch.tensor([[position]])`.
- `attend_contiguous` only ever sees `batch == 1`; assert it, then drop the batch
  dim before writing and add it back before `causal_attention`.
- The number of valid cache entries after writing is `positions.max() + 1`.
- Generating `max_new_tokens=0` tokens should return `[]` without calling the model.
- Make the cache big enough: `len(prompt) + max_new_tokens` positions.

# Questions You Should Be Able to Answer

1. Why does the decode step for token `D` only need `Q_D` but all of `K_A..K_D` and `V_A..V_D`?
2. How much memory does the cache take for SmolLM2 (30 layers, 3 KV heads, head_dim 64, float32) per token? For 2048 tokens?
3. What does the KV cache change about TTFT? About TPOT? Why?
4. Why must `positions` be passed explicitly instead of being inferred from `input_ids.shape[1]`?
5. If two requests run concurrently, what is wrong with `ContiguousKVCache` as designed?

# Optional Stretch Goal

Change `generate_with_kv_cache` to accept a `max_seq_len` smaller than the final
length and see where it fails. Then think about what "a cache that can grow" would
need — that is the problem Milestone 3 solves.
