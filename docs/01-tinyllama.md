# Milestone 1 — TinyLlama: Own the Model

# Goal

Implement a minimal Llama-like decoder in `tiny_vllm/` and load the HuggingFace
weights of `HuggingFaceTB/SmolLM2-135M` into it, so that for the same input tokens

```text
HuggingFace logits  ≈  TinyLlama logits
```

# Why This Exists

An inference engine cannot treat the model as a black box. Everything that comes
later — caching K/V, storing K/V in blocks, attending over scattered blocks —
happens *inside* the attention layer. To control where K and V go, you have to be
the one computing them.

The key realisation of this milestone:

> K/V do not need to be "extracted" from HuggingFace. Once you own the model, Q/K/V
> are ordinary tensors produced inside your own attention forward pass.

HuggingFace supplies three artefacts and nothing else:

```text
config     numbers: hidden_size, num_layers, num_heads, ...
tokenizer  text  <->  token ids
weights    a dict  {parameter name: tensor}
```

The HuggingFace *model code* is used only as a reference implementation in tests.

# Mental Model

```text
input_ids  [batch, seq_len]
    │ embed_tokens
    ▼
h   [batch, seq_len, hidden_size]
    │
    │   ┌─────────────── DecoderLayer (× num_layers) ───────────────┐
    │   │                                                           │
    ├──►│  h = h + Attention( RMSNorm(h) )     "attention block"    │
    │   │  h = h + MLP( RMSNorm(h) )           "feed-forward block" │
    │   │                                                           │
    │   └───────────────────────────────────────────────────────────┘
    ▼
RMSNorm  →  lm_head
    ▼
logits  [batch, seq_len, vocab_size]
```

Inside `Attention`:

```text
x [batch, seq_len, hidden]
 │
 ├── q_proj ──► q [batch, seq_len, num_heads,    head_dim] ──► RoPE ──┐
 ├── k_proj ──► k [batch, seq_len, num_kv_heads, head_dim] ──► RoPE ──┼──► causal_attention ──► o_proj
 └── v_proj ──► v [batch, seq_len, num_kv_heads, head_dim] ───────────┘
```

SmolLM2 uses *grouped-query attention*: 9 query heads share 3 K/V heads. Query head
`h` reads K/V head `h // 3`. This is why `k` and `v` have fewer heads than `q`, and
why the KV cache (next milestone) is smaller than you might expect.

## RMSNorm

```text
y = x / sqrt( mean(x²) + eps ) · weight
```

No mean subtraction, no bias. HuggingFace does the division in float32 and casts
back before multiplying by `weight`; match the order so loaded weights behave
identically.

## RoPE (rotary position embedding)

Instead of adding a position vector, RoPE *rotates* every pair of dimensions of
q and k by an angle proportional to the token position. There are `head_dim / 2`
frequencies:

```text
inv_freq[i] = theta ^ ( -2i / head_dim )          i = 0 .. head_dim/2 - 1
angle[p, i] = p · inv_freq[i]
```

Llama pairs dimension `i` with dimension `i + head_dim/2` ("rotate_half"), so the
cos/sin tables are the `head_dim/2` angles laid out twice along the last axis:

```text
cos[p] = [ cos(angle[p,0]) … cos(angle[p,d/2-1]) | cos(angle[p,0]) … cos(angle[p,d/2-1]) ]
```

and the rotation is

```text
x = [ x₁ | x₂ ]                     (two halves of head_dim)
rotate_half(x) = [ -x₂ | x₁ ]
RoPE(x) = x · cos + rotate_half(x) · sin
```

The important property (and the reason a KV cache works at all): the dot product
of a rotated query at position `m` and a rotated key at position `n` depends only
on `m − n`. Positions are absolute inputs, but attention sees relative distance.

## Causal attention

```text
scores  = q · kᵀ / sqrt(head_dim)         [num_heads, num_queries, num_keys]
mask    : query i may only see keys ≤ its own position
weights = softmax(scores, dim=-1)
out     = weights · v
```

`causal_attention(q, k, v)` allows `num_keys >= num_queries`: the queries are the
*last* `num_queries` positions of the key sequence. With equal lengths this is
ordinary self-attention; with one query it is a decode step. You will be glad of
this generality in Milestone 2.

# Starting State

- `TinyLlamaConfig` and all `__init__` methods are written; parameter names mirror
  HuggingFace (`layers.0.self_attn.q_proj`, `norm`, `lm_head`, …).
- `Attention.forward` dispatches to `project_qkv` and `causal_attention`; the other
  two branches belong to later milestones.
- `loader.py` downloads config, tokenizer and the raw safetensors dict.
- Every forward method raises `NotImplementedError`.

# Your Tasks

`tiny_vllm/layers.py`

- `RMSNorm.forward`
- `compute_rope_cos_sin(positions, head_dim, theta)`
- `apply_rope(x, cos, sin)`
- `MLP.forward` — `down_proj( silu(gate_proj(x)) · up_proj(x) )`

`tiny_vllm/attention.py`

- `causal_attention(q, k, v)`
- `Attention.project_qkv(hidden_states, cos, sin)`

`tiny_vllm/model.py`

- `DecoderLayer.forward`
- `TinyLlama.forward` — embed, compute cos/sin once from `positions`, run the layers,
  final norm, `lm_head`. Pass `kv_cache` / `attn_metadata` straight through even
  though they are `None` for now.

`tiny_vllm/loader.py`

- `load_weights(model, hf_state_dict)` — map HF names to ours, copy in place,
  handle dtype, tied embeddings and missing/unexpected keys.

# Checkpoints

```text
1.1 RMSNorm                       pytest -m m1 tests/test_layers.py -k RMSNorm
1.2 RoPE                          pytest -m m1 tests/test_layers.py -k RoPE
1.3 Self-attention                pytest -m m1 tests/test_model.py -k "CausalAttention or AttentionModule"
1.4 MLP + DecoderLayer            pytest -m m1 -k "MLP or DecoderLayer"
1.5 HuggingFace weight loading    pytest -m m1 -k LoadWeights
1.6 Full TinyLlama forward        pytest -m m1 -k "TinyLlamaForward or integration"
```

# Required Invariants

1. Every module matches its HuggingFace counterpart on random inputs (`rtol=atol=1e-4`).
2. Attention is causal: changing token `t` never changes logits at positions `< t`.
3. Rows of a batch are independent.
4. RoPE encodes relative position: shifting *all* positions by a constant leaves the logits unchanged.
5. `load_weights` fills every parameter and rejects missing or unexpected keys.

# Tests

The suite is built so that a failure points at one layer:

- `test_layers.py` compares `RMSNorm`, RoPE and `MLP` with the HF modules.
- `test_model.py::TestAttentionModule` feeds your attention the *exact* tensor HF's
  layer-0 attention received (captured with a forward hook) and compares outputs.
- `TestDecoderLayer` does the same per layer.
- `TestTinyLlamaForward` compares logits, then hidden states after every layer, so
  you can see where divergence starts.
- `test_m1_integration_real_model` (marker `hf`) compares against the real SmolLM2.

# Experiment

`examples/hf_reference.py` runs SmolLM2 with the HuggingFace implementation. Once
your model loads, `benchmarks/run_all_experiments.py --only A` prints the maximum
absolute logit difference between HuggingFace and TinyLlama on a few prompts.

# Expected Observations

- Max abs logit difference for the random tiny model: `< 1e-4`.
- For the real 30-layer model differences accumulate to around `1e-4 … 1e-3`
  (float32 summation order), still with identical argmax.
- Getting the RoPE pairing wrong makes the *random* model tests fail but produces
  plausible-looking text with the real model. That is why the tests exist.

# Hints

- `torch.testing.assert_close` prints the largest mismatch; look at *which*
  positions differ. Position 0 correct but position 1 wrong → RoPE. All positions
  wrong by a scale → normalisation. Only the last layer wrong → `lm_head`/tying.
- `positions` is `[batch, seq_len]`; broadcasting cos/sin over heads needs one
  `unsqueeze`.
- For GQA, `repeat_interleave` along the head dimension is the most readable way to
  expand K/V.
- The causal mask for `num_keys > num_queries`: query `i` sits at absolute position
  `num_keys - num_queries + i`.
- `hf_state_dict` for a *tied* model has no `lm_head.weight`. Our `TinyLlama.__init__`
  already ties `lm_head.weight` to `embed_tokens.weight`, so loading the embedding
  is enough — but the check that both are consistent when the key *is* present is
  yours.
- Iterate over `model.named_parameters()` and build the expected HF key from each
  name; then compare key sets in both directions.

# Questions You Should Be Able to Answer

1. Why does RoPE let you compute the K of a token once and reuse it no matter how many tokens come later?
2. Which tensors inside `Attention.forward` will the KV cache store, and which of them depend on later tokens?
3. What does grouped-query attention change about the shape of K and V, and why is that good for a KV cache?
4. Why can you compare *hidden states* per layer with HuggingFace but not the last entry of `output_hidden_states`?
5. What would silently go wrong if you loaded weights with `strict=False` semantics?

# Optional Stretch Goal

Run TinyLlama in bfloat16 (`load_tiny_llama(dtype=torch.bfloat16)`) and measure how
the max logit difference and the agreement of argmax change. Where should the
softmax and RMSNorm be computed in float32 to keep the drift small?
