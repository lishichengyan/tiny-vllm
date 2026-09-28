# tiny-vllm — Overview

tiny-vllm is a course, not a library. You will build a minimal inference engine
that reproduces the *architecture* of vLLM — block-based KV caching, paged
attention, a scheduler and continuous batching — in plain PyTorch on a laptop CPU.

The repository already contains the scaffolding: module layout, constructors,
configuration, HuggingFace loading, tests, benchmarks, docs. What is missing is
the part worth learning. Every gap is marked:

```bash
grep -R "TODO(student)" tiny_vllm/
```

Each `TODO(student)` raises `NotImplementedError("Milestone N (x.y): ...")` so you
always know which checkpoint a failure belongs to.

## The path you will build

```text
HuggingFace config + tokenizer + weights
                    │
                    ▼
                TinyLlama                       M1
                    │
                    ▼
                  Q K V
                    │
                    ▼
                KV Cache                        M2
                    │
                    ▼
          Physical KV Blocks                    M3
                    │
             ┌──────┴──────┐
             ▼             ▼
        Block Table    Slot Mapping             M3
             │             │
             └──────┬──────┘
                    ▼
             Paged Attention                    M4
                    │
                    ▼
                  Logits
                    │
                    ▼
               Next Token
```

and then turn it into a serving engine:

```text
                 Requests
                    │
                    ▼
                Scheduler                       M5
                    │
                    ▼
            Batched Execution                   M5
                    │
                    ▼
      Block allocation / reclamation            M6
                    │
                    ▼
                TinyLlama + Paged Attention
                    │
                    ▼
                  Tokens
```

## Milestones

| #  | Name                     | Doc                              | Tests                                             |
|----|--------------------------|----------------------------------|---------------------------------------------------|
| M1 | TinyLlama                | `01-tinyllama.md`                | `pytest -m m1`                                    |
| M2 | Generation + KV Cache    | `02-kv-cache.md`                 | `pytest -m m2`                                    |
| M3 | Block-Based KV Cache     | `03-block-kv-cache.md`           | `pytest -m m3`                                    |
| M4 | Paged Attention          | `04-paged-attention.md`          | `pytest -m m4`                                    |
| M5 | Multi-Request Engine     | `05-multi-request-engine.md`     | `pytest -m m5`                                    |
| M6 | Continuous Batching      | `06-continuous-batching.md`      | `pytest -m m6`                                    |
|    | How real vLLM differs    | `07-real-vllm.md`                | —                                                 |
|    | GPU track (optional)     | `08-gpu-track.md`                | `pytest tests/gpu` (needs CUDA + Triton)          |

Run the milestones in order. Later milestones reuse everything before them and
the whole suite must stay green: `pytest` runs everything.

Tests marked `hf` download the real model (`HuggingFaceTB/SmolLM2-135M`, ~270 MB)
once. Skip them with `-m "m1 and not hf"` when offline.

## What is provided vs. what you write

Provided (infrastructure, read it but do not rewrite it):

- `config.py` — `TinyLlamaConfig`, conversion from the HF config
- every `__init__` (module structure, parameter shapes) in `layers.py`, `attention.py`, `model.py`
- `loader.py` except `load_weights`
- `kv_cache.AttentionMetadata`, `block_pool.OutOfBlocksError`
- `Attention.forward` dispatch (chooses between no-cache / contiguous / paged)
- `request.Request.__init__`, `scheduler.Scheduler.__init__`, `engine.LLMEngine.__init__/add_request/run/format_state`
- `visualize.py`, `tests/`, `benchmarks/`, `examples/`, `docs/`

Yours (every `TODO(student)`):

| Milestone | You implement |
|-----------|---------------|
| M1 | `RMSNorm.forward`, `compute_rope_cos_sin`, `apply_rope`, `MLP.forward`, `causal_attention`, `Attention.project_qkv`, `DecoderLayer.forward`, `TinyLlama.forward`, `load_weights` |
| M2 | `greedy_sample`, `generate_naive`, `ContiguousKVCache`, `Attention.attend_contiguous`, `prefill`, `decode_step`, `generate_with_kv_cache` |
| M3 | `PagedKVCache` (allocation + `write`), `BlockPool`, `blocks_needed`, `logical_to_physical`, `ensure_block_capacity`, `compute_slot_mapping` |
| M4 | `gather_kv`, `paged_attention`, `Attention.attend_paged` (single sequence), `generate_paged`, `paged_attention_decode` (online softmax over blocks, no gather) |
| GPU (optional) | the Triton kernel in `kernels/paged_attention_triton.py` + a roofline writeup |
| M5 | `Request` methods, `Scheduler`, `LLMEngine.step/_prefill/_decode`, `Attention.attend_paged` (batch) |
| M6 | `LLMEngine._free_request_blocks` + making `step` continuous |

## Conventions

Shapes are always written out:

```python
# hidden_states: [batch, seq_len, hidden_size]
# q:             [batch, seq_len, num_heads,    head_dim]
# k, v:          [batch, seq_len, num_kv_heads, head_dim]
# positions:     [batch, seq_len]
# logits:        [batch, seq_len, vocab_size]
```

`batch` is the number of sequences in a forward pass and `seq_len` the number of
new tokens *per sequence*: the prompt length during prefill, `1` during decode.
Prefill always runs one sequence at a time; decode batches many sequences with one
token each.

Prefer boring code:

```python
logical_block = token_position // block_size
offset = token_position % block_size
physical_block = block_table[logical_block]
```

## Progress banner

Running a single milestone prints a checklist:

```text
========================================
MILESTONE 3 COMPLETE
Block-Based KV Cache
========================================

✓ 3.1  global KV pool works
✓ 3.2  block allocation works
...
```

## Environment

CPU only. macOS, Linux and Windows all work; Apple MPS is not required. No CUDA,
no Triton, no custom kernels — paged attention is implemented for its *semantics*.
