# tiny-vllm

Build vLLM from first principles: a minimal, CPU-only inference engine with a
block-based KV cache, paged attention, a scheduler and continuous batching —
written by you, one milestone at a time.

```text
HF weights ─► TinyLlama ─► Q K V ─► KV cache ─► physical blocks ─► block table + slot mapping
           ─► paged attention ─► logits ─► next token ─► scheduler ─► continuous batching
```

This repository is a course scaffold. The infrastructure (module layout,
constructors, HuggingFace loading, tests, benchmarks, docs) is provided; the core
inference-engine logic is left for you to implement. Every gap is a `TODO(student)`.

## Setup

Python ≥ 3.10, CPU only. No CUDA, no Triton, no custom kernels.

```bash
python -m venv .venv && source .venv/bin/activate      # or: uv venv .venv --python 3.11
pip install -e ".[dev]"                                 # or: uv pip install -e ".[dev]"
pytest --collect-only -q | tail -1                      # ~250 tests collected
python examples/hf_reference.py                         # downloads HuggingFaceTB/SmolLM2-135M (~270 MB)
python examples/visualize_blocks.py                     # works before you write any code
```

Intel Macs: PyTorch's last macOS x86_64 release is 2.2.x, which needs `numpy<2`
and `transformers<5`; `pyproject.toml` pins these for that platform.

Code style is enforced with [Ruff](https://docs.astral.sh/ruff/) (linter + formatter)
via pre-commit. Install the git hook once and it runs on every commit:

```bash
pre-commit install                 # installed by ".[dev]"
pre-commit run --all-files         # or run everything manually
ruff check . && ruff format .      # the same checks without pre-commit
```

## The model

`HuggingFaceTB/SmolLM2-135M` — a real `LlamaForCausalLM` (30 layers, hidden 576,
9 query heads / 3 KV heads, tied embeddings, bf16 safetensors). HuggingFace provides
the config, tokenizer and weights; the HuggingFace *model code* is used only as a
reference in tests. All inference runs through `tiny_vllm/model.py`.

Most tests use a 2-layer random model of the same architecture and need no download.
Tests marked `hf` use the real model and are skipped if it cannot be fetched.

## Milestones

| # | Milestone | Doc | Run |
|---|-----------|-----|-----|
| 1 | TinyLlama — own the model | [docs/01-tinyllama.md](docs/01-tinyllama.md) | `pytest -m m1` |
| 2 | Generation + KV cache | [docs/02-kv-cache.md](docs/02-kv-cache.md) | `pytest -m m2` |
| 3 | Block-based KV cache | [docs/03-block-kv-cache.md](docs/03-block-kv-cache.md) | `pytest -m m3` |
| 4 | Paged attention | [docs/04-paged-attention.md](docs/04-paged-attention.md) | `pytest -m m4` |
| 5 | Multi-request engine | [docs/05-multi-request-engine.md](docs/05-multi-request-engine.md) | `pytest -m m5` |
| 6 | Continuous batching | [docs/06-continuous-batching.md](docs/06-continuous-batching.md) | `pytest -m m6` |
|   | How real vLLM differs | [docs/07-real-vllm.md](docs/07-real-vllm.md) | — |
|   | GPU track (optional): Triton kernel + roofline | [docs/08-gpu-track.md](docs/08-gpu-track.md) | `pytest tests/gpu` (Colab works) |

Start with [docs/00-overview.md](docs/00-overview.md). Work in order; every
milestone keeps the earlier tests green.

```bash
grep -R "TODO(student)" tiny_vllm/        # what is left
pytest -m m3                              # one milestone, prints a progress checklist
pytest -m "m1 and not hf"                 # offline
pytest                                    # everything
```

When a milestone's tests pass you get:

```text
========================================
MILESTONE 3 COMPLETE
Block-Based KV Cache
========================================

✓ 3.1  global KV pool works
✓ 3.2  block allocation works
✓ 3.3  block reclamation works
✓ 3.4  block table works
✓ 3.5  slot mapping works
✓ 3.6  physical KV writes are correct
✓ integration test

Next:
Use the block table to perform attention
over physically scattered KV.
```

## Layout

```text
tiny_vllm/
  config.py       TinyLlamaConfig (provided)
  layers.py       RMSNorm, RoPE, MLP                         M1
  attention.py    causal / contiguous / paged attention      M1 M2 M4 M5
  model.py        DecoderLayer, TinyLlama                    M1
  loader.py       HF config/tokenizer/weights                M1
  sampler.py      greedy sampling                            M2
  generation.py   naive / cached / paged generation          M2 M4
  kv_cache.py     ContiguousKVCache, PagedKVCache, slot map  M2 M3
  block_pool.py   BlockPool                                  M3
  request.py      Request                                    M5
  scheduler.py    Scheduler                                  M5
  engine.py       LLMEngine                                  M5 M6
  visualize.py    block-pool pictures (provided)
  kernels/        optional GPU track: Triton decode kernel      (docs/08)
tests/            one file per concern, markers m1..m6, checkpoint ids; tests/gpu self-skips without CUDA
benchmarks/       benchmark_kv_cache.py, benchmark_serving.py, benchmark_attention_backends.py, run_all_experiments.py
examples/         hf_reference.py, generate.py, multiple_requests.py, visualize_blocks.py
docs/             00-overview … 07-real-vllm, 08-gpu-track (optional)
```

## Definition of done

```python
from tiny_vllm.engine import LLMEngine
from tiny_vllm.loader import load_tiny_llama

model, tokenizer = load_tiny_llama()
engine = LLMEngine(model, tokenizer, num_blocks=64, block_size=16, max_num_seqs=4)
engine.add_request("Explain virtual memory in one sentence.", max_tokens=32)
engine.add_request("What is paged attention?", max_tokens=32)
for request in engine.run():
    print(request.request_id, engine.decode_output(request))
```

Internally this must run: multiple requests → scheduler → block allocation → block
tables → slot mappings → TinyLlama → Q/K/V → paged KV cache → paged attention →
sampling → tokens. `python benchmarks/run_all_experiments.py` reports the four
experiments (model correctness, KV cache, paged correctness, block reuse).

## Non-goals

Tensor/pipeline parallelism, quantization, LoRA, speculative decoding, prefix
caching, chunked prefill, multimodal, beam search, an API server, multiple
architectures. GPU kernels appear only in the optional GPU track; the course itself
never requires CUDA. See [docs/07-real-vllm.md](docs/07-real-vllm.md) for how
real vLLM covers these.
