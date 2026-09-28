"""Experiment B -- No KV cache vs KV cache (Milestone 2).

Measures, on the real model:

    TTFT   time to first token  = the prefill forward pass
    TPOT   time per output token = mean latency of the following decode steps
    total  wall time for the whole generation

The two strategies must produce identical tokens (greedy decoding).

    python benchmarks/benchmark_kv_cache.py --max-new-tokens 32
"""

import argparse
import time

import torch

from tiny_vllm.generation import generate_naive, generate_with_kv_cache
from tiny_vllm.loader import DEFAULT_MODEL, load_tiny_llama


class StepTimer(torch.nn.Module):
    """Wraps the model and records the wall time of every forward call."""

    def __init__(self, model):
        super().__init__()
        self.inner = model
        self.times: list[float] = []

    @property
    def config(self):
        return self.inner.config

    @property
    def device(self):
        return self.inner.device

    @property
    def dtype(self):
        return self.inner.dtype

    def forward(self, *args, **kwargs):
        t0 = time.perf_counter()
        out = self.inner(*args, **kwargs)
        self.times.append(time.perf_counter() - t0)
        return out


def run(name, fn, model, prompt_tokens, max_new_tokens, eos):
    timed = StepTimer(model)
    t0 = time.perf_counter()
    tokens = fn(timed, prompt_tokens, max_new_tokens, eos)
    total = time.perf_counter() - t0
    ttft = timed.times[0] * 1000
    decode = timed.times[1:]
    tpot = (sum(decode) / len(decode) * 1000) if decode else float("nan")
    print(
        f"{name:<12} tokens={len(tokens):<3} TTFT={ttft:7.1f} ms   TPOT={tpot:7.1f} ms   total={total:6.2f} s"
    )
    if decode:
        first, last = decode[0] * 1000, decode[-1] * 1000
        print(f"{'':<12} first decode step {first:6.1f} ms  ->  last decode step {last:6.1f} ms")
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default="Virtual memory is a technique that")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    model, tokenizer = load_tiny_llama(args.model)
    prompt_tokens = tokenizer(args.prompt).input_ids
    eos = tokenizer.eos_token_id
    print(
        f"model: {args.model}   prompt tokens: {len(prompt_tokens)}   max_new_tokens: {args.max_new_tokens}"
    )
    print()

    naive = run("no KV cache", generate_naive, model, prompt_tokens, args.max_new_tokens, eos)
    cached = run("KV cache", generate_with_kv_cache, model, prompt_tokens, args.max_new_tokens, eos)
    print()
    print("identical tokens:", naive == cached)
    print("output:", repr(tokenizer.decode(cached)))
    print()
    print("Note: TTFT is the same for both -- a KV cache does not speed up the first token,")
    print("it avoids recomputing the past on every decode step, so TPOT stays flat.")


if __name__ == "__main__":
    main()
