"""Experiment D -- Serving / block reuse (Milestone 6).

Runs the same requests (different lengths) two ways through LLMEngine:

    sequential            max_num_seqs=1  -> one request at a time
    continuous batching   all requests share the pool and the batch

and reports engine steps, wall time and tokens per second. With --trace the block
pool picture is printed after every step so you can watch blocks being freed and
reused by later requests.

CPU numbers are modest; the point is the mechanism, not the speed-up.

    python benchmarks/benchmark_serving.py
    python benchmarks/benchmark_serving.py --trace --num-blocks 12 --block-size 8
"""

import argparse
import time

from tiny_vllm.engine import LLMEngine
from tiny_vllm.loader import DEFAULT_MODEL, load_tiny_llama

REQUESTS = [
    ("Explain virtual memory in one sentence.", 40),
    ("What is paged attention?", 6),
    ("def fibonacci(n):", 24),
    ("The three primary colors are", 10),
    ("Once upon a time", 16),
    ("Continuous batching means", 30),
]


def serve(model, tokenizer, requests, num_blocks, block_size, max_num_seqs, trace):
    engine = LLMEngine(
        model, tokenizer, num_blocks=num_blocks, block_size=block_size, max_num_seqs=max_num_seqs
    )
    for prompt, max_tokens in requests:
        engine.add_request(prompt, max_tokens=max_tokens)
    t0 = time.perf_counter()
    while engine.has_work():
        finished = engine.step()
        if trace:
            print(engine.format_state())
            for r in finished:
                print(f"  -> request {r.request_id} finished ({len(r.generated_tokens)} tokens)")
            print()
    elapsed = time.perf_counter() - t0
    outputs = {r.request_id: r.generated_tokens for r in engine.requests.values()}
    tokens = sum(len(t) for t in outputs.values())
    assert engine.block_pool.num_free_blocks == num_blocks, "blocks leaked"
    return engine.num_steps, elapsed, tokens, outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num-blocks", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--trace", action="store_true", help="print the block pool after every step")
    args = parser.parse_args()

    model, tokenizer = load_tiny_llama(args.model)
    print(
        f"model: {args.model}   {len(REQUESTS)} requests   pool: {args.num_blocks} x {args.block_size} tokens"
    )
    print()

    print("== sequential (max_num_seqs=1)")
    steps_seq, t_seq, tok_seq, out_seq = serve(
        model, tokenizer, REQUESTS, args.num_blocks, args.block_size, 1, False
    )
    print(f"steps={steps_seq}  time={t_seq:.2f}s  tokens={tok_seq}  {tok_seq / t_seq:.1f} tok/s")
    print()

    print(f"== continuous batching (max_num_seqs={args.max_num_seqs})")
    steps_cb, t_cb, tok_cb, out_cb = serve(
        model, tokenizer, REQUESTS, args.num_blocks, args.block_size, args.max_num_seqs, args.trace
    )
    print(f"steps={steps_cb}  time={t_cb:.2f}s  tokens={tok_cb}  {tok_cb / t_cb:.1f} tok/s")
    print()
    print(f"identical outputs: {out_seq == out_cb}")
    print(f"engine steps: {steps_seq} -> {steps_cb}   wall time: {t_seq:.2f}s -> {t_cb:.2f}s")


if __name__ == "__main__":
    main()
