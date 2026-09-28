"""Serve several prompts at once with LLMEngine (Milestones 5 and 6).

python examples/multiple_requests.py
python examples/multiple_requests.py --log-blocks      # print the block pool after every step
python examples/multiple_requests.py --max-num-seqs 2  # force requests to wait for a seat
"""

import argparse
import time

from tiny_vllm.engine import LLMEngine
from tiny_vllm.loader import DEFAULT_MODEL, load_tiny_llama

PROMPTS = [
    ("Explain virtual memory in one sentence.", 24),
    ("What is paged attention?", 12),
    ("def fibonacci(n):", 32),
    ("The three primary colors are", 8),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num-blocks", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--log-blocks", action="store_true")
    args = parser.parse_args()

    model, tokenizer = load_tiny_llama(args.model)
    engine = LLMEngine(
        model,
        tokenizer,
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        max_num_seqs=args.max_num_seqs,
        log_blocks=args.log_blocks,
    )
    for prompt, max_tokens in PROMPTS:
        req = engine.add_request(prompt, max_tokens=max_tokens)
        print(f"added request {req.request_id}: {prompt!r} (max_tokens={max_tokens})")
    print()

    t0 = time.perf_counter()
    for request in engine.run():
        print(
            f"[step {engine.num_steps}] request {request.request_id} finished after "
            f"{len(request.generated_tokens)} tokens: {engine.decode_output(request)!r}"
        )
    elapsed = time.perf_counter() - t0

    total_tokens = sum(len(r.generated_tokens) for r in engine.requests.values())
    print()
    print(
        f"{len(PROMPTS)} requests, {total_tokens} generated tokens, {engine.num_steps} engine steps, {elapsed:.2f}s"
    )
    print(f"pool: {engine.block_pool.num_free_blocks}/{args.num_blocks} blocks free")


if __name__ == "__main__":
    main()
