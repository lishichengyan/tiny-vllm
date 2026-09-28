"""Generate text with TinyLlama using one of the single-request strategies.

python examples/generate.py --mode naive     # Milestone 2.1
python examples/generate.py --mode cached    # Milestone 2.5
python examples/generate.py --mode paged     # Milestone 4.5
"""

import argparse
import time

from tiny_vllm.block_pool import BlockPool
from tiny_vllm.generation import generate_naive, generate_paged, generate_with_kv_cache
from tiny_vllm.kv_cache import PagedKVCache
from tiny_vllm.loader import DEFAULT_MODEL, load_tiny_llama


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--mode", choices=["naive", "cached", "paged"], default="cached")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=64)
    args = parser.parse_args()

    model, tokenizer = load_tiny_llama(args.model)
    prompt_tokens = tokenizer(args.prompt).input_ids
    eos = tokenizer.eos_token_id
    print(f"model: {args.model}   mode: {args.mode}")
    print(f"prompt: {args.prompt!r}  ({len(prompt_tokens)} tokens)")

    t0 = time.perf_counter()
    if args.mode == "naive":
        tokens = generate_naive(model, prompt_tokens, args.max_new_tokens, eos)
    elif args.mode == "cached":
        tokens = generate_with_kv_cache(model, prompt_tokens, args.max_new_tokens, eos)
    else:
        kv_cache = PagedKVCache(model.config, args.num_blocks, args.block_size)
        pool = BlockPool(args.num_blocks)
        tokens = generate_paged(model, kv_cache, pool, prompt_tokens, args.max_new_tokens, eos)
        print(f"blocks free after generation: {pool.num_free_blocks}/{args.num_blocks}")
    elapsed = time.perf_counter() - t0

    print(
        f"generated {len(tokens)} tokens in {elapsed:.2f}s ({elapsed / max(len(tokens), 1) * 1000:.1f} ms/token)"
    )
    print("output:", repr(tokenizer.decode(tokens)))


if __name__ == "__main__":
    main()
