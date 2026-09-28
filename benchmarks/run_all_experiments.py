"""Final report: the four experiments of the project on the real model.

A  Model correctness      HuggingFace logits  ≈  TinyLlama logits              (Milestone 1)
B  KV cache               uncached vs cached generation, TPOT                  (Milestone 2)
C  Paged KV correctness   contiguous attention ≈ paged attention, [7, 2, 11]   (Milestone 4)
D  Serving / block reuse  request finishes -> blocks freed -> reused           (Milestone 6)

python benchmarks/run_all_experiments.py            # everything
python benchmarks/run_all_experiments.py --only A   # after Milestone 1
python benchmarks/run_all_experiments.py --only AB
"""

import argparse
import time

import torch

from tiny_vllm.block_pool import BlockPool
from tiny_vllm.engine import LLMEngine
from tiny_vllm.generation import generate_naive, generate_paged, generate_with_kv_cache
from tiny_vllm.kv_cache import AttentionMetadata, PagedKVCache, compute_slot_mapping
from tiny_vllm.loader import DEFAULT_MODEL, load_hf_model, load_tiny_llama

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n <",
    "Paged attention lets the KV cache",
]


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def experiment_a(model, tokenizer, model_name):
    banner("Experiment A -- Model correctness: HuggingFace ≈ TinyLlama")
    hf = load_hf_model(model_name)
    for prompt in PROMPTS:
        ids = tokenizer(prompt, return_tensors="pt").input_ids
        positions = torch.arange(ids.shape[1])[None]
        with torch.no_grad():
            hf_logits = hf(ids).logits
            tiny_logits = model(ids, positions)
        diff = (hf_logits - tiny_logits).abs().max().item()
        same_argmax = bool((hf_logits.argmax(-1) == tiny_logits.argmax(-1)).all())
        print(f"{prompt!r:<45} max |Δlogit| = {diff:.2e}   argmax identical at every position: {same_argmax}")


def experiment_b(model, tokenizer, max_new_tokens=24):
    banner("Experiment B -- KV cache: uncached vs cached generation")
    prompt_tokens = tokenizer(PROMPTS[0]).input_ids
    eos = tokenizer.eos_token_id

    class Timer(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner, self.times = inner, []

        config = property(lambda self: self.inner.config)
        device = property(lambda self: self.inner.device)
        dtype = property(lambda self: self.inner.dtype)

        def forward(self, *a, **k):
            t = time.perf_counter()
            out = self.inner(*a, **k)
            self.times.append(time.perf_counter() - t)
            return out

    results = {}
    for name, fn in (("uncached", generate_naive), ("cached", generate_with_kv_cache)):
        timed = Timer(model)
        t0 = time.perf_counter()
        tokens = fn(timed, prompt_tokens, max_new_tokens, eos)
        total = time.perf_counter() - t0
        decode = timed.times[1:]
        tpot = sum(decode) / len(decode) * 1000 if decode else float("nan")
        results[name] = tokens
        print(
            f"{name:<9} TTFT={timed.times[0] * 1000:6.1f} ms   TPOT={tpot:6.1f} ms   "
            f"last decode step={decode[-1] * 1000 if decode else 0:6.1f} ms   total={total:5.2f} s"
        )
    print(f"identical tokens: {results['uncached'] == results['cached']}")
    print(f"output: {tokenizer.decode(results['cached'])!r}")
    print("TTFT is unchanged by the cache; TPOT stops growing with the sequence length.")


def experiment_c(model, tokenizer):
    banner("Experiment C -- Paged KV correctness with block_table = [7, 2, 11]")
    block_size = 4
    prompt_tokens = tokenizer(PROMPTS[2]).input_ids[: 3 * block_size - 2]  # 10 tokens -> partial last block
    ids = torch.tensor([prompt_tokens])
    positions = torch.arange(len(prompt_tokens))[None]
    block_table = [7, 2, 11]
    cache = PagedKVCache(model.config, num_blocks=12, block_size=block_size)
    metadata = AttentionMetadata(
        slot_mapping=compute_slot_mapping(block_table, positions[0], block_size),
        block_tables=[block_table],
        seq_lens=[len(prompt_tokens)],
    )
    with torch.no_grad():
        contiguous = model(ids, positions)
        paged = model(ids, positions, kv_cache=cache, attn_metadata=metadata)
    print(f"prompt tokens: {len(prompt_tokens)}   block_size: {block_size}   block_table: {block_table}")
    print(f"max |contiguous - paged| logits = {(contiguous - paged).abs().max().item():.2e}")

    pool = BlockPool(12)
    for b in [pool.allocate() for _ in range(12)]:
        if b % 2 == 0:
            pool.free(b)  # only even blocks free -> forced fragmentation
    eos = tokenizer.eos_token_id
    paged_tokens = generate_paged(
        model, PagedKVCache(model.config, 12, block_size), pool, prompt_tokens, 16, eos
    )
    naive_tokens = generate_naive(model, prompt_tokens, 16, eos)
    print(f"fragmented-pool generation == naive generation: {paged_tokens == naive_tokens}")
    print(f"output: {tokenizer.decode(paged_tokens)!r}")


def experiment_d(model, tokenizer):
    banner("Experiment D -- Serving: finish -> free -> admit -> reuse")
    requests = [
        ("Explain virtual memory in one sentence.", 20),
        ("What is paged attention?", 3),
        ("def fibonacci(n):", 14),
        ("The three primary colors are", 6),
    ]
    num_blocks, block_size = 10, 8
    engine = LLMEngine(model, tokenizer, num_blocks=num_blocks, block_size=block_size, max_num_seqs=3)
    for prompt, max_tokens in requests:
        engine.add_request(prompt, max_tokens=max_tokens)

    owners_history: dict[int, list[int]] = {}
    t0 = time.perf_counter()
    while engine.has_work():
        finished = engine.step()
        for r in engine.scheduler.running:
            for blk in r.block_table:
                owners_history.setdefault(blk, [])
                if not owners_history[blk] or owners_history[blk][-1] != r.request_id:
                    owners_history[blk].append(r.request_id)
        if finished or engine.num_steps <= 2:
            print(engine.format_state())
            for r in finished:
                print(f"  -> request {r.request_id} finished: {engine.decode_output(r)!r}")
            print()
    elapsed = time.perf_counter() - t0
    reused = {blk: owners for blk, owners in owners_history.items() if len(owners) > 1}
    total_tokens = sum(len(r.generated_tokens) for r in engine.requests.values())
    print(f"{len(requests)} requests, {total_tokens} tokens, {engine.num_steps} steps, {elapsed:.2f}s")
    print(f"blocks free at the end: {engine.block_pool.num_free_blocks}/{num_blocks}")
    print("physical blocks that served more than one request (block: request ids in order):")
    for blk, owners in sorted(reused.items()):
        print(f"  block {blk}: {owners}")
    if not reused:
        print("  (none -- the pool was large enough that no block had to be recycled)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--only", default="ABCD", help="subset of experiments, e.g. 'A' or 'AC'")
    args = parser.parse_args()
    only = args.only.upper()

    model, tokenizer = load_tiny_llama(args.model)
    print(f"model: {args.model}   device: {model.device}   dtype: {model.dtype}")
    if "A" in only:
        experiment_a(model, tokenizer, args.model)
    if "B" in only:
        experiment_b(model, tokenizer)
    if "C" in only:
        experiment_c(model, tokenizer)
    if "D" in only:
        experiment_d(model, tokenizer)
    print()


if __name__ == "__main__":
    main()
