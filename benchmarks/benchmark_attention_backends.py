"""Compare decode-attention backends over the paged KV cache and report roofline numbers.

Backends:
    gather      paged_attention          (4.3)  gather blocks into a contiguous copy, then SDPA
    blockwise   paged_attention_decode   (4.6)  online softmax over blocks, no copy (PyTorch loop)
    triton      paged_attention_decode_triton   (GPU track) same algorithm as one kernel
    contiguous  causal_attention on already-contiguous K/V -- the "no paging at all" ceiling

For every (backend, seq_len) it reports the time per decode step for a whole batch,
the KV bytes that step *must* read (K and V of every token of every sequence, once),
and the achieved bandwidth bytes / time. Compare that with your GPU's peak memory
bandwidth: decode attention is memory-bound, so achieved / peak is the roofline score.

    python benchmarks/benchmark_attention_backends.py                       # CPU, small
    python benchmarks/benchmark_attention_backends.py --device cuda --dtype float16 \
        --batch 32 --seq-lens 128 512 2048 8192 --block-size 16
"""

import argparse
import math
import time

import torch

from tiny_vllm.attention import causal_attention, paged_attention, paged_attention_decode
from tiny_vllm.config import TinyLlamaConfig
from tiny_vllm.kernels import gpu_track_available
from tiny_vllm.kv_cache import PagedKVCache, compute_slot_mapping


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def timeit(fn, device, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(device)
    return (time.perf_counter() - t0) / iters


class Scenario:
    """A batch of same-length sequences scattered through a paged pool, plus every backend."""

    def __init__(self, config, batch, seq_len, block_size, num_blocks, dtype, device, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.batch, self.seq_len, self.device = batch, seq_len, device
        self.cache = PagedKVCache(config, num_blocks, block_size, dtype=dtype, device=device)
        perm = torch.randperm(num_blocks, generator=g).tolist()
        self.tables, self.ks, self.vs = [], [], []
        needed = math.ceil(seq_len / block_size)
        for _ in range(batch):
            table, perm = perm[:needed], perm[needed:]
            k = torch.randn(seq_len, config.num_kv_heads, config.head_dim, generator=g).to(dtype)
            v = torch.randn(seq_len, config.num_kv_heads, config.head_dim, generator=g).to(dtype)
            slots = compute_slot_mapping(table, torch.arange(seq_len), block_size).to(device)
            self.cache.write(0, k.to(device), v.to(device), slots)
            self.tables.append(table)
            self.ks.append(k.to(device))
            self.vs.append(v.to(device))
        self.q = torch.randn(batch, config.num_heads, config.head_dim, generator=g).to(dtype).to(device)
        self.k_cache, self.v_cache = self.cache.k_cache[0], self.cache.v_cache[0]
        self.block_tables_tensor = None
        self.seq_lens_tensor = None

    def run_gather(self):
        return [
            paged_attention(self.q[b][None], self.k_cache, self.v_cache, self.tables[b], self.seq_len)
            for b in range(self.batch)
        ]

    def run_blockwise(self):
        return [
            paged_attention_decode(self.q[b], self.k_cache, self.v_cache, self.tables[b], self.seq_len)
            for b in range(self.batch)
        ]

    def run_contiguous(self):
        return [
            causal_attention(self.q[b][None, None], self.ks[b][None], self.vs[b][None])
            for b in range(self.batch)
        ]

    def run_triton(self):
        from tiny_vllm.kernels.paged_attention_triton import (
            build_block_tables_tensor,
            paged_attention_decode_triton,
        )

        if self.block_tables_tensor is None:
            self.block_tables_tensor = build_block_tables_tensor(self.tables, self.device)
            self.seq_lens_tensor = torch.full(
                (self.batch,), self.seq_len, dtype=torch.int32, device=self.device
            )
        return paged_attention_decode_triton(
            self.q.contiguous(), self.k_cache, self.v_cache, self.block_tables_tensor, self.seq_lens_tensor
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[64, 256, 1024])
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=9)
    parser.add_argument("--num-kv-heads", type=int, default=3)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument(
        "--peak-gbps",
        type=float,
        default=None,
        help="your device's peak memory bandwidth, for the roofline column",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    config = TinyLlamaConfig(
        vocab_size=1,
        hidden_size=args.num_heads * args.head_dim,
        intermediate_size=1,
        num_layers=1,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
    )
    use_triton = device.type == "cuda" and gpu_track_available()
    if use_triton:
        from tiny_vllm.kernels.paged_attention_triton import KERNEL_IMPLEMENTED

        use_triton = KERNEL_IMPLEMENTED
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()

    print(
        f"device={device} dtype={args.dtype} batch={args.batch} heads={args.num_heads}/{args.num_kv_heads} "
        f"head_dim={args.head_dim} block_size={args.block_size} triton={'yes' if use_triton else 'no'}"
    )
    header = f"{'seq_len':>8} {'backend':>11} {'ms/step':>9} {'KV MB read':>11} {'GB/s':>8}"
    if args.peak_gbps:
        header += f" {'% of peak':>10}"
    print(header)

    for seq_len in args.seq_lens:
        num_blocks = args.batch * math.ceil(seq_len / args.block_size) + 8
        scenario = Scenario(config, args.batch, seq_len, args.block_size, num_blocks, dtype, device)
        kv_bytes = args.batch * seq_len * args.num_kv_heads * args.head_dim * 2 * bytes_per_elem

        backends = {
            "gather": scenario.run_gather,
            "blockwise": scenario.run_blockwise,
            "contiguous": scenario.run_contiguous,
        }
        if use_triton:
            backends["triton"] = scenario.run_triton

        for name, fn in backends.items():
            try:
                seconds = timeit(fn, device)
            except NotImplementedError as exc:
                print(f"{seq_len:>8} {name:>11}   not implemented: {str(exc).splitlines()[0][:60]}")
                continue
            gbps = kv_bytes / seconds / 1e9
            line = f"{seq_len:>8} {name:>11} {seconds * 1000:9.3f} {kv_bytes / 1e6:11.2f} {gbps:8.1f}"
            if args.peak_gbps:
                line += f" {100 * gbps / args.peak_gbps:9.1f}%"
            print(line)
        print()

    print("KV MB read = batch * seq_len * num_kv_heads * head_dim * 2 (K and V) * bytes per element:")
    print("the minimum traffic a decode step needs. GB/s = that / measured time. On a GPU compare with")
    print("the device's peak HBM bandwidth (--peak-gbps); the gap is the roofline headroom.")


if __name__ == "__main__":
    main()
