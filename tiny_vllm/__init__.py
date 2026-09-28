"""tiny-vllm: build vLLM from first principles.

Milestones:
    M1  TinyLlama                 layers.py, attention.py, model.py, loader.py
    M2  Generation + KV Cache     sampler.py, generation.py, kv_cache.ContiguousKVCache
    M3  Block-Based KV Cache      kv_cache.PagedKVCache, block_pool.py, slot mapping
    M4  Paged Attention           attention.gather_kv / paged_attention, generation.generate_paged
    M5  Multi-Request Engine      request.py, scheduler.py, engine.py
    M6  Continuous Batching       engine.py (block reclamation, run loop)

Find remaining work with:  grep -R "TODO(student)" tiny_vllm/
"""

from .block_pool import BlockPool, OutOfBlocksError
from .config import TinyLlamaConfig
from .engine import LLMEngine
from .kv_cache import AttentionMetadata, ContiguousKVCache, PagedKVCache
from .model import TinyLlama
from .request import Request, RequestStatus
from .scheduler import Scheduler

__all__ = [
    "AttentionMetadata",
    "BlockPool",
    "ContiguousKVCache",
    "LLMEngine",
    "OutOfBlocksError",
    "PagedKVCache",
    "Request",
    "RequestStatus",
    "Scheduler",
    "TinyLlama",
    "TinyLlamaConfig",
]
