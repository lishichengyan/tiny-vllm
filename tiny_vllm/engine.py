"""LLMEngine: the multi-request, continuous-batching inference engine (Milestones 5 and 6).

    add_request()  ->  scheduler.waiting
                              │
             ┌────────────────┴──────────── step() ────────────────────┐
             │  batch, is_prefill = scheduler.schedule()               │
             │  prefill new requests  /  decode all running requests   │
             │       allocate blocks -> slot mapping -> model -> sample│
             │  stop finished requests, free their blocks              │
             └───────────────────────────────────────────────────────-─┘

Shared between all requests:  model, kv_cache (PagedKVCache), block_pool.
Owned by each request:        prompt/generated tokens, status, block_table.
"""

import torch

from .block_pool import BlockPool
from .kv_cache import PagedKVCache
from .model import TinyLlama
from .request import Request
from .scheduler import Scheduler
from .visualize import format_kv_pool


class LLMEngine:
    def __init__(
        self,
        model: TinyLlama,
        tokenizer=None,
        num_blocks: int = 256,
        block_size: int = 16,
        max_num_seqs: int = 8,
        eos_token_id: int | None = None,
        log_blocks: bool = False,
    ):
        """
        Args:
            model:        a TinyLlama in eval mode (weights loaded).
            tokenizer:    HuggingFace tokenizer; only needed if prompts are passed as strings.
            num_blocks:   size of the global physical KV pool.
            block_size:   tokens per block.
            max_num_seqs: maximum number of concurrently RUNNING requests.
            eos_token_id: overrides ``model.config.eos_token_id``.
            log_blocks:   print the block-pool picture after every step of ``run()``.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = model.config
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        self.log_blocks = log_blocks

        # Shared resources
        self.kv_cache = PagedKVCache(
            self.config, num_blocks, block_size, dtype=model.dtype, device=model.device
        )
        self.block_pool = BlockPool(num_blocks)
        self.scheduler = Scheduler(self.block_pool, block_size, max_num_seqs)

        self.requests: dict[int, Request] = {}
        self._next_request_id = 0
        self.num_steps = 0

    # ------------------------------------------------------------ provided
    def add_request(self, prompt: str | list[int], max_tokens: int = 16) -> Request:
        """Create a request and hand it to the scheduler. Returns the Request object."""
        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise ValueError("a tokenizer is required to accept string prompts")
            prompt_tokens = list(self.tokenizer(prompt).input_ids)
        else:
            prompt_tokens = list(prompt)
        if len(prompt_tokens) == 0:
            raise ValueError("prompt must contain at least one token")
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        worst_case = len(prompt_tokens) + max_tokens
        if worst_case > self.config.max_position_embeddings:
            raise ValueError(
                f"prompt ({len(prompt_tokens)}) + max_tokens ({max_tokens}) exceeds "
                f"max_position_embeddings ({self.config.max_position_embeddings})"
            )
        if worst_case > self.num_blocks * self.block_size:
            raise ValueError(
                f"prompt ({len(prompt_tokens)}) + max_tokens ({max_tokens}) can never fit in "
                f"{self.num_blocks} blocks of {self.block_size} tokens"
            )
        request = Request(self._next_request_id, prompt_tokens, max_tokens)
        self._next_request_id += 1
        self.requests[request.request_id] = request
        self.scheduler.add_request(request)
        return request

    def has_work(self) -> bool:
        return self.scheduler.has_unfinished_requests()

    def run(self):
        """Drive ``step()`` until every request is finished, yielding requests as they finish."""
        while self.has_work():
            finished = self.step()
            if self.log_blocks:
                print(self.format_state())
            yield from finished

    def decode_output(self, request: Request) -> str:
        """Detokenize a request's generated tokens."""
        if self.tokenizer is None:
            raise ValueError("a tokenizer is required to decode text")
        return self.tokenizer.decode(request.generated_tokens, skip_special_tokens=True)

    def format_state(self) -> str:
        """Text picture of who owns which physical block (see ``visualize.format_kv_pool``)."""
        running = self.scheduler.running
        block_tables = {str(r.request_id): list(r.block_table) for r in running}
        seq_lens = {str(r.request_id): r.num_tokens for r in running}
        waiting = [r.request_id for r in self.scheduler.waiting]
        header = (
            f"step {self.num_steps}: running={[r.request_id for r in running]} waiting={waiting} "
            f"pool free={self.block_pool.num_free_blocks}/{self.num_blocks}"
        )
        return format_kv_pool(self.num_blocks, self.block_size, block_tables, seq_lens, title=header)

    # ------------------------------------------------------------ learner
    def step(self) -> list[Request]:
        """Execute one engine iteration and return the requests that finished during it.

        Milestone 5 (5.4):
            1. ``batch, is_prefill = self.scheduler.schedule()``; return ``[]`` if empty.
            2. Prefill every request of a prefill batch (``self._prefill``) or run one
               batched decode step for a decode batch (``self._decode``).
            3. For every request in the batch whose ``should_stop`` is now True:
               ``self.scheduler.finish(request)``.
            4. Increment ``self.num_steps``.
        Milestone 6 (6.1, 6.4):
            5. Reclaim the blocks of every finished request (``self._free_request_blocks``)
               so they can be reused by the next admitted request.
        """
        # TODO(student): Milestone 5 / Milestone 6
        raise NotImplementedError("Milestone 5 (5.4): implement LLMEngine.step()")

    @torch.no_grad()
    def _prefill(self, request: Request) -> None:
        """Run a freshly admitted request's prompt and sample its first token (5.4, 5.5).

        * Allocate blocks for the prompt into ``request.block_table``.
        * Build ``AttentionMetadata`` (slot mapping for positions ``0 .. num_prompt_tokens-1``,
          the request's block table, ``seq_lens=[num_prompt_tokens]``).
        * ``input_ids`` / ``positions`` have shape [1, num_prompt_tokens].
        * Sample from the last position's logits and ``request.append_token(...)``.
        """
        # TODO(student): Milestone 5
        raise NotImplementedError("Milestone 5 (5.4/5.5): implement LLMEngine._prefill()")

    @torch.no_grad()
    def _decode(self, requests: list[Request]) -> None:
        """One batched decode step for all running requests (5.4, 5.5, 6.3).

        Every request contributes exactly ONE token: its ``last_token`` at position
        ``num_tokens - 1`` (its K/V is not in the cache yet).

        * Make sure each request's block table covers that position (dynamic block
          allocation: blocks are appended only when a sequence crosses a block boundary).
        * ``input_ids`` / ``positions`` have shape [batch, 1]; the metadata has one slot per
          request, one block table per request and ``seq_lens[b] = requests[b].num_tokens``.
        * Run ONE forward pass, sample one token per request, append it.
        """
        # TODO(student): Milestone 5
        raise NotImplementedError("Milestone 5 (5.4/5.5): implement LLMEngine._decode()")

    def _free_request_blocks(self, request: Request) -> None:
        """Return all of a finished request's blocks to the pool and clear its block table (6.4)."""
        # TODO(student): Milestone 6
        raise NotImplementedError("Milestone 6 (6.4): implement LLMEngine._free_request_blocks()")
