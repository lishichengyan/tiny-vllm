"""Scheduler (Milestone 5, checkpoints 5.2 and 5.3; Milestone 6, checkpoint 6.2).

The scheduler decides which requests take part in the next engine step. It is
deliberately simple -- no priorities, no preemption, no chunked prefill:

    waiting  (FIFO)                running
    ┌───┬───┬───┐                  ┌───┬───┐
    │ C │ D │ E │  ── admit ──►    │ A │ B │  ── finish ──►  FINISHED
    └───┴───┴───┘                  └───┴───┘

Every call to ``schedule()`` returns ONE of two kinds of batch:

    * a **prefill batch**: the requests admitted *by this call*, which have no KV yet.
      The engine prefills each of them (one forward pass per request).
    * a **decode batch**: all currently running requests, each contributing one new
      token to a single batched forward pass.

Admission policy (``can_admit``) -- simple and safe:

    1. ``len(running) < max_num_seqs``
    2. the *block budget* covers the request's worst case::

           worst_case(r) = blocks_needed(r.num_prompt_tokens + r.max_tokens, block_size)
           promised      = sum(worst_case(r) - len(r.block_table) for r in running)
           budget        = block_pool.num_free_blocks - promised
           admit iff     budget >= worst_case(request)

       Blocks are allocated lazily (a request only holds blocks for tokens it has actually
       produced), so free blocks are not the same as *available* blocks: some of them are
       already promised to running requests that will grow into them.

Rule 2 guarantees that a running request can always get a block when its sequence
crosses a block boundary, so the engine never has to preempt or evict. (Real vLLM
admits more optimistically and preempts when memory runs out; see docs/07-real-vllm.md.)

Admission is strictly FIFO: if the request at the head of the waiting queue cannot be
admitted, nothing behind it is admitted either.
"""

from .block_pool import BlockPool
from .kv_cache import blocks_needed  # noqa: F401  (used by your can_admit)
from .request import Request, RequestStatus  # noqa: F401  (RequestStatus is used by your transitions)


class Scheduler:
    def __init__(self, block_pool: BlockPool, block_size: int, max_num_seqs: int):
        if max_num_seqs < 1:
            raise ValueError("max_num_seqs must be >= 1")
        self.block_pool = block_pool
        self.block_size = block_size
        self.max_num_seqs = max_num_seqs
        self.waiting: list[Request] = []
        self.running: list[Request] = []

    # ------------------------------------------------------------ 5.2
    def add_request(self, request: Request) -> None:
        """Enqueue a new request. It must be (and stay) WAITING until admitted."""
        # TODO(student): Milestone 5
        raise NotImplementedError("Milestone 5 (5.2): implement Scheduler.add_request()")

    def has_unfinished_requests(self) -> bool:
        """True while any request is waiting or running."""
        # TODO(student): Milestone 5
        raise NotImplementedError("Milestone 5 (5.2): implement Scheduler.has_unfinished_requests()")

    def finish(self, request: Request) -> None:
        """Move a running request to FINISHED and remove it from ``running``."""
        # TODO(student): Milestone 5
        raise NotImplementedError("Milestone 5 (5.2): implement Scheduler.finish()")

    # ------------------------------------------------------------ 5.3 / 6.2
    def can_admit(self, request: Request) -> bool:
        """Apply the admission policy described in the module docstring."""
        # TODO(student): Milestone 5
        raise NotImplementedError("Milestone 5 (5.3): implement Scheduler.can_admit()")

    def schedule(self) -> tuple[list[Request], bool]:
        """Decide what the next engine step executes.

        Returns:
            (batch, is_prefill)
            * If at least one waiting request can be admitted: admit as many as the policy
              allows (FIFO, in order), mark them RUNNING, append them to ``running`` and
              return ``(admitted_requests, True)``.
            * Otherwise return ``(list(running), False)`` -- possibly an empty list.
        """
        # TODO(student): Milestone 5
        raise NotImplementedError("Milestone 5 (5.3): implement Scheduler.schedule()")

    def __repr__(self) -> str:
        return (
            f"Scheduler(waiting={[r.request_id for r in self.waiting]}, "
            f"running={[r.request_id for r in self.running]}, max_num_seqs={self.max_num_seqs})"
        )
