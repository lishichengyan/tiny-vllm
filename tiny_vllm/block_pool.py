"""Physical KV block management (Milestone 3, checkpoints 3.2 and 3.3).

A physical block is just an integer id in ``range(num_blocks)``. The pool
tracks which ids are free. It knows nothing about tensors (that is
``PagedKVCache``) and nothing about requests (that is the block table).

    0 FREE
    1 USED
    2 USED
    3 FREE
    ...

Required behaviour:
    * ``allocate`` hands out a block that is currently free and marks it used.
    * A block is owned by at most one live request at a time.
    * ``free`` returns a block to the pool so it can be handed out again.
    * Exhaustion is explicit: ``allocate`` raises ``OutOfBlocksError`` instead of
      returning garbage.
    * Freeing a block that is not currently allocated (double free, unknown id)
      raises ``ValueError``.

The allocation *order* is not specified. Blocks given to one request do not have
to be contiguous.
"""


class OutOfBlocksError(RuntimeError):
    """Raised when a block is requested but no physical block is free."""


class BlockPool:
    def __init__(self, num_blocks: int):
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        self.num_blocks = num_blocks
        # TODO(student): Milestone 3 -- initialise your free-block bookkeeping.
        raise NotImplementedError("Milestone 3 (3.2): initialise BlockPool")

    @property
    def num_free_blocks(self) -> int:
        # TODO(student): Milestone 3
        raise NotImplementedError("Milestone 3 (3.2): implement BlockPool.num_free_blocks")

    def allocate(self) -> int:
        """Return the id of a free physical block and mark it as used.

        Raises:
            OutOfBlocksError: if no block is free.
        """
        # TODO(student): Milestone 3
        raise NotImplementedError("Milestone 3 (3.2): implement BlockPool.allocate()")

    def free(self, block_id: int) -> None:
        """Return ``block_id`` to the pool.

        Raises:
            ValueError: if ``block_id`` is out of range or is not currently allocated.
        """
        # TODO(student): Milestone 3
        raise NotImplementedError("Milestone 3 (3.3): implement BlockPool.free()")

    def __repr__(self) -> str:
        try:
            free = self.num_free_blocks
        except NotImplementedError:
            free = "?"
        return f"BlockPool(num_blocks={self.num_blocks}, num_free_blocks={free})"
