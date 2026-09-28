"""Draw the block-pool picture from block tables (Milestone 3 visualization).

Works before any milestone is implemented: it only formats numbers. Edit the tables
below to build intuition for block tables, partial blocks and fragmentation.

    python examples/visualize_blocks.py
"""

from tiny_vllm.visualize import format_kv_pool

BLOCK_SIZE = 4
NUM_BLOCKS = 8

# The example from the plan: A holds 8 tokens in physical blocks 7 and 2, B holds 4 in block 5.
print(format_kv_pool(NUM_BLOCKS, BLOCK_SIZE, {"A": [7, 2], "B": [5]}, {"A": 8, "B": 4}, title="t0"))
print()

# B finishes: its block is free again. A has grown by two tokens into a third block.
print(format_kv_pool(NUM_BLOCKS, BLOCK_SIZE, {"A": [7, 2, 0]}, {"A": 10}, title="t1  (B finished)"))
print()

# D is admitted and receives B's old block. Nothing had to be contiguous at any point.
print(
    format_kv_pool(
        NUM_BLOCKS, BLOCK_SIZE, {"A": [7, 2, 0], "D": [5]}, {"A": 10, "D": 3}, title="t2  (D admitted)"
    )
)
