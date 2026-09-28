"""Debug/visualization helpers for the block-based KV cache.

Provided infrastructure (not a learner exercise). Nothing here touches tensors or
the block pool's internals: the picture is derived purely from block tables and
sequence lengths, so it also works as an independent cross-check of the pool.

Example output::

    Request A
    block_table = [7, 2]

    Request B
    block_table = [5]

    Physical KV Pool (num_blocks=8, block_size=4)

    0 FREE
    1 FREE
    2 A [tokens 4-5]
    3 FREE
    4 FREE
    5 B [tokens 0-3]
    6 FREE
    7 A [tokens 0-3]

    free = [0, 1, 3, 4, 6]
"""


def format_kv_pool(
    num_blocks: int,
    block_size: int,
    block_tables: dict[str, list[int]],
    seq_lens: dict[str, int] | None = None,
    title: str | None = None,
) -> str:
    """Render request block tables and the physical pool as text.

    Args:
        num_blocks:   size of the physical pool.
        block_size:   tokens per block.
        block_tables: request name -> block table (list of physical block ids).
        seq_lens:     request name -> number of tokens the request currently holds.
                      Used to label blocks with the token range they contain. Optional.
        title:        optional first line.

    Raises:
        ValueError: if two requests claim the same physical block or an id is out of range.
    """
    seq_lens = seq_lens or {}
    owner: dict[int, tuple[str, int]] = {}  # physical block -> (request name, logical block index)
    for name, table in block_tables.items():
        for logical_idx, physical in enumerate(table):
            if not 0 <= physical < num_blocks:
                raise ValueError(
                    f"request {name!r}: physical block {physical} out of range [0, {num_blocks})"
                )
            if physical in owner:
                other, _ = owner[physical]
                raise ValueError(f"physical block {physical} is claimed by both {other!r} and {name!r}")
            owner[physical] = (name, logical_idx)

    lines: list[str] = []
    if title:
        lines += [title, ""]
    for name, table in block_tables.items():
        lines += [f"Request {name}", f"block_table = {list(table)}", ""]

    lines.append(f"Physical KV Pool (num_blocks={num_blocks}, block_size={block_size})")
    lines.append("")
    free: list[int] = []
    for physical in range(num_blocks):
        if physical not in owner:
            lines.append(f"{physical} FREE")
            free.append(physical)
            continue
        name, logical_idx = owner[physical]
        label = f"{physical} {name}"
        if name in seq_lens:
            first = logical_idx * block_size
            last = min((logical_idx + 1) * block_size, seq_lens[name]) - 1
            if last < first:
                label += " [empty]"
            else:
                label += f" [tokens {first}-{last}]"
        lines.append(label)
    lines.append("")
    lines.append(f"free = {free}")
    return "\n".join(lines)
