"""Sampling: turn logits into the next token (Milestone 2, checkpoint 2.1).

tiny-vllm only uses greedy decoding. It is deterministic, which makes every
comparison in the test-suite exact (uncached vs cached vs paged vs batched).
"""

import torch


def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    """Pick the highest-scoring token.

    Args:
        logits: [..., vocab_size]  (typically [vocab_size] or [batch, vocab_size])
    Returns:
        int64 tensor of shape ``logits.shape[:-1]`` with the argmax token id(s).
    """
    # TODO(student): Milestone 2
    raise NotImplementedError("Milestone 2 (2.1): implement greedy_sample()")
