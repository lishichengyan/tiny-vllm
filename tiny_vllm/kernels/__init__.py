"""Optional GPU track: Triton kernels (docs/08-gpu-track.md).

Nothing in the CPU course imports this package. It is safe to import on a machine
without CUDA or Triton; use ``has_triton()`` / ``has_cuda()`` before calling anything.
"""

import importlib.util

import torch


def has_triton() -> bool:
    return importlib.util.find_spec("triton") is not None


def has_cuda() -> bool:
    return torch.cuda.is_available()


def gpu_track_available() -> bool:
    return has_triton() and has_cuda()
