"""Basic Llama building blocks: RMSNorm, RoPE, gated MLP.

Milestone 1, checkpoints 1.1, 1.2 and 1.4.

Shape conventions used throughout tiny-vllm:

    hidden_states: [batch, seq_len, hidden_size]
    positions:     [batch, seq_len]                 (absolute token positions)
    q:             [batch, seq_len, num_heads,    head_dim]
    k, v:          [batch, seq_len, num_kv_heads, head_dim]
    cos, sin:      [batch, seq_len, head_dim]
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square layer normalization (checkpoint 1.1).

    Normalizes each vector along the last dimension by its root-mean-square,
    then scales by a learned per-dimension ``weight``. Unlike LayerNorm there
    is no mean subtraction and no bias.

    Numerics: HuggingFace computes the normalization in float32 and casts the
    result back to the input dtype before multiplying by ``weight``. Follow the
    same order so that weights loaded from HuggingFace behave identically.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., hidden_size]  ->  [..., hidden_size]
        # TODO(student): Milestone 1
        raise NotImplementedError("Milestone 1 (1.1): implement RMSNorm.forward()")


def compute_rope_cos_sin(
    positions: torch.Tensor,
    head_dim: int,
    theta: float,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the rotary cos/sin tables for the given absolute positions (checkpoint 1.2).

    Args:
        positions: [batch, seq_len] integer tensor of absolute token positions.
            Positions do NOT have to be contiguous or start at zero: during decode a
            single new token sits at position ``seq_len - 1``.
        head_dim: size of one attention head. Must be even.
        theta: RoPE base frequency (``rope_theta`` in the HF config).

    Returns:
        (cos, sin), each of shape [batch, seq_len, head_dim], in ``dtype``.

    Convention (must match HuggingFace Llama so that pretrained weights work):
        * There are ``head_dim // 2`` rotation frequencies:
              inv_freq[i] = theta ** (-2 * i / head_dim),   i = 0 .. head_dim//2 - 1
        * The angle for position ``p`` and frequency ``i`` is ``p * inv_freq[i]``.
        * The returned tables have length ``head_dim`` along the last axis because
          the same ``head_dim // 2`` angles are laid out twice, back to back
          (first half and second half are identical). This matches the
          "rotate_half" pairing used by ``apply_rope``.
    """
    # TODO(student): Milestone 1
    raise NotImplementedError("Milestone 1 (1.2): implement compute_rope_cos_sin()")


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate query or key vectors by their position (checkpoint 1.2).

    Args:
        x:   [batch, seq_len, num_heads, head_dim]  (works for any number of heads,
             so the same function serves q with ``num_heads`` and k with ``num_kv_heads``)
        cos: [batch, seq_len, head_dim]
        sin: [batch, seq_len, head_dim]

    Returns:
        Rotated tensor with the same shape and dtype as ``x``.

    Convention ("rotate_half", as in HuggingFace Llama): dimension ``i`` is paired
    with dimension ``i + head_dim // 2`` and the pair is rotated as a 2-D vector by
    the angle stored at index ``i`` of cos/sin. See docs/01-tinyllama.md for the
    equation. Getting this pairing wrong yields plausible-looking but incorrect
    logits, so test it against HuggingFace early.
    """
    # TODO(student): Milestone 1
    raise NotImplementedError("Milestone 1 (1.2): implement apply_rope()")


class MLP(nn.Module):
    """Gated feed-forward network (SwiGLU) used by Llama (checkpoint 1.4).

    Three linear maps without bias:

        gate_proj: hidden_size       -> intermediate_size
        up_proj:   hidden_size       -> intermediate_size
        down_proj: intermediate_size -> hidden_size

    The activation is SiLU applied to the gate branch; the result is multiplied
    element-wise with the up branch before projecting back down.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, hidden_size]  ->  [batch, seq_len, hidden_size]
        # TODO(student): Milestone 1
        raise NotImplementedError("Milestone 1 (1.4): implement MLP.forward()")
