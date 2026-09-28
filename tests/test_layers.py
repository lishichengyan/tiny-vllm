"""Milestone 1 -- checkpoints 1.1 (RMSNorm), 1.2 (RoPE), 1.4 (MLP)."""

import pytest
import torch

from tiny_vllm.layers import MLP, RMSNorm, apply_rope, compute_rope_cos_sin

pytestmark = pytest.mark.m1

TOL = dict(rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# 1.1 RMSNorm
# ---------------------------------------------------------------------------


@pytest.mark.checkpoint("1.1")
class TestRMSNorm:
    def test_output_shape_2d_and_3d(self):
        norm = RMSNorm(16, eps=1e-5)
        assert norm(torch.randn(5, 16)).shape == (5, 16)
        assert norm(torch.randn(2, 7, 16)).shape == (2, 7, 16)

    def test_matches_huggingface(self, hf_tiny_model):
        from transformers.models.llama.modeling_llama import LlamaRMSNorm

        hidden = hf_tiny_model.config.hidden_size
        eps = hf_tiny_model.config.rms_norm_eps
        hf_norm = LlamaRMSNorm(hidden, eps=eps)
        with torch.no_grad():
            hf_norm.weight.copy_(torch.randn(hidden) * 0.5 + 1.0)  # not all ones
        ours = RMSNorm(hidden, eps=eps)
        with torch.no_grad():
            ours.weight.copy_(hf_norm.weight)

        x = torch.randn(2, 9, hidden) * 3.0
        torch.testing.assert_close(ours(x), hf_norm(x), **TOL)

    def test_unit_weight_gives_unit_rms(self):
        norm = RMSNorm(32, eps=1e-6)
        x = torch.randn(4, 6, 32) * 7.0
        y = norm(x)
        rms = y.pow(2).mean(dim=-1).sqrt()
        torch.testing.assert_close(rms, torch.ones_like(rms), rtol=1e-4, atol=1e-4)

    def test_scale_invariance(self):
        norm = RMSNorm(32, eps=1e-8)
        x = torch.randn(3, 32)
        torch.testing.assert_close(norm(x * 5.0), norm(x), rtol=1e-4, atol=1e-4)
        # Negative scale flips the sign.
        torch.testing.assert_close(norm(-x), -norm(x), rtol=1e-4, atol=1e-4)

    def test_rows_are_normalised_independently(self):
        norm = RMSNorm(16)
        x = torch.randn(2, 3, 16)
        y = norm(x)
        x2 = x.clone()
        x2[1, 2] *= 100.0
        y2 = norm(x2)
        torch.testing.assert_close(y2[0], y[0], **TOL)
        torch.testing.assert_close(y2[1, :2], y[1, :2], **TOL)

    def test_zero_input_is_finite(self):
        norm = RMSNorm(16, eps=1e-5)
        y = norm(torch.zeros(2, 16))
        assert torch.isfinite(y).all()
        torch.testing.assert_close(y, torch.zeros_like(y))

    def test_eps_is_used(self):
        # With a large eps and a tiny input, ignoring eps gives a very different answer.
        from transformers.models.llama.modeling_llama import LlamaRMSNorm

        hf_norm = LlamaRMSNorm(16, eps=1e-2)
        ours = RMSNorm(16, eps=1e-2)
        x = torch.randn(4, 16) * 1e-2
        torch.testing.assert_close(ours(x), hf_norm(x), **TOL)


# ---------------------------------------------------------------------------
# 1.2 RoPE
# ---------------------------------------------------------------------------


@pytest.mark.checkpoint("1.2")
class TestRoPE:
    HEAD_DIM = 16
    THETA = 10000.0

    def test_cos_sin_shapes_and_basic_values(self):
        positions = torch.tensor([[0, 1, 2, 3], [0, 5, 6, 7]])
        cos, sin = compute_rope_cos_sin(positions, self.HEAD_DIM, self.THETA)
        assert cos.shape == (2, 4, self.HEAD_DIM)
        assert sin.shape == (2, 4, self.HEAD_DIM)
        assert cos.dtype == torch.float32 and sin.dtype == torch.float32
        # position 0: no rotation
        torch.testing.assert_close(cos[:, 0], torch.ones(2, self.HEAD_DIM))
        torch.testing.assert_close(sin[:, 0], torch.zeros(2, self.HEAD_DIM))
        # the two halves carry the same angles
        half = self.HEAD_DIM // 2
        torch.testing.assert_close(cos[..., :half], cos[..., half:])
        torch.testing.assert_close(sin[..., :half], sin[..., half:])
        # frequency 0 is exactly angle = position
        torch.testing.assert_close(cos[1, :, 0], torch.cos(positions[1].float()))
        torch.testing.assert_close(sin[1, :, 0], torch.sin(positions[1].float()))
        # cos^2 + sin^2 = 1
        torch.testing.assert_close(cos**2 + sin**2, torch.ones_like(cos), **TOL)

    def test_cos_sin_match_huggingface(self, hf_tiny_model):
        cfg = hf_tiny_model.config
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        positions = torch.tensor([[3, 17, 5, 100, 0]])
        cos, sin = compute_rope_cos_sin(positions, head_dim, cfg.rope_theta)
        dummy = torch.zeros(1, 5, cfg.hidden_size)
        hf_cos, hf_sin = hf_tiny_model.model.rotary_emb(dummy, positions)
        torch.testing.assert_close(cos, hf_cos, **TOL)
        torch.testing.assert_close(sin, hf_sin, **TOL)

    def test_apply_rope_matches_huggingface(self, hf_tiny_model):
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

        cfg = hf_tiny_model.config
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        batch, seq_len = 2, 6
        positions = torch.tensor([[0, 1, 2, 3, 4, 5], [10, 11, 12, 13, 14, 15]])
        # q and k have different numbers of heads (GQA)
        q = torch.randn(batch, seq_len, cfg.num_attention_heads, head_dim)
        k = torch.randn(batch, seq_len, cfg.num_key_value_heads, head_dim)

        dummy = torch.zeros(batch, seq_len, cfg.hidden_size)
        hf_cos, hf_sin = hf_tiny_model.model.rotary_emb(dummy, positions)
        # HF layout is [batch, heads, seq, head_dim]
        hf_q, hf_k = apply_rotary_pos_emb(q.transpose(1, 2), k.transpose(1, 2), hf_cos, hf_sin)

        cos, sin = compute_rope_cos_sin(positions, head_dim, cfg.rope_theta)
        torch.testing.assert_close(apply_rope(q, cos, sin), hf_q.transpose(1, 2), **TOL)
        torch.testing.assert_close(apply_rope(k, cos, sin), hf_k.transpose(1, 2), **TOL)

    def test_position_zero_is_identity(self):
        x = torch.randn(1, 1, 3, self.HEAD_DIM)
        cos, sin = compute_rope_cos_sin(torch.zeros(1, 1, dtype=torch.long), self.HEAD_DIM, self.THETA)
        torch.testing.assert_close(apply_rope(x, cos, sin), x, **TOL)

    def test_preserves_shape_dtype_and_norm(self):
        x = torch.randn(2, 5, 4, self.HEAD_DIM)
        positions = torch.arange(5).repeat(2, 1) * 3
        cos, sin = compute_rope_cos_sin(positions, self.HEAD_DIM, self.THETA)
        y = apply_rope(x, cos, sin)
        assert y.shape == x.shape and y.dtype == x.dtype
        torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1), rtol=1e-4, atol=1e-4)

    def test_only_relative_position_matters(self):
        """<RoPE(q, m), RoPE(k, n)> depends only on m - n."""
        q = torch.randn(1, 1, 2, self.HEAD_DIM)
        k = torch.randn(1, 1, 2, self.HEAD_DIM)

        def dot(m, n):
            cq, sq = compute_rope_cos_sin(torch.tensor([[m]]), self.HEAD_DIM, self.THETA)
            ck, sk = compute_rope_cos_sin(torch.tensor([[n]]), self.HEAD_DIM, self.THETA)
            return (apply_rope(q, cq, sq) * apply_rope(k, ck, sk)).sum(-1)

        torch.testing.assert_close(dot(7, 3), dot(27, 23), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(dot(0, 5), dot(100, 105), rtol=1e-4, atol=1e-4)
        # ... and does change when the distance changes
        assert not torch.allclose(dot(7, 3), dot(7, 4), atol=1e-3)

    def test_decode_position_consistent_with_prefill(self):
        full_cos, full_sin = compute_rope_cos_sin(torch.arange(8)[None], self.HEAD_DIM, self.THETA)
        one_cos, one_sin = compute_rope_cos_sin(torch.tensor([[5]]), self.HEAD_DIM, self.THETA)
        torch.testing.assert_close(one_cos[0, 0], full_cos[0, 5])
        torch.testing.assert_close(one_sin[0, 0], full_sin[0, 5])

    def test_apply_rope_accepts_any_head_count(self):
        positions = torch.tensor([[1, 2, 3]])
        cos, sin = compute_rope_cos_sin(positions, self.HEAD_DIM, self.THETA)
        for heads in (1, 2, 9):
            x = torch.randn(1, 3, heads, self.HEAD_DIM)
            assert apply_rope(x, cos, sin).shape == x.shape


# ---------------------------------------------------------------------------
# 1.4 MLP
# ---------------------------------------------------------------------------


@pytest.mark.checkpoint("1.4")
class TestMLP:
    def test_output_shape(self):
        mlp = MLP(hidden_size=16, intermediate_size=40)
        assert mlp(torch.randn(2, 5, 16)).shape == (2, 5, 16)

    def test_matches_huggingface(self, hf_tiny_model):
        cfg = hf_tiny_model.config
        hf_mlp = hf_tiny_model.model.layers[0].mlp
        ours = MLP(cfg.hidden_size, cfg.intermediate_size)
        with torch.no_grad():
            ours.gate_proj.weight.copy_(hf_mlp.gate_proj.weight)
            ours.up_proj.weight.copy_(hf_mlp.up_proj.weight)
            ours.down_proj.weight.copy_(hf_mlp.down_proj.weight)
        x = torch.randn(2, 7, cfg.hidden_size)
        torch.testing.assert_close(ours(x), hf_mlp(x), **TOL)

    def test_is_gated(self):
        """Zeroing either branch of the gate kills the output; a non-gated MLP would not."""
        x = torch.randn(3, 16)
        for branch in ("gate_proj", "up_proj"):
            mlp = MLP(16, 32)
            with torch.no_grad():
                getattr(mlp, branch).weight.zero_()
            torch.testing.assert_close(mlp(x), torch.zeros(3, 16))

    def test_rows_independent(self):
        mlp = MLP(16, 32)
        x = torch.randn(4, 16)
        y = mlp(x)
        x2 = x.clone()
        x2[3] += 10.0
        torch.testing.assert_close(mlp(x2)[:3], y[:3], **TOL)
