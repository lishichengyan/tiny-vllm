"""Milestone 1 -- checkpoints 1.3 (attention), 1.4 (decoder layer), 1.5 (weights), 1.6 (full model)."""

import pytest
import torch
import torch.nn.functional as F

from helpers import capture_io, random_tokens
from tiny_vllm.attention import Attention, causal_attention
from tiny_vllm.layers import compute_rope_cos_sin
from tiny_vllm.loader import load_weights
from tiny_vllm.model import DecoderLayer, TinyLlama

pytestmark = pytest.mark.m1

TOL = dict(rtol=1e-4, atol=1e-4)


def _hf_layer_output(output):
    return output[0] if isinstance(output, tuple) else output


def _hf_layer_input(args, kwargs):
    return args[0] if args else kwargs["hidden_states"]


# ---------------------------------------------------------------------------
# 1.3 causal attention
# ---------------------------------------------------------------------------


def _reference_attention(q, k, v):
    """Ground truth from torch's built-in SDPA (queries == last positions of the keys)."""
    batch, num_queries, num_heads, head_dim = q.shape
    num_keys, num_kv_heads = k.shape[1], k.shape[2]
    rep = num_heads // num_kv_heads
    k = k.repeat_interleave(rep, dim=2)  # GQA: kv head j serves query heads j*rep .. (j+1)*rep-1
    v = v.repeat_interleave(rep, dim=2)
    q_pad = torch.zeros(batch, num_keys, num_heads, head_dim, dtype=q.dtype)
    q_pad[:, num_keys - num_queries :] = q
    out = F.scaled_dot_product_attention(
        q_pad.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
    )
    return out.transpose(1, 2)[:, num_keys - num_queries :]


@pytest.mark.checkpoint("1.3")
class TestCausalAttention:
    def test_output_shape(self):
        q = torch.randn(2, 5, 4, 8)
        k = torch.randn(2, 5, 2, 8)
        v = torch.randn(2, 5, 2, 8)
        assert causal_attention(q, k, v).shape == (2, 5, 4, 8)

    @pytest.mark.parametrize("num_kv_heads", [4, 2, 1])
    def test_matches_reference_full_sequence(self, num_kv_heads):
        q = torch.randn(2, 7, 4, 8)
        k = torch.randn(2, 7, num_kv_heads, 8)
        v = torch.randn(2, 7, num_kv_heads, 8)
        torch.testing.assert_close(causal_attention(q, k, v), _reference_attention(q, k, v), **TOL)

    @pytest.mark.parametrize("num_queries", [1, 3])
    def test_matches_reference_with_history(self, num_queries):
        """Decode-style call: few queries against a longer key history."""
        q = torch.randn(1, num_queries, 4, 8)
        k = torch.randn(1, 9, 2, 8)
        v = torch.randn(1, 9, 2, 8)
        torch.testing.assert_close(causal_attention(q, k, v), _reference_attention(q, k, v), **TOL)

    def test_is_causal(self):
        q = torch.randn(1, 6, 2, 8)
        k = torch.randn(1, 6, 2, 8)
        v = torch.randn(1, 6, 2, 8)
        out = causal_attention(q, k, v)
        k2, v2 = k.clone(), v.clone()
        k2[:, 4:] = torch.randn(1, 2, 2, 8)  # change the future ...
        v2[:, 4:] = torch.randn(1, 2, 2, 8)
        out2 = causal_attention(q, k2, v2)
        torch.testing.assert_close(out2[:, :4], out[:, :4], **TOL)  # ... the past is unaffected
        assert not torch.allclose(out2[:, 4:], out[:, 4:], atol=1e-3)

    def test_suffix_queries_consistent_with_full_pass(self):
        """Attending with only the last s queries equals the last s rows of the full pass."""
        q = torch.randn(1, 8, 4, 8)
        k = torch.randn(1, 8, 2, 8)
        v = torch.randn(1, 8, 2, 8)
        full = causal_attention(q, k, v)
        for s in (1, 3, 8):
            torch.testing.assert_close(causal_attention(q[:, -s:], k, v), full[:, -s:], **TOL)

    def test_single_key_returns_value(self):
        q = torch.randn(2, 1, 4, 8)
        k = torch.randn(2, 1, 2, 8)
        v = torch.randn(2, 1, 2, 8)
        out = causal_attention(q, k, v)
        torch.testing.assert_close(out, v.repeat_interleave(2, dim=2), **TOL)

    def test_uses_scaling(self):
        """Without the 1/sqrt(d) scale the softmax saturates for large logits."""
        q = torch.randn(1, 4, 1, 64) * 4
        k = torch.randn(1, 4, 1, 64) * 4
        v = torch.randn(1, 4, 1, 64)
        torch.testing.assert_close(causal_attention(q, k, v), _reference_attention(q, k, v), **TOL)

    def test_batch_rows_independent(self):
        q = torch.randn(2, 5, 4, 8)
        k = torch.randn(2, 5, 2, 8)
        v = torch.randn(2, 5, 2, 8)
        out = causal_attention(q, k, v)
        torch.testing.assert_close(causal_attention(q[:1], k[:1], v[:1]), out[:1], **TOL)

    def test_more_queries_than_keys_raises(self):
        with pytest.raises(ValueError):
            causal_attention(torch.randn(1, 5, 2, 8), torch.randn(1, 3, 2, 8), torch.randn(1, 3, 2, 8))


@pytest.mark.checkpoint("1.3")
class TestAttentionModule:
    def test_parameter_shapes(self, tiny_config):
        attn = Attention(tiny_config, layer_idx=0)
        h, nh, nkv, hd = (
            tiny_config.hidden_size,
            tiny_config.num_heads,
            tiny_config.num_kv_heads,
            tiny_config.head_dim,
        )
        assert attn.q_proj.weight.shape == (nh * hd, h)
        assert attn.k_proj.weight.shape == (nkv * hd, h)
        assert attn.v_proj.weight.shape == (nkv * hd, h)
        assert attn.o_proj.weight.shape == (h, nh * hd)

    def test_project_qkv_shapes_and_rope(self, tiny_config):
        attn = Attention(tiny_config, layer_idx=0)
        batch, seq_len = 2, 5
        positions = torch.arange(seq_len).repeat(batch, 1)
        cos, sin = compute_rope_cos_sin(positions, tiny_config.head_dim, tiny_config.rope_theta)
        x = torch.randn(batch, seq_len, tiny_config.hidden_size)
        q, k, v = attn.project_qkv(x, cos, sin)
        assert q.shape == (batch, seq_len, tiny_config.num_heads, tiny_config.head_dim)
        assert k.shape == (batch, seq_len, tiny_config.num_kv_heads, tiny_config.head_dim)
        assert v.shape == (batch, seq_len, tiny_config.num_kv_heads, tiny_config.head_dim)
        # V is a plain projection; Q and K are rotated (so depend on position)
        v_ref = attn.v_proj(x).view(batch, seq_len, tiny_config.num_kv_heads, tiny_config.head_dim)
        torch.testing.assert_close(v, v_ref, **TOL)
        positions2 = positions + 7
        cos2, sin2 = compute_rope_cos_sin(positions2, tiny_config.head_dim, tiny_config.rope_theta)
        q2, k2, v2 = attn.project_qkv(x, cos2, sin2)
        torch.testing.assert_close(v2, v, **TOL)
        assert not torch.allclose(q2, q, atol=1e-3)
        assert not torch.allclose(k2, k, atol=1e-3)

    def test_matches_huggingface_attention(self, hf_tiny_model, tiny_config):
        """Feed the exact input HF's layer-0 attention saw and compare its output."""
        hf_attn = hf_tiny_model.model.layers[0].self_attn
        ours = Attention(tiny_config, layer_idx=0)
        with torch.no_grad():
            ours.q_proj.weight.copy_(hf_attn.q_proj.weight)
            ours.k_proj.weight.copy_(hf_attn.k_proj.weight)
            ours.v_proj.weight.copy_(hf_attn.v_proj.weight)
            ours.o_proj.weight.copy_(hf_attn.o_proj.weight)

        input_ids = torch.tensor([random_tokens(9, tiny_config.vocab_size, seed=3)])
        with capture_io(hf_attn) as records, torch.no_grad():
            hf_tiny_model(input_ids)
        args, kwargs, output = records[0]
        hidden = _hf_layer_input(args, kwargs)
        expected = _hf_layer_output(output)

        positions = torch.arange(input_ids.shape[1])[None]
        cos, sin = compute_rope_cos_sin(positions, tiny_config.head_dim, tiny_config.rope_theta)
        with torch.no_grad():
            actual = ours(hidden, positions, cos, sin)
        assert actual.shape == hidden.shape
        torch.testing.assert_close(actual, expected, **TOL)


# ---------------------------------------------------------------------------
# 1.4 decoder layer
# ---------------------------------------------------------------------------


@pytest.mark.checkpoint("1.4")
class TestDecoderLayer:
    def test_output_shape(self, tiny_config):
        layer = DecoderLayer(tiny_config, layer_idx=0)
        x = torch.randn(2, 5, tiny_config.hidden_size)
        positions = torch.arange(5).repeat(2, 1)
        cos, sin = compute_rope_cos_sin(positions, tiny_config.head_dim, tiny_config.rope_theta)
        assert layer(x, positions, cos, sin).shape == x.shape

    def test_has_residual_connections(self, tiny_config):
        """Zero attention and MLP output projections -> the layer is the identity."""
        layer = DecoderLayer(tiny_config, layer_idx=0)
        with torch.no_grad():
            layer.self_attn.o_proj.weight.zero_()
            layer.mlp.down_proj.weight.zero_()
        x = torch.randn(1, 4, tiny_config.hidden_size)
        positions = torch.arange(4)[None]
        cos, sin = compute_rope_cos_sin(positions, tiny_config.head_dim, tiny_config.rope_theta)
        torch.testing.assert_close(layer(x, positions, cos, sin), x, **TOL)

    def test_isolated_layer_matches_huggingface(self, hf_tiny_model, tiny_config):
        hf_layer = hf_tiny_model.model.layers[1]
        ours = DecoderLayer(tiny_config, layer_idx=1)
        with torch.no_grad():
            ours.input_layernorm.weight.copy_(hf_layer.input_layernorm.weight)
            ours.post_attention_layernorm.weight.copy_(hf_layer.post_attention_layernorm.weight)
            ours.self_attn.q_proj.weight.copy_(hf_layer.self_attn.q_proj.weight)
            ours.self_attn.k_proj.weight.copy_(hf_layer.self_attn.k_proj.weight)
            ours.self_attn.v_proj.weight.copy_(hf_layer.self_attn.v_proj.weight)
            ours.self_attn.o_proj.weight.copy_(hf_layer.self_attn.o_proj.weight)
            ours.mlp.gate_proj.weight.copy_(hf_layer.mlp.gate_proj.weight)
            ours.mlp.up_proj.weight.copy_(hf_layer.mlp.up_proj.weight)
            ours.mlp.down_proj.weight.copy_(hf_layer.mlp.down_proj.weight)

        input_ids = torch.tensor([random_tokens(11, tiny_config.vocab_size, seed=4)])
        with capture_io(hf_layer) as records, torch.no_grad():
            hf_tiny_model(input_ids)
        args, kwargs, output = records[0]
        hidden = _hf_layer_input(args, kwargs)
        expected = _hf_layer_output(output)

        positions = torch.arange(input_ids.shape[1])[None]
        cos, sin = compute_rope_cos_sin(positions, tiny_config.head_dim, tiny_config.rope_theta)
        with torch.no_grad():
            actual = ours(hidden, positions, cos, sin)
        torch.testing.assert_close(actual, expected, **TOL)

    def test_every_layer_matches_huggingface(self, hf_tiny_model, loaded_tiny_model, tiny_config):
        """Per-layer comparison with loaded weights: pinpoints the first diverging layer."""
        input_ids = torch.tensor([random_tokens(10, tiny_config.vocab_size, seed=5)])
        positions = torch.arange(input_ids.shape[1])[None]
        cos, sin = compute_rope_cos_sin(positions, tiny_config.head_dim, tiny_config.rope_theta)
        for i, (hf_layer, our_layer) in enumerate(zip(hf_tiny_model.model.layers, loaded_tiny_model.layers)):
            with capture_io(hf_layer) as records, torch.no_grad():
                hf_tiny_model(input_ids)
            args, kwargs, output = records[0]
            with torch.no_grad():
                actual = our_layer(_hf_layer_input(args, kwargs), positions, cos, sin)
            torch.testing.assert_close(actual, _hf_layer_output(output), **TOL, msg=f"layer {i} differs")


# ---------------------------------------------------------------------------
# 1.5 weight loading
# ---------------------------------------------------------------------------


@pytest.mark.checkpoint("1.5")
class TestLoadWeights:
    def test_every_parameter_is_overwritten(self, tiny_config, hf_tiny_model):
        model = TinyLlama(tiny_config)
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(float("nan"))
        load_weights(model, hf_tiny_model.state_dict())
        for name, p in model.named_parameters():
            assert torch.isfinite(p).all(), f"{name} was not loaded"

    def test_selected_tensors_match(self, tiny_config, hf_tiny_model):
        model = TinyLlama(tiny_config)
        load_weights(model, hf_tiny_model.state_dict())
        hf = hf_tiny_model
        torch.testing.assert_close(model.embed_tokens.weight, hf.model.embed_tokens.weight)
        torch.testing.assert_close(model.norm.weight, hf.model.norm.weight)
        torch.testing.assert_close(model.lm_head.weight, hf.lm_head.weight)
        torch.testing.assert_close(
            model.layers[1].self_attn.k_proj.weight, hf.model.layers[1].self_attn.k_proj.weight
        )
        torch.testing.assert_close(model.layers[0].mlp.up_proj.weight, hf.model.layers[0].mlp.up_proj.weight)
        torch.testing.assert_close(
            model.layers[1].post_attention_layernorm.weight,
            hf.model.layers[1].post_attention_layernorm.weight,
        )

    def test_parameter_objects_are_not_replaced(self, tiny_config, hf_tiny_model):
        model = TinyLlama(tiny_config)
        ids_before = {name: id(p) for name, p in model.named_parameters()}
        load_weights(model, hf_tiny_model.state_dict())
        ids_after = {name: id(p) for name, p in model.named_parameters()}
        assert ids_before == ids_after

    def test_missing_key_raises(self, tiny_config, hf_tiny_model):
        sd = dict(hf_tiny_model.state_dict())
        del sd["model.layers.1.mlp.down_proj.weight"]
        with pytest.raises(ValueError, match="down_proj"):
            load_weights(TinyLlama(tiny_config), sd)

    def test_unexpected_key_raises(self, tiny_config, hf_tiny_model):
        sd = dict(hf_tiny_model.state_dict())
        sd["model.layers.0.self_attn.extra.weight"] = torch.zeros(1)
        with pytest.raises(ValueError, match="extra"):
            load_weights(TinyLlama(tiny_config), sd)

    def test_inv_freq_buffers_are_ignored(self, tiny_config, hf_tiny_model):
        sd = dict(hf_tiny_model.state_dict())
        sd["model.layers.0.self_attn.rotary_emb.inv_freq"] = torch.zeros(4)
        load_weights(TinyLlama(tiny_config), sd)  # must not raise

    def test_dtype_is_converted(self, tiny_config, hf_tiny_model):
        sd = {k: v.to(torch.bfloat16) for k, v in hf_tiny_model.state_dict().items()}
        model = TinyLlama(tiny_config)
        load_weights(model, sd)
        for name, p in model.named_parameters():
            assert p.dtype == torch.float32, name
        torch.testing.assert_close(model.norm.weight, sd["model.norm.weight"].float())

    def test_tied_embeddings_without_lm_head_key(self, tiny_config, hf_tiny_model):
        from dataclasses import replace

        tied_config = replace(tiny_config, tie_word_embeddings=True)
        model = TinyLlama(tied_config)
        sd = dict(hf_tiny_model.state_dict())
        del sd["lm_head.weight"]  # exactly what a tied safetensors checkpoint looks like
        load_weights(model, sd)
        torch.testing.assert_close(model.lm_head.weight, sd["model.embed_tokens.weight"])
        assert model.lm_head.weight.data_ptr() == model.embed_tokens.weight.data_ptr()

    def test_tied_embeddings_with_matching_lm_head_key(self, tiny_config, hf_tiny_model):
        from dataclasses import replace

        tied_config = replace(tiny_config, tie_word_embeddings=True)
        model = TinyLlama(tied_config)
        sd = dict(hf_tiny_model.state_dict())
        sd["lm_head.weight"] = sd["model.embed_tokens.weight"].clone()
        load_weights(model, sd)
        torch.testing.assert_close(model.lm_head.weight, sd["model.embed_tokens.weight"])

    def test_tied_embeddings_with_conflicting_lm_head_raises(self, tiny_config, hf_tiny_model):
        from dataclasses import replace

        tied_config = replace(tiny_config, tie_word_embeddings=True)
        model = TinyLlama(tied_config)
        sd = dict(hf_tiny_model.state_dict())
        sd["lm_head.weight"] = sd["model.embed_tokens.weight"] + 1.0
        with pytest.raises(ValueError):
            load_weights(model, sd)


# ---------------------------------------------------------------------------
# 1.6 full model
# ---------------------------------------------------------------------------


@pytest.mark.checkpoint("1.6")
class TestTinyLlamaForward:
    def test_logits_shape_and_dtype(self, tiny_model, tiny_config):
        input_ids = torch.tensor([random_tokens(6, tiny_config.vocab_size, seed=6)])
        positions = torch.arange(6)[None]
        logits = tiny_model(input_ids, positions)
        assert logits.shape == (1, 6, tiny_config.vocab_size)
        assert logits.dtype == torch.float32
        assert torch.isfinite(logits).all()

    @pytest.mark.parametrize("seq_len", [1, 2, 13])
    def test_matches_huggingface_batch1(self, hf_tiny_model, loaded_tiny_model, tiny_config, seq_len):
        input_ids = torch.tensor([random_tokens(seq_len, tiny_config.vocab_size, seed=seq_len)])
        positions = torch.arange(seq_len)[None]
        with torch.no_grad():
            expected = hf_tiny_model(input_ids).logits
            actual = loaded_tiny_model(input_ids, positions)
        torch.testing.assert_close(actual, expected, **TOL)

    def test_matches_huggingface_batch2(self, hf_tiny_model, loaded_tiny_model, tiny_config):
        input_ids = torch.tensor(
            [
                random_tokens(8, tiny_config.vocab_size, seed=20),
                random_tokens(8, tiny_config.vocab_size, seed=21),
            ]
        )
        positions = torch.arange(8).repeat(2, 1)
        with torch.no_grad():
            expected = hf_tiny_model(input_ids).logits
            actual = loaded_tiny_model(input_ids, positions)
        torch.testing.assert_close(actual, expected, **TOL)

    def test_per_layer_hidden_states_match_huggingface(self, hf_tiny_model, loaded_tiny_model, tiny_config):
        """Compare the hidden state after every layer -- tells you WHERE things diverge."""
        input_ids = torch.tensor([random_tokens(7, tiny_config.vocab_size, seed=30)])
        positions = torch.arange(7)[None]
        with torch.no_grad():
            hf_out = hf_tiny_model(input_ids, output_hidden_states=True)
        ours = []
        handles = [
            layer.register_forward_hook(lambda m, a, o: ours.append(o)) for layer in loaded_tiny_model.layers
        ]
        try:
            with torch.no_grad():
                loaded_tiny_model(input_ids, positions)
        finally:
            for h in handles:
                h.remove()
        # hf_out.hidden_states[0] is the embedding; [i+1] is the output of layer i.
        # The last HF entry has the final norm applied, so compare all but the last layer here.
        for i in range(tiny_config.num_layers - 1):
            torch.testing.assert_close(ours[i], hf_out.hidden_states[i + 1], **TOL, msg=f"layer {i} differs")

    def test_batch_rows_are_independent(self, tiny_model, tiny_config):
        a = torch.tensor([random_tokens(6, tiny_config.vocab_size, seed=40)])
        b = torch.tensor([random_tokens(6, tiny_config.vocab_size, seed=41)])
        positions = torch.arange(6)[None]
        with torch.no_grad():
            single = tiny_model(a, positions)
            batched = tiny_model(torch.cat([a, b]), positions.repeat(2, 1))
        torch.testing.assert_close(batched[:1], single, **TOL)

    def test_is_causal(self, tiny_model, tiny_config):
        tokens = random_tokens(9, tiny_config.vocab_size, seed=50)
        with torch.no_grad():
            short = tiny_model(torch.tensor([tokens[:5]]), torch.arange(5)[None])
            long = tiny_model(torch.tensor([tokens]), torch.arange(9)[None])
        torch.testing.assert_close(long[:, :5], short, **TOL)

    def test_shifting_all_positions_does_not_change_logits(self, tiny_model, tiny_config):
        """RoPE only encodes relative distances."""
        input_ids = torch.tensor([random_tokens(6, tiny_config.vocab_size, seed=60)])
        with torch.no_grad():
            base = tiny_model(input_ids, torch.arange(6)[None])
            shifted = tiny_model(input_ids, torch.arange(6)[None] + 37)
        torch.testing.assert_close(shifted, base, rtol=1e-3, atol=1e-3)

    def test_positions_are_used(self, tiny_model, tiny_config):
        input_ids = torch.tensor([random_tokens(6, tiny_config.vocab_size, seed=61)])
        with torch.no_grad():
            base = tiny_model(input_ids, torch.arange(6)[None])
            scrambled = tiny_model(input_ids, torch.tensor([[0, 5, 1, 4, 2, 3]]))
        assert not torch.allclose(base, scrambled, atol=1e-3)


@pytest.mark.integration
@pytest.mark.checkpoint("1.6")
def test_m1_integration_random_model(hf_tiny_model, loaded_tiny_model, tiny_config):
    """HuggingFace logits ≈ TinyLlama logits (random weights, no download)."""
    input_ids = torch.tensor([random_tokens(16, tiny_config.vocab_size, seed=99)])
    with torch.no_grad():
        hf_logits = hf_tiny_model(input_ids).logits
        tiny_logits = loaded_tiny_model(input_ids, torch.arange(16)[None])
    torch.testing.assert_close(tiny_logits, hf_logits, **TOL)


@pytest.mark.integration
@pytest.mark.hf
@pytest.mark.checkpoint("1.6")
def test_m1_integration_real_model(real_model_and_tokenizer, real_hf_model):
    """HuggingFace logits ≈ TinyLlama logits on the real SmolLM2-135M checkpoint."""
    model, tokenizer = real_model_and_tokenizer
    prompts = ["The capital of France is", "def fibonacci(n):\n    if n <"]
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        positions = torch.arange(input_ids.shape[1])[None]
        with torch.no_grad():
            hf_logits = real_hf_model(input_ids).logits
            tiny_logits = model(input_ids, positions)
        torch.testing.assert_close(tiny_logits, hf_logits, rtol=1e-3, atol=1e-3)
        assert tiny_logits[0, -1].argmax() == hf_logits[0, -1].argmax()
