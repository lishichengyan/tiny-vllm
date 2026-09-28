"""Milestone 2 -- checkpoints 2.1 (naive generation, greedy sampling), 2.3 (prefill),
2.4 (decode), 2.5 (cached generation). Checkpoint 2.2 (the cache itself) lives in test_kv_cache.py."""

import pytest
import torch

from helpers import RecordingModel, ScriptedModel, random_tokens
from tiny_vllm.generation import decode_step, generate_naive, generate_with_kv_cache, prefill
from tiny_vllm.kv_cache import ContiguousKVCache
from tiny_vllm.sampler import greedy_sample

pytestmark = pytest.mark.m2

TOL = dict(rtol=1e-4, atol=1e-4)
EOS = 2


def _uncached_last_logits(model, tokens: list[int]) -> torch.Tensor:
    input_ids = torch.tensor([tokens])
    positions = torch.arange(len(tokens))[None]
    with torch.no_grad():
        return model(input_ids, positions)[0, -1]


# ---------------------------------------------------------------------------
# 2.1 greedy sampling
# ---------------------------------------------------------------------------


@pytest.mark.checkpoint("2.1")
class TestGreedySample:
    def test_1d(self):
        logits = torch.tensor([0.1, 3.0, -1.0, 2.9])
        token = greedy_sample(logits)
        assert token.shape == ()
        assert token.dtype == torch.int64
        assert int(token) == 1

    def test_2d(self):
        logits = torch.tensor([[0.0, 1.0, 0.5], [9.0, -1.0, 0.0], [0.0, 0.0, 0.1]])
        tokens = greedy_sample(logits)
        assert tokens.shape == (3,)
        assert tokens.tolist() == [1, 0, 2]

    def test_3d(self):
        logits = torch.zeros(2, 4, 7)
        logits[1, 3, 6] = 1.0
        logits[0, 0, 2] = 1.0
        tokens = greedy_sample(logits)
        assert tokens.shape == (2, 4)
        assert int(tokens[1, 3]) == 6 and int(tokens[0, 0]) == 2

    def test_handles_large_negative_values(self):
        logits = torch.full((5,), -1e9)
        logits[3] = -1e8
        assert int(greedy_sample(logits)) == 3


# ---------------------------------------------------------------------------
# 2.1 naive generation
# ---------------------------------------------------------------------------


@pytest.mark.checkpoint("2.1")
class TestGenerateNaive:
    def test_follows_scripted_model(self, tiny_config):
        # next token after position p is script[p]; prompt occupies positions 0..2
        script = [0, 0, 10, 11, 12, 13, 14]
        model = ScriptedModel(tiny_config, script)
        assert generate_naive(model, [5, 6, 7], max_new_tokens=4) == [10, 11, 12, 13]

    def test_returns_plain_python_ints(self, tiny_config):
        model = ScriptedModel(tiny_config, [0, 0, 10, 11])
        out = generate_naive(model, [5, 6, 7], max_new_tokens=2)
        assert isinstance(out, list) and all(type(t) is int for t in out)

    def test_stops_at_eos_and_includes_it(self, tiny_config):
        script = [0, 0, 10, EOS, 12, 13]
        model = ScriptedModel(tiny_config, script)
        assert generate_naive(model, [5, 6, 7], max_new_tokens=10, eos_token_id=EOS) == [10, EOS]

    def test_eos_ignored_when_not_given(self, tiny_config):
        script = [0, 0, 10, EOS, 12, 13]
        model = ScriptedModel(tiny_config, script)
        assert generate_naive(model, [5, 6, 7], max_new_tokens=3, eos_token_id=None) == [10, EOS, 12]

    def test_eos_in_prompt_does_not_stop(self, tiny_config):
        script = [0, 0, 10, 11]
        model = ScriptedModel(tiny_config, script)
        assert generate_naive(model, [5, EOS, 7], max_new_tokens=2, eos_token_id=EOS) == [10, 11]

    def test_zero_new_tokens(self, tiny_config):
        model = RecordingModel(ScriptedModel(tiny_config, [0, 0, 10]))
        assert generate_naive(model, [5, 6, 7], max_new_tokens=0) == []
        assert model.calls == []

    def test_does_not_mutate_prompt(self, tiny_config):
        prompt = [5, 6, 7]
        generate_naive(ScriptedModel(tiny_config, [0, 0, 10, 11]), prompt, max_new_tokens=2)
        assert prompt == [5, 6, 7]

    def test_recomputes_the_whole_sequence_every_step(self, tiny_config):
        model = RecordingModel(ScriptedModel(tiny_config, list(range(10, 30))))
        generate_naive(model, [5, 6, 7], max_new_tokens=4)
        assert model.input_lengths() == [3, 4, 5, 6]
        assert model.batch_sizes() == [1, 1, 1, 1]
        for call in model.calls:
            n = call["input_ids"].shape[1]
            assert call["positions"].tolist() == [list(range(n))]
            assert call["kv_cache"] is None
        # the sequence fed at step i is prompt + tokens generated so far
        assert model.calls[-1]["input_ids"][0].tolist() == [5, 6, 7, 12, 13, 14]

    def test_matches_huggingface_generate(self, hf_tiny_model, loaded_tiny_model, tiny_config):
        prompt = random_tokens(5, tiny_config.vocab_size, seed=7)
        input_ids = torch.tensor([prompt])
        with torch.no_grad():
            hf_out = hf_tiny_model.generate(
                input_ids, max_new_tokens=8, do_sample=False, eos_token_id=EOS, pad_token_id=0
            )
        expected = hf_out[0, len(prompt) :].tolist()
        actual = generate_naive(loaded_tiny_model, prompt, max_new_tokens=8, eos_token_id=EOS)
        assert actual == expected


# ---------------------------------------------------------------------------
# 2.3 prefill
# ---------------------------------------------------------------------------


@pytest.mark.checkpoint("2.3")
class TestPrefill:
    def test_returns_last_position_logits(self, tiny_model, tiny_config):
        prompt = random_tokens(6, tiny_config.vocab_size, seed=8)
        cache = ContiguousKVCache(tiny_config, max_seq_len=16)
        logits = prefill(tiny_model, cache, prompt)
        assert logits.shape == (tiny_config.vocab_size,)
        torch.testing.assert_close(logits, _uncached_last_logits(tiny_model, prompt), **TOL)

    def test_single_forward_with_full_prompt(self, tiny_model, tiny_config):
        model = RecordingModel(tiny_model)
        cache = ContiguousKVCache(tiny_config, max_seq_len=16)
        prompt = random_tokens(5, tiny_config.vocab_size, seed=9)
        prefill(model, cache, prompt)
        assert len(model.calls) == 1
        call = model.calls[0]
        assert call["input_ids"].tolist() == [prompt]
        assert call["positions"].tolist() == [[0, 1, 2, 3, 4]]
        assert call["kv_cache"] is cache

    def test_fills_exactly_prompt_positions_for_every_layer(self, tiny_model, tiny_config):
        cache = ContiguousKVCache(tiny_config, max_seq_len=16)
        cache.k_cache.fill_(float("nan"))
        cache.v_cache.fill_(float("nan"))
        prompt = random_tokens(5, tiny_config.vocab_size, seed=10)
        prefill(tiny_model, cache, prompt)
        for layer in range(tiny_config.num_layers):
            assert torch.isfinite(cache.k_cache[layer, :5]).all(), f"layer {layer} K not written"
            assert torch.isfinite(cache.v_cache[layer, :5]).all(), f"layer {layer} V not written"
            assert torch.isnan(cache.k_cache[layer, 5:]).all(), f"layer {layer} wrote beyond the prompt"
            assert torch.isnan(cache.v_cache[layer, 5:]).all(), f"layer {layer} wrote beyond the prompt"

    def test_cache_of_a_prefix_is_independent_of_later_tokens(self, tiny_model, tiny_config):
        """K/V at position p depend only on tokens 0..p, so a longer prompt shares the prefix cache."""
        tokens = random_tokens(8, tiny_config.vocab_size, seed=11)
        short = ContiguousKVCache(tiny_config, max_seq_len=16)
        long = ContiguousKVCache(tiny_config, max_seq_len=16)
        prefill(tiny_model, short, tokens[:4])
        prefill(tiny_model, long, tokens)
        for layer in range(tiny_config.num_layers):
            k_short, v_short = short.read(layer, 4)
            k_long, v_long = long.read(layer, 8)
            torch.testing.assert_close(k_long[:4], k_short, **TOL)
            torch.testing.assert_close(v_long[:4], v_short, **TOL)

    def test_layers_store_different_kv(self, tiny_model, tiny_config):
        cache = ContiguousKVCache(tiny_config, max_seq_len=16)
        prefill(tiny_model, cache, random_tokens(4, tiny_config.vocab_size, seed=12))
        k0, _ = cache.read(0, 4)
        k1, _ = cache.read(1, 4)
        assert not torch.allclose(k0, k1)


# ---------------------------------------------------------------------------
# 2.4 decode
# ---------------------------------------------------------------------------


@pytest.mark.checkpoint("2.4")
class TestDecodeStep:
    def test_logits_match_uncached_forward(self, tiny_model, tiny_config):
        tokens = random_tokens(7, tiny_config.vocab_size, seed=13)
        cache = ContiguousKVCache(tiny_config, max_seq_len=16)
        prefill(tiny_model, cache, tokens[:4])
        for p in range(4, 7):
            logits = decode_step(tiny_model, cache, tokens[p], p)
            assert logits.shape == (tiny_config.vocab_size,)
            torch.testing.assert_close(logits, _uncached_last_logits(tiny_model, tokens[: p + 1]), **TOL)

    def test_processes_only_the_newest_token(self, tiny_model, tiny_config):
        model = RecordingModel(tiny_model)
        cache = ContiguousKVCache(tiny_config, max_seq_len=16)
        prefill(model, cache, random_tokens(4, tiny_config.vocab_size, seed=14))
        model.calls.clear()
        decode_step(model, cache, token=42, position=4)
        assert len(model.calls) == 1
        call = model.calls[0]
        assert call["input_ids"].tolist() == [[42]]
        assert call["positions"].tolist() == [[4]]
        assert call["kv_cache"] is cache

    def test_writes_exactly_one_position_per_layer(self, tiny_model, tiny_config):
        cache = ContiguousKVCache(tiny_config, max_seq_len=16)
        prefill(tiny_model, cache, random_tokens(4, tiny_config.vocab_size, seed=15))
        cache.k_cache[:, 4:].fill_(float("nan"))
        cache.v_cache[:, 4:].fill_(float("nan"))
        decode_step(tiny_model, cache, token=42, position=4)
        for layer in range(tiny_config.num_layers):
            assert torch.isfinite(cache.k_cache[layer, 4]).all()
            assert torch.isfinite(cache.v_cache[layer, 4]).all()
            assert torch.isnan(cache.k_cache[layer, 5:]).all()
            assert torch.isnan(cache.v_cache[layer, 5:]).all()

    def test_incremental_cache_equals_prefill_cache(self, tiny_model, tiny_config):
        """prefill(ABC) + decode(D) + decode(E) leaves the same cache as prefill(ABCDE)."""
        tokens = random_tokens(6, tiny_config.vocab_size, seed=16)
        incremental = ContiguousKVCache(tiny_config, max_seq_len=16)
        prefill(tiny_model, incremental, tokens[:3])
        for p in range(3, 6):
            decode_step(tiny_model, incremental, tokens[p], p)
        full = ContiguousKVCache(tiny_config, max_seq_len=16)
        prefill(tiny_model, full, tokens)
        for layer in range(tiny_config.num_layers):
            k_inc, v_inc = incremental.read(layer, 6)
            k_full, v_full = full.read(layer, 6)
            torch.testing.assert_close(k_inc, k_full, **TOL)
            torch.testing.assert_close(v_inc, v_full, **TOL)


# ---------------------------------------------------------------------------
# 2.5 cached generation
# ---------------------------------------------------------------------------


@pytest.mark.checkpoint("2.5")
class TestGenerateWithKVCache:
    def test_follows_scripted_model(self, tiny_config):
        model = ScriptedModel(tiny_config, [0, 0, 10, 11, 12, 13])
        assert generate_with_kv_cache(model, [5, 6, 7], max_new_tokens=4) == [10, 11, 12, 13]

    def test_stops_at_eos(self, tiny_config):
        model = ScriptedModel(tiny_config, [0, 0, 10, EOS, 12])
        assert generate_with_kv_cache(model, [5, 6, 7], max_new_tokens=10, eos_token_id=EOS) == [10, EOS]

    def test_zero_new_tokens(self, tiny_config):
        model = ScriptedModel(tiny_config, [0, 0, 10])
        assert generate_with_kv_cache(model, [5, 6, 7], max_new_tokens=0) == []

    def test_only_the_first_forward_sees_more_than_one_token(self, tiny_model, tiny_config):
        model = RecordingModel(tiny_model)
        prompt = random_tokens(5, tiny_config.vocab_size, seed=17)
        out = generate_with_kv_cache(model, prompt, max_new_tokens=6)
        assert len(out) == 6
        assert model.input_lengths() == [5, 1, 1, 1, 1, 1]
        assert [c["positions"].reshape(-1)[-1].item() for c in model.calls] == [4, 5, 6, 7, 8, 9]
        for call in model.calls:
            assert isinstance(call["kv_cache"], ContiguousKVCache)
        # decode step i feeds the token generated at step i-1
        assert [c["input_ids"].item() for c in model.calls[1:]] == out[:5]

    @pytest.mark.parametrize("seed", [18, 19, 20])
    def test_matches_naive_generation(self, tiny_model, tiny_config, seed):
        prompt = random_tokens(6, tiny_config.vocab_size, seed=seed)
        cached = generate_with_kv_cache(tiny_model, prompt, max_new_tokens=8, eos_token_id=EOS)
        naive = generate_naive(tiny_model, prompt, max_new_tokens=8, eos_token_id=EOS)
        assert cached == naive

    def test_matches_huggingface_generate(self, hf_tiny_model, loaded_tiny_model, tiny_config):
        prompt = random_tokens(4, tiny_config.vocab_size, seed=21)
        with torch.no_grad():
            hf_out = hf_tiny_model.generate(
                torch.tensor([prompt]), max_new_tokens=8, do_sample=False, eos_token_id=EOS, pad_token_id=0
            )
        expected = hf_out[0, len(prompt) :].tolist()
        assert (
            generate_with_kv_cache(loaded_tiny_model, prompt, max_new_tokens=8, eos_token_id=EOS) == expected
        )

    def test_single_token_prompt(self, tiny_model, tiny_config):
        prompt = [17]
        assert generate_with_kv_cache(tiny_model, prompt, 5) == generate_naive(tiny_model, prompt, 5)


@pytest.mark.integration
def test_m2_integration_cached_equals_uncached(tiny_model, tiny_config):
    """uncached logits ≈ cached logits at every step, and the tokens agree."""
    prompt = random_tokens(7, tiny_config.vocab_size, seed=22)
    max_new = 10
    recording = RecordingModel(tiny_model)
    cached_tokens = generate_with_kv_cache(recording, prompt, max_new_tokens=max_new)
    naive_tokens = generate_naive(tiny_model, prompt, max_new_tokens=max_new)
    assert cached_tokens == naive_tokens
    assert recording.input_lengths() == [len(prompt)] + [1] * (max_new - 1)

    # Step-by-step logits agree, not just the argmax.
    cache = ContiguousKVCache(tiny_config, max_seq_len=len(prompt) + max_new)
    logits = prefill(tiny_model, cache, prompt)
    torch.testing.assert_close(logits, _uncached_last_logits(tiny_model, prompt), **TOL)
    seq = list(prompt)
    for token in cached_tokens[:-1]:
        seq.append(token)
        logits = decode_step(tiny_model, cache, token, len(seq) - 1)
        torch.testing.assert_close(logits, _uncached_last_logits(tiny_model, seq), **TOL)


@pytest.mark.integration
@pytest.mark.hf
def test_m2_integration_real_model(real_model_and_tokenizer):
    model, tokenizer = real_model_and_tokenizer
    prompt = tokenizer("The quick brown fox", return_tensors="pt").input_ids[0].tolist()
    naive = generate_naive(model, prompt, max_new_tokens=6, eos_token_id=tokenizer.eos_token_id)
    cached = generate_with_kv_cache(model, prompt, max_new_tokens=6, eos_token_id=tokenizer.eos_token_id)
    assert naive == cached
    assert len(cached) >= 1
