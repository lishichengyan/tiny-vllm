"""Milestone 5 -- checkpoints 5.4 (batched execution) and 5.5 (per-request KV mappings).
Milestone 6 -- checkpoints 6.1 to 6.6 (continuous batching)."""

from dataclasses import replace

import pytest
import torch

from helpers import RecordingModel, assert_block_tables_disjoint, random_tokens
from tiny_vllm.attention import gather_kv
from tiny_vllm.engine import LLMEngine
from tiny_vllm.generation import generate_naive, prefill
from tiny_vllm.kv_cache import ContiguousKVCache, PagedKVCache, blocks_needed
from tiny_vllm.model import TinyLlama
from tiny_vllm.request import RequestStatus

BLOCK_SIZE = 4


@pytest.fixture
def model(tiny_config) -> TinyLlama:
    """Same weights as ``tiny_model`` (same seed) but with EOS disabled, so lengths are predictable."""
    torch.manual_seed(42)
    m = TinyLlama(replace(tiny_config, eos_token_id=None)).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _engine(model, num_blocks=64, max_num_seqs=8, **kw) -> LLMEngine:
    return LLMEngine(
        model, tokenizer=None, num_blocks=num_blocks, block_size=BLOCK_SIZE, max_num_seqs=max_num_seqs, **kw
    )


def _reference(model, prompt, max_tokens):
    return generate_naive(model, prompt, max_tokens, eos_token_id=None)


def _run_to_completion(engine, max_steps=200):
    steps = 0
    while engine.has_work():
        engine.step()
        steps += 1
        assert steps < max_steps, "engine did not finish"
    return steps


PROMPTS = [random_tokens(n, 128, seed=100 + n) for n in (3, 6, 9)]


# ---------------------------------------------------------------------------
# 5.4 batched model execution
# ---------------------------------------------------------------------------


@pytest.mark.m5
@pytest.mark.checkpoint("5.4")
class TestBatchedExecution:
    def test_single_request_matches_naive(self, model):
        engine = _engine(model)
        req = engine.add_request(PROMPTS[1], max_tokens=6)
        _run_to_completion(engine)
        assert req.status is RequestStatus.FINISHED
        assert req.generated_tokens == _reference(model, PROMPTS[1], 6)

    def test_multiple_requests_match_naive(self, model):
        engine = _engine(model)
        reqs = [engine.add_request(p, max_tokens=m) for p, m in zip(PROMPTS, (5, 8, 3))]
        _run_to_completion(engine)
        for req, prompt, max_tokens in zip(reqs, PROMPTS, (5, 8, 3)):
            assert req.status is RequestStatus.FINISHED
            assert req.generated_tokens == _reference(model, prompt, max_tokens), f"request {req.request_id}"

    def test_status_transitions(self, model):
        engine = _engine(model)
        req = engine.add_request(PROMPTS[0], max_tokens=3)
        assert req.status is RequestStatus.WAITING
        finished = engine.step()  # prefill -> 1 token
        assert req.status is RequestStatus.RUNNING and finished == []
        engine.step()  # 2 tokens
        assert req.status is RequestStatus.RUNNING
        finished = engine.step()  # 3 tokens -> done
        assert finished == [req] and req.status is RequestStatus.FINISHED
        assert not engine.has_work()
        assert engine.step() == []

    def test_decode_is_one_forward_for_the_whole_batch(self, model):
        recording = RecordingModel(model)
        engine = _engine(recording)
        for p in PROMPTS:
            engine.add_request(p, max_tokens=5)
        _run_to_completion(engine)
        prefill_calls = [c for c in recording.calls if c["input_ids"].shape[1] > 1]
        decode_calls = [c for c in recording.calls if c["input_ids"].shape[1] == 1]
        assert len(prefill_calls) == 3  # one per request
        assert sorted(c["input_ids"].shape[1] for c in prefill_calls) == [3, 6, 9]
        assert len(decode_calls) == 4  # 5 tokens = 1 from prefill + 4 decode steps ...
        assert all(c["input_ids"].shape[0] == 3 for c in decode_calls)  # ... each covering all 3 requests
        for c in recording.calls:
            assert isinstance(c["kv_cache"], PagedKVCache)
            assert c["attn_metadata"] is not None

    def test_step_returns_each_finished_request_exactly_once(self, model):
        engine = _engine(model)
        reqs = [engine.add_request(p, max_tokens=m) for p, m in zip(PROMPTS, (2, 4, 6))]
        finished = []
        while engine.has_work():
            finished += engine.step()
        assert sorted(r.request_id for r in finished) == sorted(r.request_id for r in reqs)
        assert len(finished) == 3

    def test_result_does_not_depend_on_batch_mates(self, model):
        alone = _engine(model)
        r_alone = alone.add_request(PROMPTS[2], max_tokens=7)
        _run_to_completion(alone)

        together = _engine(model)
        together.add_request(PROMPTS[0], max_tokens=7)
        r_together = together.add_request(PROMPTS[2], max_tokens=7)
        together.add_request(PROMPTS[1], max_tokens=2)
        _run_to_completion(together)
        assert r_together.generated_tokens == r_alone.generated_tokens

    def test_requests_added_mid_run(self, model):
        engine = _engine(model)
        a = engine.add_request(PROMPTS[0], max_tokens=6)
        engine.step()
        engine.step()
        b = engine.add_request(PROMPTS[1], max_tokens=4)
        _run_to_completion(engine)
        assert a.generated_tokens == _reference(model, PROMPTS[0], 6)
        assert b.generated_tokens == _reference(model, PROMPTS[1], 4)

    def test_provided_validation(self, model):
        engine = _engine(model, num_blocks=2)
        with pytest.raises(ValueError):
            engine.add_request("text prompt", max_tokens=1)  # no tokenizer
        with pytest.raises(ValueError):
            engine.add_request([], max_tokens=1)
        with pytest.raises(ValueError):
            engine.add_request([5], max_tokens=0)
        with pytest.raises(ValueError):
            engine.add_request(list(range(4, 10)), max_tokens=4)  # 10 tokens > 2 blocks * 4


# ---------------------------------------------------------------------------
# 5.5 per-request KV mappings
# ---------------------------------------------------------------------------


@pytest.mark.m5
@pytest.mark.checkpoint("5.5")
class TestPerRequestKVMappings:
    def test_block_tables_are_disjoint_at_every_step(self, model):
        engine = _engine(model, num_blocks=32)
        for p, m in zip(PROMPTS, (6, 9, 4)):
            engine.add_request(p, max_tokens=m)
        while engine.has_work():
            engine.step()
            assert_block_tables_disjoint(engine.scheduler.running)

    def test_block_tables_grow_lazily(self, model):
        engine = _engine(model, num_blocks=32)
        for p, m in zip(PROMPTS, (6, 9, 4)):
            engine.add_request(p, max_tokens=m)
        while engine.has_work():
            engine.step()
            for req in engine.scheduler.running:
                # KV exists for num_tokens - 1 positions (the newest token is not processed yet)
                lo = blocks_needed(req.num_tokens - 1, BLOCK_SIZE)
                hi = blocks_needed(req.num_tokens, BLOCK_SIZE)
                assert lo <= len(req.block_table) <= hi, f"request {req.request_id}: {req.block_table}"

    def test_decode_metadata_describes_every_request(self, model):
        recording = RecordingModel(model)
        engine = _engine(recording)
        reqs = [engine.add_request(p, max_tokens=4) for p in PROMPTS]
        engine.step()  # prefill all three
        recording.calls.clear()
        engine.step()  # one decode step
        assert len(recording.calls) == 1
        call = recording.calls[0]
        md = call["attn_metadata"]
        assert call["input_ids"].shape == (3, 1)
        assert call["positions"].shape == (3, 1)
        assert md.slot_mapping.shape == (3,)
        assert len(md.block_tables) == 3 and len(md.seq_lens) == 3
        # after this step each request has prompt + 2 tokens; the decode ran with prompt + 1
        assert sorted(md.seq_lens) == sorted(len(p) + 1 for p in PROMPTS)
        assert sorted(call["positions"].reshape(-1).tolist()) == sorted(len(p) for p in PROMPTS)
        assert len(set(md.slot_mapping.tolist())) == 3
        assert sorted(md.block_tables) == sorted(r.block_table for r in reqs)

    def test_kv_of_each_request_matches_single_request_cache(self, model, tiny_config):
        """No request reads or overwrites another's KV: the paged pool content for request R
        equals a private contiguous cache filled with R's own tokens."""
        engine = _engine(model, num_blocks=32)
        for p, m in zip(PROMPTS, (5, 7, 6)):
            engine.add_request(p, max_tokens=m)
        checked = 0
        while engine.has_work():
            engine.step()
            for req in engine.scheduler.running:
                computed = req.all_tokens[:-1]  # tokens whose KV is in the cache
                private = ContiguousKVCache(tiny_config, max_seq_len=32)
                prefill(model, private, computed)
                for layer in (0, tiny_config.num_layers - 1):
                    k_ref, v_ref = private.read(layer, len(computed))
                    k_paged = gather_kv(engine.kv_cache.k_cache[layer], req.block_table, len(computed))
                    v_paged = gather_kv(engine.kv_cache.v_cache[layer], req.block_table, len(computed))
                    torch.testing.assert_close(k_paged, k_ref, rtol=1e-4, atol=1e-4)
                    torch.testing.assert_close(v_paged, v_ref, rtol=1e-4, atol=1e-4)
                checked += 1
        assert checked > 5

    def test_shared_pool_stays_consistent(self, model):
        engine = _engine(model, num_blocks=32)
        for p, m in zip(PROMPTS, (6, 9, 4)):
            engine.add_request(p, max_tokens=m)
        while engine.has_work():
            engine.step()
            live = sum(len(r.block_table) for r in engine.requests.values())
            assert engine.block_pool.num_free_blocks + live == 32


@pytest.mark.m5
@pytest.mark.integration
def test_m5_integration_multi_request_engine(model):
    engine = _engine(model, num_blocks=48, max_num_seqs=4)
    specs = [(PROMPTS[0], 8), (PROMPTS[1], 3), (PROMPTS[2], 6), (random_tokens(5, 128, seed=7), 5)]
    reqs = [engine.add_request(p, max_tokens=m) for p, m in specs]
    assert all(r.status is RequestStatus.WAITING for r in reqs)
    while engine.has_work():
        engine.step()
        assert_block_tables_disjoint(engine.scheduler.running)
    for req, (prompt, max_tokens) in zip(reqs, specs):
        assert req.status is RequestStatus.FINISHED
        assert req.generated_tokens == _reference(model, prompt, max_tokens)


# ---------------------------------------------------------------------------
# Milestone 6
# ---------------------------------------------------------------------------


@pytest.mark.m6
@pytest.mark.checkpoint("6.1")
class TestRequestCompletion:
    def test_finished_exactly_when_max_tokens_reached(self, model):
        engine = _engine(model)
        a = engine.add_request(PROMPTS[0], max_tokens=2)
        b = engine.add_request(PROMPTS[1], max_tokens=4)
        assert engine.step() == []  # prefill: 1 token each
        assert engine.step() == [a]  # a has 2
        assert a.status is RequestStatus.FINISHED and a not in engine.scheduler.running
        assert b.status is RequestStatus.RUNNING
        assert engine.step() == []
        assert engine.step() == [b]
        assert not engine.has_work()

    def test_eos_finishes_a_request(self, tiny_model, tiny_config):
        # tiny_model has eos_token_id=2; find a prompt whose greedy continuation hits it, else
        # emulate by making EOS the first generated token.
        prompt = PROMPTS[1]
        first = generate_naive(tiny_model, prompt, 1)[0]
        engine = LLMEngine(tiny_model, num_blocks=16, block_size=BLOCK_SIZE, eos_token_id=first)
        req = engine.add_request(prompt, max_tokens=10)
        finished = engine.step()
        assert finished == [req]
        assert req.generated_tokens == [first]
        assert not engine.has_work()

    def test_finished_requests_do_not_participate_in_later_steps(self, model):
        recording = RecordingModel(model)
        engine = _engine(recording)
        engine.add_request(PROMPTS[0], max_tokens=1)
        b = engine.add_request(PROMPTS[1], max_tokens=3)
        engine.step()  # prefill both; a finishes immediately
        recording.calls.clear()
        engine.step()
        assert recording.calls[0]["input_ids"].shape == (1, 1)
        assert recording.calls[0]["attn_metadata"].block_tables == [b.block_table]


@pytest.mark.m6
@pytest.mark.checkpoint("6.2")
class TestDynamicAdmission:
    def test_new_request_starts_while_others_still_run(self, model):
        engine = _engine(model, max_num_seqs=2)
        specs = [(PROMPTS[0], 8), (PROMPTS[1], 2), (PROMPTS[2], 3), (random_tokens(4, 128, seed=8), 2)]
        a, b, c, d = [engine.add_request(p, max_tokens=m) for p, m in specs]

        timeline = []
        while engine.has_work():
            engine.step()
            timeline.append({r.request_id for r in engine.scheduler.running})
        # c must have been running at the same time as a, and so must d
        assert any(a.request_id in s and c.request_id in s for s in timeline)
        assert any(a.request_id in s and d.request_id in s for s in timeline)
        # never more than max_num_seqs running
        assert all(len(s) <= 2 for s in timeline)
        # c was admitted right after b finished
        assert timeline[1] == {a.request_id}  # b finished in step 2; c not yet prefilled
        assert timeline[2] == {a.request_id, c.request_id}
        for req, (prompt, max_tokens) in zip((a, b, c, d), specs):
            assert req.generated_tokens == _reference(model, prompt, max_tokens)

    def test_admission_waits_for_block_budget(self, model):
        # pool of 4 blocks; each request's worst case is 2 blocks -> two at a time
        engine = _engine(model, num_blocks=4, max_num_seqs=8)
        specs = [(PROMPTS[0], 5), (random_tokens(4, 128, seed=9), 4), (random_tokens(2, 128, seed=10), 6)]
        a, b, c = [engine.add_request(p, max_tokens=m) for p, m in specs]
        engine.step()
        assert a.status is RequestStatus.RUNNING and b.status is RequestStatus.RUNNING
        assert c.status is RequestStatus.WAITING
        while b.status is not RequestStatus.FINISHED:
            engine.step()
        engine.step()
        assert c.status is RequestStatus.RUNNING
        _run_to_completion(engine)
        for req, (prompt, max_tokens) in zip((a, b, c), specs):
            assert req.generated_tokens == _reference(model, prompt, max_tokens)

    def test_pool_is_never_exhausted(self, model):
        """Many requests through a small pool: admission control alone must prevent OutOfBlocksError."""
        engine = _engine(model, num_blocks=6, max_num_seqs=8)
        specs = [(random_tokens(2 + i % 5, 128, seed=20 + i), 1 + (i * 3) % 7) for i in range(10)]
        reqs = [engine.add_request(p, max_tokens=m) for p, m in specs]
        _run_to_completion(engine, max_steps=500)
        for req, (prompt, max_tokens) in zip(reqs, specs):
            assert req.generated_tokens == _reference(model, prompt, max_tokens), f"request {req.request_id}"
        assert engine.block_pool.num_free_blocks == 6


@pytest.mark.m6
@pytest.mark.checkpoint("6.3")
class TestDynamicBlockAllocation:
    def test_blocks_are_appended_only_at_block_boundaries(self, model):
        engine = _engine(model, num_blocks=16)
        req = engine.add_request(PROMPTS[0], max_tokens=10)  # 3 prompt tokens + 10 -> up to 13 tokens
        history = []
        while engine.has_work():
            engine.step()
            if req.status is RequestStatus.RUNNING:
                history.append((req.num_tokens, len(req.block_table)))
        assert history[0][1] == blocks_needed(len(PROMPTS[0]), BLOCK_SIZE) == 1  # prompt only
        lengths = [n for _, n in history]
        assert lengths == sorted(lengths)  # never shrinks while running
        assert max(lengths) >= 3  # KV for 12 tokens needs 3 blocks
        assert max(lengths) <= blocks_needed(13, BLOCK_SIZE)  # never more than the worst case
        for num_tokens, n in history:
            assert blocks_needed(num_tokens - 1, BLOCK_SIZE) <= n <= blocks_needed(num_tokens, BLOCK_SIZE)

    def test_prefill_does_not_reserve_the_worst_case(self, model):
        engine = _engine(model, num_blocks=16)
        req = engine.add_request(PROMPTS[0], max_tokens=20)
        engine.step()
        assert len(req.block_table) == 1
        assert engine.block_pool.num_free_blocks == 15


@pytest.mark.m6
@pytest.mark.checkpoint("6.4")
class TestBlockReclamation:
    def test_blocks_freed_when_request_finishes(self, model):
        engine = _engine(model, num_blocks=16)
        a = engine.add_request(PROMPTS[2], max_tokens=6)  # 9 + 6: ends holding 4 blocks
        b = engine.add_request(PROMPTS[0], max_tokens=12)
        while a.status is not RequestStatus.FINISHED:
            before_free = engine.block_pool.num_free_blocks
            a_blocks = len(a.block_table)
            b_blocks_before = len(b.block_table)
            engine.step()
        assert a.block_table == []
        b_growth = len(b.block_table) - b_blocks_before
        assert engine.block_pool.num_free_blocks == before_free + a_blocks - b_growth

    def test_pool_fully_free_at_the_end(self, model):
        engine = _engine(model, num_blocks=24)
        for p, m in zip(PROMPTS, (6, 9, 4)):
            engine.add_request(p, max_tokens=m)
        _run_to_completion(engine)
        assert engine.block_pool.num_free_blocks == 24
        assert all(r.block_table == [] for r in engine.requests.values())

    def test_accounting_is_exact_at_every_step(self, model):
        engine = _engine(model, num_blocks=24, max_num_seqs=2)
        for i in range(5):
            engine.add_request(random_tokens(2 + i, 128, seed=30 + i), max_tokens=3 + i)
        while engine.has_work():
            engine.step()
            live = sum(len(r.block_table) for r in engine.scheduler.running)
            assert sum(len(r.block_table) for r in engine.requests.values()) == live  # finished hold nothing
            assert engine.block_pool.num_free_blocks + live == 24


@pytest.mark.m6
@pytest.mark.checkpoint("6.5")
class TestBlockReuse:
    def test_new_request_receives_blocks_of_a_finished_one(self, model):
        """Leave exactly two blocks free. B fills both, finishes, and C can only get B's blocks."""
        engine = _engine(model, num_blocks=8, max_num_seqs=8)
        held = [engine.block_pool.allocate() for _ in range(6)]  # simulate other tenants
        b = engine.add_request(random_tokens(7, 128, seed=40), max_tokens=1)  # 7 + 1 -> 2 blocks, uses both
        c = engine.add_request(random_tokens(4, 128, seed=41), max_tokens=4)  # 4 + 4 -> 2 blocks
        engine.step()  # prefill b -> finishes immediately, c still waiting (no budget)
        assert b.status is RequestStatus.FINISHED
        b_blocks = set(range(8)) - set(held)
        assert len(b_blocks) == 2 and b.block_table == []
        assert c.status is RequestStatus.WAITING
        engine.step()  # c admitted + prefilled
        assert c.status is RequestStatus.RUNNING
        assert set(c.block_table) <= b_blocks
        _run_to_completion(engine)
        assert set(c.block_table) == set()  # freed again
        assert engine.block_pool.num_free_blocks == 2
        assert c.generated_tokens == _reference(model, c.prompt_tokens, 4)

    def test_blocks_cycle_through_many_requests(self, model):
        engine = _engine(model, num_blocks=4, max_num_seqs=8)
        used_by: dict[int, set[int]] = {}
        reqs = [engine.add_request(random_tokens(3, 128, seed=50 + i), max_tokens=5) for i in range(6)]
        while engine.has_work():
            engine.step()
            for r in engine.scheduler.running:
                used_by.setdefault(r.request_id, set()).update(r.block_table)
        assert all(used_by[r.request_id] for r in reqs)
        # 6 requests x 2 blocks each, only 4 physical blocks: reuse is unavoidable and must be clean
        assert set().union(*used_by.values()) <= set(range(4))
        for r in reqs:
            assert r.generated_tokens == _reference(model, r.prompt_tokens, 5)


@pytest.mark.m6
@pytest.mark.checkpoint("6.6")
class TestContinuousBatchingLoop:
    def test_run_yields_every_request_as_it_finishes(self, model):
        engine = _engine(model, max_num_seqs=2)
        specs = [(PROMPTS[0], 6), (PROMPTS[1], 2), (PROMPTS[2], 4)]
        reqs = [engine.add_request(p, max_tokens=m) for p, m in specs]
        order = [r.request_id for r in engine.run()]
        assert sorted(order) == [0, 1, 2]
        assert order[0] == reqs[1].request_id  # the shortest finishes first
        assert not engine.has_work()
        for req, (prompt, max_tokens) in zip(reqs, specs):
            assert req.status is RequestStatus.FINISHED
            assert req.generated_tokens == _reference(model, prompt, max_tokens)

    def test_format_state_reflects_block_ownership(self, model):
        engine = _engine(model, num_blocks=8)
        a = engine.add_request(PROMPTS[2], max_tokens=4)  # 9 tokens -> 3 blocks after prefill
        engine.step()
        text = engine.format_state()
        assert f"Request {a.request_id}" in text
        assert f"block_table = {a.block_table}" in text
        assert f"{a.block_table[0]} {a.request_id} [tokens 0-3]" in text
        assert f"{a.block_table[2]} {a.request_id} [tokens 8-9]" in text
        assert text.count("FREE") == 5

    def test_log_blocks_prints_after_each_step(self, model, capsys):
        engine = _engine(model, log_blocks=True)
        engine.add_request(PROMPTS[0], max_tokens=2)
        list(engine.run())
        out = capsys.readouterr().out
        assert out.count("Physical KV Pool") == 2


@pytest.mark.m6
@pytest.mark.integration
def test_m6_integration_continuous_batching(model):
    """The plan's demonstration: B finishes early, its block is freed, D is admitted and reuses it,
    everything stays correct, and the pool is clean at the end."""
    engine = _engine(model, num_blocks=6, max_num_seqs=3)
    specs = [
        (random_tokens(4, 128, seed=60), 4),  # A: 2 blocks worst case
        (random_tokens(3, 128, seed=61), 1),  # B: 1 block, finishes at prefill
        (random_tokens(4, 128, seed=62), 4),  # C: 2 blocks
        (random_tokens(2, 128, seed=63), 2),  # D: 1 block -> can only be admitted after B leaves
    ]
    a, b, c, d = [engine.add_request(p, max_tokens=m) for p, m in specs]

    finished = engine.step()  # t0: A, B, C prefilled; B done
    assert finished == [b]
    assert {r.request_id for r in engine.scheduler.running} == {a.request_id, c.request_id}
    assert d.status is RequestStatus.WAITING
    assert b.block_table == []
    # Everything not owned by A or C (including B's former block) must be free again.
    free_now = set(range(6)) - set(a.block_table) - set(c.block_table)
    assert engine.block_pool.num_free_blocks == len(free_now) == 4

    engine.step()  # t1: D admitted and prefilled
    assert d.status is RequestStatus.RUNNING
    assert set(d.block_table) <= free_now
    assert_block_tables_disjoint(engine.scheduler.running)

    while engine.has_work():
        engine.step()
        assert_block_tables_disjoint(engine.scheduler.running)
    for req, (prompt, max_tokens) in zip((a, b, c, d), specs):
        assert req.generated_tokens == _reference(model, prompt, max_tokens)
    assert engine.block_pool.num_free_blocks == 6
