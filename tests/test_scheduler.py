"""Milestone 5 -- checkpoints 5.1 (Request), 5.2 (waiting/running/finished), 5.3 (scheduler).
Milestone 6 -- checkpoint 6.2 (dynamic admission at the scheduler level)."""

import pytest

from tiny_vllm.block_pool import BlockPool
from tiny_vllm.request import Request, RequestStatus
from tiny_vllm.scheduler import Scheduler

EOS = 2
BLOCK_SIZE = 4


def _req(request_id=0, prompt_len=5, max_tokens=4) -> Request:
    return Request(request_id, list(range(10, 10 + prompt_len)), max_tokens)


# ---------------------------------------------------------------------------
# 5.1 Request
# ---------------------------------------------------------------------------


@pytest.mark.m5
@pytest.mark.checkpoint("5.1")
class TestRequest:
    def test_initial_state(self):
        req = Request(7, [1, 2, 3], max_tokens=4)
        assert req.request_id == 7
        assert req.prompt_tokens == [1, 2, 3]
        assert req.generated_tokens == []
        assert req.status is RequestStatus.WAITING
        assert req.max_tokens == 4
        assert req.block_table == []
        assert req.num_prompt_tokens == 3
        assert req.num_tokens == 3
        assert req.all_tokens == [1, 2, 3]
        assert req.last_token == 3

    def test_prompt_is_copied(self):
        prompt = [1, 2, 3]
        req = Request(0, prompt, 4)
        prompt.append(99)
        assert req.prompt_tokens == [1, 2, 3]

    def test_invalid_arguments(self):
        with pytest.raises(ValueError):
            Request(0, [], 4)
        with pytest.raises(ValueError):
            Request(0, [1], 0)

    def test_append_token_updates_sequence(self):
        req = Request(0, [1, 2, 3], 4)
        req.append_token(50)
        req.append_token(60)
        assert req.generated_tokens == [50, 60]
        assert req.num_tokens == 5
        assert req.all_tokens == [1, 2, 3, 50, 60]
        assert req.last_token == 60
        assert req.prompt_tokens == [1, 2, 3]  # prompt untouched

    def test_should_stop_on_max_tokens(self):
        req = Request(0, [1, 2, 3], max_tokens=2)
        assert not req.should_stop(EOS)
        req.append_token(50)
        assert not req.should_stop(EOS)
        req.append_token(51)
        assert req.should_stop(EOS)

    def test_should_stop_on_eos(self):
        req = Request(0, [1, 2, 3], max_tokens=10)
        req.append_token(50)
        assert not req.should_stop(EOS)
        req.append_token(EOS)
        assert req.should_stop(EOS)

    def test_eos_none_disables_eos_stopping(self):
        req = Request(0, [1, 2, 3], max_tokens=10)
        req.append_token(EOS)
        assert not req.should_stop(None)

    def test_eos_in_prompt_does_not_stop(self):
        req = Request(0, [1, EOS, 3], max_tokens=10)
        assert not req.should_stop(EOS)
        req.append_token(50)
        assert not req.should_stop(EOS)

    def test_max_tokens_one(self):
        req = Request(0, [1], max_tokens=1)
        req.append_token(9)
        assert req.should_stop(None)

    def test_repr_does_not_crash(self):
        assert "Request" in repr(Request(0, [1], 1))


# ---------------------------------------------------------------------------
# 5.2 waiting / running / finished
# ---------------------------------------------------------------------------


@pytest.mark.m5
@pytest.mark.checkpoint("5.2")
class TestRequestStates:
    def test_add_request_goes_to_waiting(self):
        sched = Scheduler(BlockPool(16), BLOCK_SIZE, max_num_seqs=4)
        assert not sched.has_unfinished_requests()
        req = _req(0)
        sched.add_request(req)
        assert sched.waiting == [req]
        assert sched.running == []
        assert req.status is RequestStatus.WAITING
        assert sched.has_unfinished_requests()

    def test_fifo_order_in_waiting(self):
        sched = Scheduler(BlockPool(16), BLOCK_SIZE, max_num_seqs=4)
        reqs = [_req(i) for i in range(3)]
        for r in reqs:
            sched.add_request(r)
        assert [r.request_id for r in sched.waiting] == [0, 1, 2]

    def test_schedule_moves_to_running(self):
        sched = Scheduler(BlockPool(16), BLOCK_SIZE, max_num_seqs=4)
        req = _req(0)
        sched.add_request(req)
        batch, is_prefill = sched.schedule()
        assert batch == [req] and is_prefill is True
        assert req.status is RequestStatus.RUNNING
        assert sched.waiting == [] and sched.running == [req]
        assert sched.has_unfinished_requests()

    def test_finish_removes_from_running(self):
        sched = Scheduler(BlockPool(16), BLOCK_SIZE, max_num_seqs=4)
        a, b = _req(0), _req(1)
        sched.add_request(a)
        sched.add_request(b)
        sched.schedule()
        sched.finish(a)
        assert a.status is RequestStatus.FINISHED
        assert b.status is RequestStatus.RUNNING
        assert sched.running == [b]
        assert sched.has_unfinished_requests()
        sched.finish(b)
        assert sched.running == []
        assert not sched.has_unfinished_requests()

    def test_finished_request_is_never_scheduled_again(self):
        sched = Scheduler(BlockPool(16), BLOCK_SIZE, max_num_seqs=4)
        a = _req(0)
        sched.add_request(a)
        sched.schedule()
        sched.finish(a)
        batch, is_prefill = sched.schedule()
        assert batch == [] and is_prefill is False


# ---------------------------------------------------------------------------
# 5.3 scheduling policy
# ---------------------------------------------------------------------------


@pytest.mark.m5
@pytest.mark.checkpoint("5.3")
class TestSchedulePolicy:
    def test_empty_scheduler(self):
        sched = Scheduler(BlockPool(16), BLOCK_SIZE, max_num_seqs=4)
        assert sched.schedule() == ([], False)

    def test_prefill_batch_then_decode_batch(self):
        sched = Scheduler(BlockPool(64), BLOCK_SIZE, max_num_seqs=4)
        reqs = [_req(i) for i in range(3)]
        for r in reqs:
            sched.add_request(r)
        batch, is_prefill = sched.schedule()
        assert is_prefill and batch == reqs  # all three fit: admitted in FIFO order
        batch, is_prefill = sched.schedule()
        assert not is_prefill and batch == reqs  # nothing new to admit: decode everyone
        assert batch is not sched.running  # a snapshot, not the live list

    def test_max_num_seqs_caps_running(self):
        sched = Scheduler(BlockPool(64), BLOCK_SIZE, max_num_seqs=2)
        reqs = [_req(i) for i in range(3)]
        for r in reqs:
            sched.add_request(r)
        batch, is_prefill = sched.schedule()
        assert is_prefill and batch == reqs[:2]
        assert sched.waiting == [reqs[2]] and reqs[2].status is RequestStatus.WAITING
        batch, is_prefill = sched.schedule()
        assert not is_prefill and batch == reqs[:2]

    def test_can_admit_respects_block_budget(self):
        # worst case = prompt 5 + max_tokens 4 = 9 tokens -> 3 blocks of 4
        sched = Scheduler(BlockPool(3), BLOCK_SIZE, max_num_seqs=8)
        assert sched.can_admit(_req(0, prompt_len=5, max_tokens=4))
        assert not sched.can_admit(_req(1, prompt_len=5, max_tokens=8))  # 13 tokens -> 4 blocks
        assert not sched.can_admit(_req(2, prompt_len=9, max_tokens=4))  # 13 tokens -> 4 blocks

    def test_can_admit_counts_only_free_blocks(self):
        pool = BlockPool(6)
        sched = Scheduler(pool, BLOCK_SIZE, max_num_seqs=8)
        req = _req(0, prompt_len=5, max_tokens=4)  # needs 3
        assert sched.can_admit(req)
        for _ in range(4):
            pool.allocate()  # 2 left
        assert not sched.can_admit(req)

    def test_can_admit_respects_max_num_seqs(self):
        sched = Scheduler(BlockPool(64), BLOCK_SIZE, max_num_seqs=1)
        sched.add_request(_req(0))
        sched.schedule()
        assert not sched.can_admit(_req(1))

    def test_can_admit_subtracts_blocks_promised_to_running_requests(self):
        """Free blocks that a running request will still grow into are not available."""
        pool = BlockPool(4)
        sched = Scheduler(pool, BLOCK_SIZE, max_num_seqs=8)
        running = _req(0, prompt_len=4, max_tokens=4)  # worst case 8 tokens -> 2 blocks
        sched.add_request(running)
        sched.schedule()
        running.block_table = [pool.allocate()]  # prefilled: holds 1 of its 2 blocks
        assert pool.num_free_blocks == 3
        # budget = 3 free - 1 promised = 2
        assert sched.can_admit(_req(1, prompt_len=4, max_tokens=4))  # needs 2
        assert not sched.can_admit(_req(2, prompt_len=5, max_tokens=4))  # needs 3

    def test_admissions_within_one_schedule_call_share_the_budget(self):
        """Nothing is allocated during schedule(), so the second admission must account for the first."""
        sched = Scheduler(BlockPool(5), BLOCK_SIZE, max_num_seqs=8)
        reqs = [_req(i, prompt_len=5, max_tokens=4) for i in range(2)]  # 3 blocks each
        for r in reqs:
            sched.add_request(r)
        batch, is_prefill = sched.schedule()
        assert is_prefill and batch == reqs[:1]
        assert sched.waiting == [reqs[1]]

    def test_block_budget_limits_admission(self):
        # 6 blocks; each request needs 3 -> exactly two fit
        sched = Scheduler(BlockPool(6), BLOCK_SIZE, max_num_seqs=8)
        reqs = [_req(i, prompt_len=5, max_tokens=4) for i in range(3)]
        for r in reqs:
            sched.add_request(r)
        batch, is_prefill = sched.schedule()
        assert is_prefill and batch == reqs[:2]
        assert sched.waiting == [reqs[2]]

    def test_admission_is_strictly_fifo(self):
        """A small request behind a too-big one must wait (no head-of-line skipping)."""
        sched = Scheduler(BlockPool(4), BLOCK_SIZE, max_num_seqs=8)
        big = _req(0, prompt_len=12, max_tokens=8)  # 20 tokens -> 5 blocks: never fits in 4
        small = _req(1, prompt_len=1, max_tokens=1)
        sched.add_request(big)
        sched.add_request(small)
        assert sched.schedule() == ([], False)
        assert sched.waiting == [big, small]

    def test_admission_does_not_touch_the_pool(self):
        pool = BlockPool(16)
        sched = Scheduler(pool, BLOCK_SIZE, max_num_seqs=4)
        sched.add_request(_req(0))
        sched.schedule()
        assert pool.num_free_blocks == 16  # blocks are allocated by the engine at prefill time


# ---------------------------------------------------------------------------
# 6.2 dynamic admission (scheduler level)
# ---------------------------------------------------------------------------


@pytest.mark.m6
@pytest.mark.checkpoint("6.2")
class TestDynamicAdmission:
    def test_waiting_request_admitted_when_a_slot_frees_up(self):
        sched = Scheduler(BlockPool(64), BLOCK_SIZE, max_num_seqs=2)
        a, b, c = _req(0), _req(1), _req(2)
        for r in (a, b, c):
            sched.add_request(r)
        assert sched.schedule() == ([a, b], True)
        assert sched.schedule() == ([a, b], False)
        sched.finish(b)
        assert sched.schedule() == ([c], True)  # c admitted the moment b left
        assert sched.running == [a, c]
        assert sched.schedule() == ([a, c], False)

    def test_waiting_request_admitted_when_blocks_free_up(self):
        pool = BlockPool(6)
        sched = Scheduler(pool, BLOCK_SIZE, max_num_seqs=8)
        a, b, c = (_req(i, prompt_len=5, max_tokens=4) for i in range(3))  # 3 blocks each
        for r in (a, b, c):
            sched.add_request(r)
        assert sched.schedule() == ([a, b], True)
        # the engine prefills + decodes: both requests grow into all their blocks
        a.block_table = [pool.allocate() for _ in range(3)]
        b.block_table = [pool.allocate() for _ in range(3)]
        assert sched.schedule() == ([a, b], False)
        sched.finish(a)
        assert sched.schedule() == ([b], False)  # a is gone, but its blocks are not freed yet
        for blk in a.block_table:
            pool.free(blk)
        a.block_table = []
        assert sched.schedule() == ([c], True)

    def test_requests_added_while_running_are_admitted(self):
        sched = Scheduler(BlockPool(64), BLOCK_SIZE, max_num_seqs=4)
        a = _req(0)
        sched.add_request(a)
        assert sched.schedule() == ([a], True)
        late = _req(1)
        sched.add_request(late)
        assert sched.schedule() == ([late], True)
        assert sched.running == [a, late]
