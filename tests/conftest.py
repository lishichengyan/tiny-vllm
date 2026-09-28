"""Shared fixtures and the milestone progress banner.

Run one milestone at a time:

    pytest -m m1            # Milestone 1 only
    pytest -m "m2 and not hf"

Tests marked ``hf`` download the real HuggingFace model (about 270 MB) the first
time; they are skipped automatically when the download is impossible.
"""

import os
import re
from collections import defaultdict

import pytest
import torch

from tiny_vllm.config import TinyLlamaConfig
from tiny_vllm.model import TinyLlama

# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


@pytest.fixture(autouse=True)
def _seed_everything():
    torch.manual_seed(0)
    yield


# ---------------------------------------------------------------------------
# Small random model (no download required)
# ---------------------------------------------------------------------------

TINY = dict(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=128,
    num_layers=2,
    num_heads=4,
    num_kv_heads=2,
    max_position_embeddings=256,
    rms_norm_eps=1e-5,
    rope_theta=10000.0,
    tie_word_embeddings=False,
    bos_token_id=1,
    eos_token_id=2,
)


@pytest.fixture
def tiny_config() -> TinyLlamaConfig:
    return TinyLlamaConfig(**TINY)


@pytest.fixture(scope="session")
def hf_tiny_config():
    from transformers import LlamaConfig

    return LlamaConfig(
        vocab_size=TINY["vocab_size"],
        hidden_size=TINY["hidden_size"],
        intermediate_size=TINY["intermediate_size"],
        num_hidden_layers=TINY["num_layers"],
        num_attention_heads=TINY["num_heads"],
        num_key_value_heads=TINY["num_kv_heads"],
        max_position_embeddings=TINY["max_position_embeddings"],
        rms_norm_eps=TINY["rms_norm_eps"],
        rope_theta=TINY["rope_theta"],
        tie_word_embeddings=TINY["tie_word_embeddings"],
        bos_token_id=TINY["bos_token_id"],
        eos_token_id=TINY["eos_token_id"],
        attn_implementation="eager",
    )


@pytest.fixture(scope="session")
def hf_tiny_model(hf_tiny_config):
    """Randomly initialised HuggingFace LlamaForCausalLM with a fixed seed. Never mutate it."""
    from transformers import LlamaForCausalLM

    torch.manual_seed(1234)
    model = LlamaForCausalLM(hf_tiny_config).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@pytest.fixture
def tiny_model(tiny_config) -> TinyLlama:
    """Our TinyLlama with random (seeded) weights. Independent of weight loading (1.5)."""
    torch.manual_seed(42)
    model = TinyLlama(tiny_config).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@pytest.fixture
def loaded_tiny_model(tiny_config, hf_tiny_model) -> TinyLlama:
    """Our TinyLlama carrying the weights of ``hf_tiny_model`` (requires checkpoint 1.5)."""
    from tiny_vllm.loader import load_weights

    model = TinyLlama(tiny_config)
    load_weights(model, hf_tiny_model.state_dict())
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# Real model (download required, marker ``hf``)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def real_model_and_tokenizer():
    from tiny_vllm.loader import load_tiny_llama

    try:
        model, tokenizer = load_tiny_llama()
    except NotImplementedError:
        raise
    except Exception as exc:  # network / cache problems
        pytest.skip(f"real HuggingFace model unavailable: {exc!r}")
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


@pytest.fixture(scope="session")
def real_hf_model():
    from tiny_vllm.loader import load_hf_model

    try:
        model = load_hf_model()
    except Exception as exc:
        pytest.skip(f"real HuggingFace model unavailable: {exc!r}")
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# Milestone progress banner
# ---------------------------------------------------------------------------

MILESTONES = {
    1: (
        "TinyLlama",
        "Generate tokens autoregressively and cache K/V\nso decode only touches the newest token.",
    ),
    2: (
        "Generation + KV Cache",
        "Replace the per-request contiguous cache with a\nglobal pool of fixed-size physical blocks.",
    ),
    3: ("Block-Based KV Cache", "Use the block table to perform attention\nover physically scattered KV."),
    4: (
        "Paged Attention",
        "Serve several requests at once: Request, Scheduler,\nbatched execution over the shared KV pool.",
    ),
    5: (
        "Multi-Request Engine",
        "Free blocks the moment a request finishes and admit\nnew requests while others are still running.",
    ),
    6: ("Continuous Batching", "Read docs/07-real-vllm.md to see how real vLLM\ngoes beyond tiny-vllm."),
}

CHECKPOINTS = {
    "1.1": "RMSNorm matches HuggingFace",
    "1.2": "RoPE matches HuggingFace",
    "1.3": "self-attention matches HuggingFace",
    "1.4": "MLP + decoder layer match HuggingFace",
    "1.5": "HuggingFace weights load into TinyLlama",
    "1.6": "TinyLlama logits ≈ HuggingFace logits",
    "2.1": "naive autoregressive generation works",
    "2.2": "contiguous KV cache stores K/V by position",
    "2.3": "prefill fills the cache",
    "2.4": "decode processes only the newest token",
    "2.5": "cached generation == uncached generation",
    "3.1": "global KV pool works",
    "3.2": "block allocation works",
    "3.3": "block reclamation works",
    "3.4": "block table works",
    "3.5": "slot mapping works",
    "3.6": "physical KV writes are correct",
    "4.1": "logical token -> physical KV resolution works",
    "4.2": "K/V read through the block table",
    "4.3": "attention over scattered KV matches contiguous attention",
    "4.4": "partial final block handled",
    "4.5": "TinyLlama generates over the paged KV cache",
    "4.6": "blockwise decode attention (online softmax, no gather)",
    "5.1": "Request abstraction works",
    "5.2": "waiting / running / finished transitions work",
    "5.3": "scheduler admits and batches requests",
    "5.4": "batched model execution works",
    "5.5": "per-request KV mappings are isolated",
    "6.1": "request completion detected",
    "6.2": "requests admitted dynamically",
    "6.3": "blocks allocated dynamically during decode",
    "6.4": "blocks reclaimed on completion",
    "6.5": "reclaimed blocks reused",
    "6.6": "continuous batching loop works",
}

_checkpoint_results: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "failed": 0, "skipped": 0})
_integration_results: dict[int, dict[str, int]] = defaultdict(
    lambda: {"passed": 0, "failed": 0, "skipped": 0}
)
_item_checkpoints: dict[str, list[str]] = {}
_item_integration: dict[str, int] = {}


def pytest_collection_modifyitems(config, items):
    for item in items:
        checkpoints = []
        for marker in item.iter_markers(name="checkpoint"):
            checkpoints.extend(str(a) for a in marker.args)
        _item_checkpoints[item.nodeid] = checkpoints
        if item.get_closest_marker("integration") is not None:
            for milestone in MILESTONES:
                if item.get_closest_marker(f"m{milestone}") is not None:
                    _item_integration[item.nodeid] = milestone


def pytest_runtest_logreport(report):
    if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
        outcome = report.outcome  # passed / failed / skipped
        for cp in _item_checkpoints.get(report.nodeid, []):
            _checkpoint_results[cp][outcome] += 1
        milestone = _item_integration.get(report.nodeid)
        if milestone is not None:
            _integration_results[milestone][outcome] += 1


def _status_symbol(counts: dict[str, int]) -> str:
    if counts["failed"]:
        return "✗"
    if counts["passed"]:
        return "✓"
    return "-"


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_sessionfinish(session, exitstatus):
    """Print the milestone checklist as the very last output (outermost wrapper)."""
    result = yield
    config = session.config
    terminalreporter = config.pluginmanager.get_plugin("terminalreporter")
    match = re.fullmatch(r"\s*m([1-6])\s*(and\s+not\s+hf\s*)?", config.option.markexpr or "")
    if terminalreporter is None or not match:
        return result
    _print_banner(terminalreporter, exitstatus, int(match.group(1)))
    return result


def _print_banner(terminalreporter, exitstatus, milestone):
    name, next_text = MILESTONES[milestone]
    prefix = f"{milestone}."
    ids = [cp for cp in CHECKPOINTS if cp.startswith(prefix)]

    lines = []
    all_green = True
    for cp in ids:
        counts = _checkpoint_results.get(cp, {"passed": 0, "failed": 0, "skipped": 0})
        symbol = _status_symbol(counts)
        all_green &= symbol == "✓"
        lines.append(f"{symbol} {cp}  {CHECKPOINTS[cp]}")
    integ = _integration_results.get(milestone, {"passed": 0, "failed": 0, "skipped": 0})
    symbol = _status_symbol(integ)
    all_green &= symbol == "✓"
    lines.append(f"{symbol} integration test")

    tr = terminalreporter
    tr.write_line("")
    tr.write_line("=" * 40)
    if all_green and exitstatus == 0:
        tr.write_line(f"MILESTONE {milestone} COMPLETE")
    else:
        tr.write_line(f"MILESTONE {milestone} — in progress")
    tr.write_line(name)
    tr.write_line("=" * 40)
    tr.write_line("")
    for line in lines:
        tr.write_line(line)
    tr.write_line("")
    if all_green and exitstatus == 0:
        tr.write_line("Next:")
        tr.write_line(next_text)
    else:
        tr.write_line('Remaining work:  grep -R "TODO(student)" tiny_vllm/')
    tr.write_line("")
