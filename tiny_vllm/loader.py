"""Loading HuggingFace config, tokenizer and weights (Milestone 1, checkpoint 1.5).

HuggingFace provides three things:

    config      -> TinyLlamaConfig.from_hf          (provided)
    tokenizer   -> load_tokenizer                   (provided)
    weights     -> load_hf_state_dict + load_weights (download provided, mapping is yours)

The HuggingFace *model implementation* is used only as a reference (``load_hf_model``)
for tests and experiments. The tiny-vllm inference path always runs ``TinyLlama``.
"""

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .config import TinyLlamaConfig
from .model import TinyLlama

# A genuinely small Llama-architecture model (135M params, 30 layers, GQA 9/3,
# tied embeddings, bf16 safetensors). Runs comfortably on a laptop CPU.
DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"


def load_hf_config(model_name: str = DEFAULT_MODEL):
    return AutoConfig.from_pretrained(model_name)


def load_tokenizer(model_name: str = DEFAULT_MODEL):
    return AutoTokenizer.from_pretrained(model_name)


def load_hf_model(model_name: str = DEFAULT_MODEL, dtype: torch.dtype = torch.float32):
    """Reference implementation. Only for comparisons, never on the tiny-vllm inference path."""
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, attn_implementation="eager")
    return model.eval()


def load_hf_state_dict(model_name: str = DEFAULT_MODEL) -> dict[str, torch.Tensor]:
    """Download the checkpoint and return its raw tensors keyed by HuggingFace parameter name.

    Only single-file ``model.safetensors`` checkpoints are supported (true for the default
    model). Note that models with ``tie_word_embeddings=True`` do NOT store ``lm_head.weight``.
    """
    path = hf_hub_download(model_name, "model.safetensors")
    return load_file(path)


def load_weights(model: TinyLlama, hf_state_dict: dict[str, torch.Tensor]) -> None:
    """Copy HuggingFace weights into our TinyLlama (checkpoint 1.5).

    HuggingFace names look like::

        model.embed_tokens.weight
        model.layers.0.self_attn.q_proj.weight
        model.layers.0.self_attn.k_proj.weight
        model.layers.0.self_attn.v_proj.weight
        model.layers.0.self_attn.o_proj.weight
        model.layers.0.mlp.gate_proj.weight
        model.layers.0.mlp.up_proj.weight
        model.layers.0.mlp.down_proj.weight
        model.layers.0.input_layernorm.weight
        model.layers.0.post_attention_layernorm.weight
        ...
        model.norm.weight
        lm_head.weight            (absent when tie_word_embeddings=True)

    Requirements:
        * Every parameter of ``model`` must receive a value. Raise ``ValueError`` naming
          the missing HuggingFace keys otherwise.
        * Every key in ``hf_state_dict`` must be consumed. Raise ``ValueError`` naming the
          unexpected keys otherwise. (Old checkpoints may contain ``rotary_emb.inv_freq``
          buffers; treat keys ending in ``inv_freq`` as ignorable.)
        * If ``model.config.tie_word_embeddings`` is True and ``lm_head.weight`` is absent,
          the embedding weight serves as the LM head (``TinyLlama.__init__`` already ties
          the two parameters, so loading ``embed_tokens`` is enough). If ``lm_head.weight``
          IS present for a tied model, it must equal ``embed_tokens.weight``.
        * Convert dtype: checkpoints are often bf16 while the model runs in float32.
        * Copy tensors in place (``param.data.copy_`` / ``torch.no_grad``); do not
          replace ``nn.Parameter`` objects.
        * Beware: ``model.named_parameters()`` yields a tied parameter only ONCE (under
          ``embed_tokens.weight``), so ``lm_head.weight`` never shows up in that loop.
    """
    # TODO(student): Milestone 1
    raise NotImplementedError("Milestone 1 (1.5): implement load_weights()")


def load_tiny_llama(
    model_name: str = DEFAULT_MODEL,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
):
    """Build a TinyLlama from a HuggingFace checkpoint.

    Returns:
        (model, tokenizer) with the model in eval mode on ``device``.
    """
    hf_config = load_hf_config(model_name)
    config = TinyLlamaConfig.from_hf(hf_config)
    model = TinyLlama(config).to(dtype=dtype)
    load_weights(model, load_hf_state_dict(model_name))
    model = model.to(device).eval()
    tokenizer = load_tokenizer(model_name)
    return model, tokenizer
