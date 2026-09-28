"""TinyLlama: our own minimal Llama-like decoder (Milestone 1, checkpoints 1.4 and 1.6).

    input_ids [batch, seq_len]
        │  embed_tokens
        ▼
    hidden_states [batch, seq_len, hidden_size]
        │
        │  ┌───────────────────────────────────────────┐
        │  │ DecoderLayer × num_layers                 │
        ├─►│   h = h + self_attn(input_layernorm(h))   │
        │  │   h = h + mlp(post_attention_layernorm(h))│
        │  └───────────────────────────────────────────┘
        ▼
    norm  ->  lm_head
        ▼
    logits [batch, seq_len, vocab_size]

The module names mirror HuggingFace's ``LlamaForCausalLM`` (``embed_tokens``,
``layers.{i}.self_attn.q_proj``, ``norm``, ``lm_head`` ...) so that weight loading
is a matter of mapping names, not reshaping tensors.
"""

import torch
import torch.nn as nn

from .attention import Attention
from .config import TinyLlamaConfig
from .kv_cache import AttentionMetadata, ContiguousKVCache, PagedKVCache
from .layers import (  # noqa: F401  (compute_rope_cos_sin is used by your forward)
    MLP,
    RMSNorm,
    compute_rope_cos_sin,
)


class DecoderLayer(nn.Module):
    def __init__(self, config: TinyLlamaConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Attention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: ContiguousKVCache | PagedKVCache | None = None,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        """Pre-norm transformer block with two residual connections (checkpoint 1.4).

        hidden_states: [batch, seq_len, hidden_size] -> [batch, seq_len, hidden_size]

        Pass ``positions``, ``cos``, ``sin``, ``kv_cache`` and ``attn_metadata`` straight
        through to ``self.self_attn``; this layer does not interpret them.
        """
        # TODO(student): Milestone 1
        raise NotImplementedError("Milestone 1 (1.4): implement DecoderLayer.forward()")


class TinyLlama(nn.Module):
    def __init__(self, config: TinyLlamaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(config, layer_idx) for layer_idx in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            # Share one tensor: loading embed_tokens also loads lm_head.
            self.lm_head.weight = self.embed_tokens.weight

    @property
    def device(self) -> torch.device:
        return self.embed_tokens.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.embed_tokens.weight.dtype

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: ContiguousKVCache | PagedKVCache | None = None,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        """Full model forward (checkpoint 1.6).

        Args:
            input_ids:     [batch, seq_len] token ids.
            positions:     [batch, seq_len] absolute position of every token. During prefill
                           this is ``0 .. seq_len-1``; during decode it is the single position of
                           the new token.
            kv_cache:      None (M1), ContiguousKVCache (M2) or PagedKVCache (M3+).
            attn_metadata: required with a PagedKVCache, otherwise None.

        Returns:
            logits: [batch, seq_len, vocab_size] in float32.

        Steps: embed -> compute RoPE cos/sin once from ``positions`` (``compute_rope_cos_sin``)
        -> every decoder layer -> final norm -> lm_head.
        """
        # TODO(student): Milestone 1
        raise NotImplementedError("Milestone 1 (1.6): implement TinyLlama.forward()")
