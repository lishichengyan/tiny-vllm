"""Model configuration for TinyLlama.

This module is provided infrastructure (not a learner exercise). It only
carries the handful of numbers the rest of the code needs. HuggingFace's
``LlamaConfig`` has many more fields; we ignore everything we do not use.
"""

from dataclasses import dataclass


@dataclass
class TinyLlamaConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False
    bos_token_id: int | None = None
    eos_token_id: int | None = None

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size={self.hidden_size} must be divisible by num_heads={self.num_heads}"
            )
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads={self.num_heads} must be divisible by num_kv_heads={self.num_kv_heads}"
            )
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def num_queries_per_kv(self) -> int:
        """How many query heads share one K/V head (1 means plain multi-head attention)."""
        return self.num_heads // self.num_kv_heads

    @classmethod
    def from_hf(cls, hf_config) -> "TinyLlamaConfig":
        """Build a TinyLlamaConfig from a HuggingFace ``LlamaConfig``."""
        if getattr(hf_config, "model_type", "llama") != "llama":
            raise ValueError(
                f"tiny-vllm only supports Llama-like models, got model_type={hf_config.model_type!r}"
            )
        hidden_size = hf_config.hidden_size
        num_heads = hf_config.num_attention_heads
        hf_head_dim = getattr(hf_config, "head_dim", None)
        if hf_head_dim is not None and hf_head_dim != hidden_size // num_heads:
            raise ValueError(
                f"unsupported head_dim={hf_head_dim}; expected hidden_size // num_heads = {hidden_size // num_heads}"
            )
        if getattr(hf_config, "attention_bias", False) or getattr(hf_config, "mlp_bias", False):
            raise ValueError("tiny-vllm does not support attention/MLP biases")
        if getattr(hf_config, "rope_scaling", None) not in (None, {}):
            raise ValueError("tiny-vllm does not support rope_scaling")
        return cls(
            vocab_size=hf_config.vocab_size,
            hidden_size=hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_layers=hf_config.num_hidden_layers,
            num_heads=num_heads,
            num_kv_heads=getattr(hf_config, "num_key_value_heads", None) or num_heads,
            max_position_embeddings=hf_config.max_position_embeddings,
            rms_norm_eps=hf_config.rms_norm_eps,
            rope_theta=getattr(hf_config, "rope_theta", 10000.0),
            tie_word_embeddings=bool(getattr(hf_config, "tie_word_embeddings", False)),
            bos_token_id=getattr(hf_config, "bos_token_id", None),
            eos_token_id=getattr(hf_config, "eos_token_id", None),
        )
