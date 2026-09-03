from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

@dataclass
class ModelConfig:
    vocab_size: int = 32768
    max_seq_len: int = 2048
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int | None = None
    d_ff: int | None = None
    dropout: float = 0.0
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    use_swiglu: bool = True
    use_rms_norm: bool = True
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        if self.n_kv_heads is None: self.n_kv_heads = self.n_heads
        if self.d_ff is None:
            self.d_ff = int(8 * self.d_model / 3) if self.use_swiglu else 4 * self.d_model
            self.d_ff = ((self.d_ff + 63) // 64) * 64 if self.use_swiglu else self.d_ff

    def validate(self) -> "ModelConfig":
        if self.d_model % self.n_heads: raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads: raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.vocab_size < 2 or self.max_seq_len < 2: raise ValueError("vocab_size and max_seq_len must be >= 2")
        if self.n_layers < 1: raise ValueError("n_layers must be >= 1")
        if self.d_ff < self.d_model: raise ValueError("d_ff should be >= d_model")
        return self

    def to_dict(self) -> dict[str, Any]: return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})
