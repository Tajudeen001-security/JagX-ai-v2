"""Central configuration from environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Settings:
    version: str = "8.0.0"
    app_name: str = "JagX AI"

    # Providers
    groq_api_key: str = field(default_factory=lambda: os.environ.get("GROQ_API_KEY", ""))
    openrouter_api_key: str = field(default_factory=lambda: os.environ.get("OPENROUTER_API_KEY", ""))
    hf_token: str = field(default_factory=lambda: os.environ.get("HF_TOKEN", ""))
    nvidia_api_key: str = field(
        default_factory=lambda: os.environ.get("NVIDIA_API_KEY")
        or os.environ.get("NVIDIA_NIM_API_KEY", "")
    )

    groq_model: str = field(
        default_factory=lambda: os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    )
    openrouter_model: str = field(
        default_factory=lambda: os.environ.get(
            "OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct"
        )
    )
    hf_model: str = field(
        default_factory=lambda: os.environ.get("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    )

    # Auth
    admin_secret: str = field(
        default_factory=lambda: os.environ.get("JAGX_ADMIN_SECRET", "change-this-admin-secret")
    )
    keys_file: str = field(default_factory=lambda: os.environ.get("JAGX_KEYS_FILE", "keys.json"))
    permanent_keys: set = field(default_factory=set)

    # Limits
    tier_hourly: dict = field(
        default_factory=lambda: {
            "free": 80,
            "premium": 350,
            "premium_plus": 900,
            "master": None,
            "admin": None,
        }
    )
    global_ip_rpm: int = field(
        default_factory=lambda: int(os.environ.get("JAGX_GLOBAL_IP_RPM", "120"))
    )
    max_body_bytes: int = 15 * 1024 * 1024
    max_code_len: int = 12000
    max_history: int = 12
    default_max_tokens: int = 1200
    temperature: float = 0.35

    # Paths
    memory_file: str = field(
        default_factory=lambda: os.environ.get("JAGX_MEMORY_FILE", "jagx_memory.json")
    )
    training_file: str = field(
        default_factory=lambda: os.environ.get("JAGX_TRAINING_FILE", "jagx_training_data.jsonl")
    )
    training_enabled: bool = field(
        default_factory=lambda: os.environ.get("JAGX_TRAINING_DATA_ENABLED", "true").lower()
        == "true"
    )

    # Tools
    piston_url: str = "https://emkc.org/api/v2/piston"
    port: int = field(default_factory=lambda: int(os.environ.get("PORT", "10000")))

    def __post_init__(self) -> None:
        raw = os.environ.get("JAGX_PERMANENT_KEYS", "")
        self.permanent_keys = {k.strip() for k in raw.split(",") if k.strip()}


settings = Settings()
