"""Central configuration from environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Settings:
    version: str = "8.1.0"
    app_name: str = "JagX AI"

    offline_mode: bool = field(default_factory=lambda: os.environ.get("JAGX_OFFLINE_MODE", "false").lower() == "true")

    # Providers — prefer the strongest free/fast models by default
    groq_api_key: str = field(default_factory=lambda: os.environ.get("GROQ_API_KEY", ""))
    openrouter_api_key: str = field(default_factory=lambda: os.environ.get("OPENROUTER_API_KEY", ""))
    hf_token: str = field(default_factory=lambda: os.environ.get("HF_TOKEN", ""))
    pollinations_api_key: str = field(default_factory=lambda: os.environ.get("POLLINATIONS_API_KEY", ""))
    nvidia_api_key: str = field(
        default_factory=lambda: os.environ.get("NVIDIA_API_KEY")
        or os.environ.get("NVIDIA_NIM_API_KEY", "")
    )

    # Strong defaults: 70B-class where available
    groq_model: str = field(
        default_factory=lambda: os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    )
    openrouter_model: str = field(
        default_factory=lambda: os.environ.get(
            "OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct"
        )
    )
    nvidia_model: str = field(default_factory=lambda: os.environ.get("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct"))
    hf_model: str = field(
        default_factory=lambda: os.environ.get("HF_MODEL", "Qwen/Qwen2.5-72B-Instruct")
    )

    # Auth
    admin_secret: str = field(
        default_factory=lambda: os.environ.get("JAGX_ADMIN_SECRET", "")
    )
    keys_file: str = field(default_factory=lambda: os.environ.get("JAGX_KEYS_FILE", "keys.json"))
    permanent_keys: set = field(default_factory=set)

    # Limits — generous for bot quality
    tier_hourly: dict = field(
        default_factory=lambda: {
            "free": 100,
            "premium": 400,
            "premium_plus": 1000,
            "master": None,
            "admin": None,
        }
    )
    global_ip_rpm: int = field(
        default_factory=lambda: int(os.environ.get("JAGX_GLOBAL_IP_RPM", "150"))
    )
    public_chat_rpm: int = field(
        default_factory=lambda: max(1, int(os.environ.get("JAGX_PUBLIC_CHAT_RPM", "20")))
    )
    max_body_bytes: int = 15 * 1024 * 1024
    max_code_len: int = 16000
    max_history: int = 20
    llm_timeout_seconds: float = field(default_factory=lambda: max(8.0, float(os.environ.get("JAGX_LLM_TIMEOUT", "35"))))
    default_max_tokens: int = 2000
    temperature: float = 0.5

    # Paths
    memory_file: str = field(
        default_factory=lambda: os.environ.get("JAGX_MEMORY_FILE", "jagx_memory.json")
    )
    training_file: str = field(
        default_factory=lambda: os.environ.get("JAGX_TRAINING_FILE", "jagx_training_data.jsonl")
    )
    training_enabled: bool = field(
        default_factory=lambda: os.environ.get("JAGX_TRAINING_DATA_ENABLED", "false").lower()
        == "true"
    )

    conversation_db: str = field(default_factory=lambda: os.environ.get("JAGX_CONVERSATION_DB", "jagx_conversations.sqlite3"))
    jobs_db: str = field(default_factory=lambda: os.environ.get("JAGX_JOBS_DB", "jagx_jobs.sqlite3"))
    job_poll_seconds: float = field(default_factory=lambda: max(0.5, float(os.environ.get("JAGX_JOB_POLL_SECONDS", "2"))))
    data_encryption_key: str = field(default_factory=lambda: os.environ.get("JAGX_DATA_ENCRYPTION_KEY", ""))

    # Tools
    piston_url: str = "https://emkc.org/api/v2/piston"
    port: int = field(default_factory=lambda: int(os.environ.get("PORT", "10000")))

    def __post_init__(self) -> None:
        raw = os.environ.get("JAGX_PERMANENT_KEYS", "")
        self.permanent_keys = {k.strip() for k in raw.split(",") if k.strip()}


settings = Settings()
