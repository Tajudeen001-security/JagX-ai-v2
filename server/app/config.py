"""JagX v12 — central configuration, loaded from environment variables.

Design rule: **nothing here is required to start the app.** With an empty
environment JagX runs against a local Ollama on :11434 and needs no API key
from any company.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

APP_NAME = "JagX AI"
VERSION = "12.0.0"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class ProviderConfig:
    """One OpenAI-compatible chat-completions endpoint.

    Local runtimes (ollama / llamacpp / vllm) need no API key at all — that is
    the whole point of the local-first chain.
    """

    name: str
    base_url: str
    api_key: str
    model: str
    local: bool = False
    supports_tools: bool = True

    @property
    def configured(self) -> bool:
        if not self.base_url:
            return False
        if self.local:
            return True
        return bool(self.api_key)


@dataclass
class Settings:
    app_name: str = APP_NAME
    version: str = VERSION

    # --- storage -------------------------------------------------------
    data_dir: Path = field(default_factory=lambda: Path(_env("JAGX_DATA_DIR", "data")))

    # --- auth ----------------------------------------------------------
    jwt_secret: str = field(default_factory=lambda: _env("JAGX_JWT_SECRET", ""))
    jwt_algorithm: str = "HS256"
    jwt_ttl_hours: int = field(default_factory=lambda: _env_int("JAGX_JWT_TTL_HOURS", 24 * 14))
    allow_signup: bool = field(default_factory=lambda: _env_bool("JAGX_ALLOW_SIGNUP", True))

    # --- agent ---------------------------------------------------------
    max_steps: int = field(default_factory=lambda: _env_int("JAGX_MAX_STEPS", 12))
    max_tokens: int = field(default_factory=lambda: _env_int("JAGX_MAX_TOKENS", 2048))
    temperature: float = field(default_factory=lambda: _env_float("JAGX_TEMPERATURE", 0.4))
    request_timeout: float = field(default_factory=lambda: _env_float("JAGX_LLM_TIMEOUT", 120.0))
    tool_timeout: float = field(default_factory=lambda: _env_float("JAGX_TOOL_TIMEOUT", 30.0))
    max_history_messages: int = field(default_factory=lambda: _env_int("JAGX_MAX_HISTORY", 30))
    max_tool_output_chars: int = field(default_factory=lambda: _env_int("JAGX_MAX_TOOL_OUTPUT", 6000))
    critique_passes: int = field(default_factory=lambda: _env_int("JAGX_CRITIQUE_PASSES", 2))
    council_lanes: int = field(default_factory=lambda: _env_int("JAGX_COUNCIL_LANES", 3))

    # --- tools ---------------------------------------------------------
    tavily_api_key: str = field(default_factory=lambda: _env("TAVILY_API_KEY", ""))
    serper_api_key: str = field(default_factory=lambda: _env("SERPER_API_KEY", ""))
    enable_web_tools: bool = field(default_factory=lambda: _env_bool("JAGX_ENABLE_WEB_TOOLS", True))

    # --- sandbox -------------------------------------------------------
    enable_code_execution: bool = field(default_factory=lambda: _env_bool("JAGX_ENABLE_CODE_EXECUTION", True))
    sandbox_timeout: int = field(default_factory=lambda: _env_int("JAGX_SANDBOX_TIMEOUT", 10))
    sandbox_memory_mb: int = field(default_factory=lambda: _env_int("JAGX_SANDBOX_MEMORY_MB", 512))
    sandbox_allow_network: bool = field(default_factory=lambda: _env_bool("JAGX_SANDBOX_ALLOW_NETWORK", False))
    piston_url: str = field(default_factory=lambda: _env("JAGX_PISTON_URL", ""))

    # --- embeddings ----------------------------------------------------
    embedding_backend: str = field(default_factory=lambda: _env("JAGX_EMBEDDING_BACKEND", "auto"))
    embedding_model: str = field(default_factory=lambda: _env("JAGX_EMBEDDING_MODEL", "all-MiniLM-L6-v2"))

    # --- http ----------------------------------------------------------
    cors_origins: list = field(default_factory=list)
    port: int = field(default_factory=lambda: _env_int("PORT", 8000))

    def __post_init__(self) -> None:
        raw = _env("JAGX_CORS_ORIGINS", "")
        self.cors_origins = [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]
        if not self.cors_origins:
            self.cors_origins = [
                "http://localhost:5173",
                "http://127.0.0.1:5173",
                "http://localhost:8000",
                "http://127.0.0.1:8000",
            ]
        if not self.jwt_secret:
            # Dev fallback only. Production must set JAGX_JWT_SECRET.
            self.jwt_secret = "dev-insecure-secret-change-me"
            self.jwt_is_default = True
        else:
            self.jwt_is_default = False

    # --- providers -----------------------------------------------------
    def providers(self) -> list[ProviderConfig]:
        """Ordered provider chain. **Local runtimes come first** so that a
        machine with Ollama running never needs a third-party key."""
        chain = [
            ProviderConfig(
                name="ollama",
                base_url=_env("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                api_key=_env("OLLAMA_API_KEY", "ollama"),
                model=_env("OLLAMA_MODEL", "qwen2.5:7b"),
                local=True,
            ),
            ProviderConfig(
                name="llamacpp",
                base_url=_env("LLAMACPP_BASE_URL", "http://localhost:8080/v1"),
                api_key=_env("LLAMACPP_API_KEY", "local"),
                model=_env("LLAMACPP_MODEL", "local-gguf"),
                local=True,
            ),
            ProviderConfig(
                name="vllm",
                base_url=_env("VLLM_BASE_URL", "http://localhost:8000/v1"),
                api_key=_env("VLLM_API_KEY", "local"),
                model=_env("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
                local=True,
            ),
            # --- optional, pluggable, off unless a key is set -----------
            ProviderConfig(
                name="openai",
                base_url=_env("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                api_key=_env("OPENAI_API_KEY"),
                model=_env("OPENAI_MODEL", "gpt-4o-mini"),
            ),
            ProviderConfig(
                name="groq",
                base_url=_env("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
                api_key=_env("GROQ_API_KEY"),
                model=_env("GROQ_MODEL", "llama-3.3-70b-versatile"),
            ),
            ProviderConfig(
                name="openrouter",
                base_url=_env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                api_key=_env("OPENROUTER_API_KEY"),
                model=_env("OPENROUTER_MODEL", "qwen/qwen-2.5-72b-instruct"),
            ),
            ProviderConfig(
                name="deepseek",
                base_url=_env("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
                api_key=_env("DEEPSEEK_API_KEY"),
                model=_env("DEEPSEEK_MODEL", "deepseek-chat"),
            ),
        ]
        return [p for p in chain if p.configured]

    def provider_status(self) -> list[dict]:
        return [
            {"name": p.name, "model": p.model, "local": p.local, "configured": True}
            for p in self.providers()
        ]

    def local_providers(self) -> list[ProviderConfig]:
        return [p for p in self.providers() if p.local]


settings = Settings()
