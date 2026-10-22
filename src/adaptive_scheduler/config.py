"""Runtime configuration.

All settings come from environment variables (prefix ``APSCHED_``) or a local ``.env`` file.
The only setting most users touch is the LLM backend:

* ``APSCHED_LLM_PROVIDER=auto`` (default): use Anthropic if ``ANTHROPIC_API_KEY`` is set,
  otherwise fall back to the deterministic mock LLM. Works out of the box with no key.
* ``APSCHED_LLM_PROVIDER=mock``: always use the mock LLM (reproducible, free).
* ``APSCHED_LLM_PROVIDER=anthropic``: require a real key and fail fast if it is missing.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(StrEnum):
    """Requested LLM backend. ``AUTO`` is resolved to a concrete backend at startup."""

    AUTO = "auto"
    MOCK = "mock"
    ANTHROPIC = "anthropic"


class ConfigurationError(ValueError):
    """Raised when the configuration is inconsistent (e.g. Anthropic requested without a key)."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="APSCHED_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: LLMProvider = LLMProvider.AUTO
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "APSCHED_ANTHROPIC_API_KEY"),
    )
    anthropic_model: str = "claude-haiku-5-5"
    anthropic_max_tokens: int = Field(default=16_000, ge=256, le=128_000)

    mock_seed: int = Field(default=0, description="Seed for the deterministic mock LLM.")
    mock_time_scale: float = Field(
        default=0.1, ge=0.0, description="Wall-clock seconds per simulated second for the mock."
    )
    max_concurrency: int = Field(
        default=8, ge=1, le=256, description="Provider concurrency limit (calls in flight)."
    )
    max_plan_tasks: int = Field(default=32, ge=1, le=256)
    max_retries: int = Field(default=2, ge=0, le=10)
    task_timeout_s: float = Field(default=120.0, gt=0)
    selector_samples: int = Field(default=64, ge=1, le=10_000)
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "json"

    # HTTP service
    api_key: SecretStr | None = Field(
        default=None, description="If set, requests must send it in the X-API-Key header."
    )
    max_concurrent_runs: int = Field(default=8, ge=1, le=1024)
    max_request_chars: int = Field(default=8_000, ge=1)
    run_timeout_s: float = Field(default=300.0, gt=0)
    service_name: str = "adaptive-scheduler"
    otel_console: bool = Field(default=False, description="Print spans to stdout (debugging).")

    @property
    def has_anthropic_key(self) -> bool:
        key = self.anthropic_api_key
        return key is not None and bool(key.get_secret_value().strip())

    def resolve_provider(self) -> LLMProvider:
        """Return the concrete backend to use (never ``AUTO``)."""
        if self.llm_provider is LLMProvider.MOCK:
            return LLMProvider.MOCK
        if self.llm_provider is LLMProvider.ANTHROPIC:
            if not self.has_anthropic_key:
                raise ConfigurationError(
                    "APSCHED_LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set. "
                    "Set the key in your environment or .env, or use provider 'auto'/'mock'."
                )
            return LLMProvider.ANTHROPIC
        return LLMProvider.ANTHROPIC if self.has_anthropic_key else LLMProvider.MOCK


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
