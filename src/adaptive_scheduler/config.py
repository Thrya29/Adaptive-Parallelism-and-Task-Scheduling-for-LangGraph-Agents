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

    mock_seed: int = Field(default=0, description="Seed for the deterministic mock LLM.")
    max_concurrency: int = Field(default=4, ge=1, le=256)
    log_level: str = "INFO"

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
