"""Create the chat model selected by configuration (mock or Anthropic)."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from adaptive_scheduler.config import LLMProvider, Settings
from adaptive_scheduler.llm.mock import MockChatModel


def create_chat_model(settings: Settings) -> BaseChatModel:
    provider = settings.resolve_provider()
    if provider is LLMProvider.ANTHROPIC:
        try:
            from langchain_anthropic import ChatAnthropic  # noqa: PLC0415 - optional dependency
        except ImportError as exc:  # pragma: no cover - depends on installed extras
            raise RuntimeError(
                "Anthropic backend requested but 'langchain-anthropic' is not installed. "
                "Install with: pip install -e '.[anthropic]'"
            ) from exc
        if settings.anthropic_api_key is None:  # pragma: no cover - resolve_provider() ensures it
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        return ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            max_retries=0,  # retries are owned by the executor so they are counted and scheduled
        )
    return MockChatModel(seed=settings.mock_seed)
