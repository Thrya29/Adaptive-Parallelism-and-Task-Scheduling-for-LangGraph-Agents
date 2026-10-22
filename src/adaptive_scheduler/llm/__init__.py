"""LLM backends: deterministic mock and Anthropic, behind LangChain's chat-model interface."""

from adaptive_scheduler.llm.factory import create_chat_model
from adaptive_scheduler.llm.mock import MockChatModel, MockLLMError

__all__ = ["MockChatModel", "MockLLMError", "create_chat_model"]
