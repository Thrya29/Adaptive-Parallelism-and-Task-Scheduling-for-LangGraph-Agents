"""Live smoke test against the real Anthropic API. Skipped unless ANTHROPIC_API_KEY is set.

Run explicitly with:  pytest -m live
"""

import os

import pytest

from adaptive_scheduler.agent import AdaptiveAgent
from adaptive_scheduler.config import LLMProvider, Settings

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"),
]


async def test_live_end_to_end() -> None:
    settings = Settings(llm_provider=LLMProvider.ANTHROPIC, selector_samples=16)
    agent = AdaptiveAgent.from_settings(settings)
    result = await agent.arun(
        "List three benefits of unit testing and three benefits of code review, "
        "then write a two-sentence summary."
    )
    assert result.answer.strip()
    assert result.metrics.model_calls >= 2
    assert result.metrics.input_tokens > 0
