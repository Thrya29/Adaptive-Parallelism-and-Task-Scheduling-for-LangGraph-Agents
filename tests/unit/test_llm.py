import asyncio

import pytest
from langchain_core.messages import HumanMessage

from adaptive_scheduler.config import Settings
from adaptive_scheduler.cost import ProviderProfile
from adaptive_scheduler.llm import MockChatModel, MockLLMError, create_chat_model


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("ANTHROPIC_API_KEY", "APSCHED_ANTHROPIC_API_KEY", "APSCHED_LLM_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setitem(Settings.model_config, "env_file", None)


def test_mock_reports_usage_and_is_deterministic() -> None:
    a = MockChatModel(seed=1, time_scale=0)
    b = MockChatModel(seed=1, time_scale=0)
    msg = [HumanMessage("Summarize the report")]
    ra, rb = a.invoke(msg), b.invoke(msg)
    assert ra.content == rb.content
    assert ra.usage_metadata == rb.usage_metadata
    assert ra.usage_metadata is not None
    assert ra.usage_metadata["output_tokens"] > 0
    assert ra.usage_metadata["total_tokens"] == (
        ra.usage_metadata["input_tokens"] + ra.usage_metadata["output_tokens"]
    )
    # Repeating a prompt draws a new (but reproducible) sample.
    assert a.invoke(msg).response_metadata == b.invoke(msg).response_metadata


def test_mock_honours_expected_output_tokens() -> None:
    m = MockChatModel(time_scale=0, provider_profile=ProviderProfile(output_tokens_sigma=0))
    r = m.invoke([HumanMessage("x")], expected_output_tokens=777)
    assert r.usage_metadata is not None
    assert r.usage_metadata["output_tokens"] == 777


def test_mock_failure_injection() -> None:
    m = MockChatModel(time_scale=0, failure_rate=1.0)
    with pytest.raises(MockLLMError):
        m.invoke([HumanMessage("x")])


async def test_mock_enforces_provider_concurrency_cap() -> None:
    m = MockChatModel(time_scale=0.01, provider_profile=ProviderProfile(max_concurrency=3))
    results = await asyncio.gather(*(m.ainvoke([HumanMessage(f"task {i}")]) for i in range(10)))
    assert len(results) == 10
    assert m.peak_concurrency == 3
    penalized = [r for r in results if r.response_metadata["simulated_latency_s"] >= 1.0]
    assert penalized  # calls that queued paid the rate-limit penalty


def test_factory_without_key_returns_mock() -> None:
    assert isinstance(create_chat_model(Settings()), MockChatModel)


def test_factory_with_key_returns_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    from langchain_anthropic import ChatAnthropic  # noqa: PLC0415

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    model = create_chat_model(Settings())
    assert isinstance(model, ChatAnthropic)
    assert model.model == Settings().anthropic_model
