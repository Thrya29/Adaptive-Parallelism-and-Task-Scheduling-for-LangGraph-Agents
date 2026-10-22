"""Deterministic mock chat model with realistic latency, token usage and provider limits.

It plugs into LangChain/LangGraph like any chat model, so the real executor runs unchanged with
or without an API key. Behaviour follows ``provider_profile`` (a :class:`ProviderProfile`), the
same model the simulator uses: sampled latency (with contention from concurrent calls), a
provider concurrency cap with a rate-limit penalty, token usage reported via ``usage_metadata``,
and optional transient failures.

Determinism: the random stream of each call is derived from ``(seed, prompt, call_index)`` where
``call_index`` counts previous calls with the *same* prompt, so results do not depend on the order
in which concurrent calls happen to be scheduled.
"""

from __future__ import annotations

import asyncio
import hashlib
import threading
import time
from collections import Counter
from typing import Any

import numpy as np
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, Field, PrivateAttr

from adaptive_scheduler.cost.profiles import estimate_tokens
from adaptive_scheduler.cost.provider import ProviderProfile


class MockLLMError(RuntimeError):
    """A simulated transient provider error (retryable)."""


class MockChatModel(BaseChatModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Named ``provider_profile`` because LangChain's BaseChatModel already defines ``profile``.
    provider_profile: ProviderProfile = Field(default_factory=ProviderProfile)
    seed: int = 0
    time_scale: float = Field(default=1.0, ge=0.0, description="0 = do not actually sleep")
    failure_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    default_output_tokens: int = Field(default=256, ge=1)

    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _active: int = PrivateAttr(default=0)
    _peak_active: int = PrivateAttr(default=0)
    _calls_per_prompt: Counter[str] = PrivateAttr(default_factory=Counter)
    _semaphores: dict[int, asyncio.Semaphore] = PrivateAttr(default_factory=dict)

    @property
    def _llm_type(self) -> str:
        return "adaptive-scheduler-mock"

    @property
    def peak_concurrency(self) -> int:
        """Highest number of calls served concurrently (for tests and diagnostics)."""
        return self._peak_active

    # ------------------------------------------------------------------ core sampling
    def _prompt_text(self, messages: list[BaseMessage]) -> str:
        return "\n".join(f"{m.type}: {m.text}" for m in messages)

    def _rng(self, prompt: str) -> np.random.Generator:
        with self._lock:
            index = self._calls_per_prompt[prompt]
            self._calls_per_prompt[prompt] += 1
        digest = hashlib.sha256(f"{self.seed}|{index}|{prompt}".encode()).digest()
        return np.random.default_rng(int.from_bytes(digest[:8], "little"))

    def _plan_call(
        self, messages: list[BaseMessage], kwargs: dict[str, Any], waited: bool
    ) -> tuple[float, AIMessage | None]:
        """Return ``(delay, message)``; ``message`` is ``None`` for a simulated failure."""
        prompt = self._prompt_text(messages)
        rng = self._rng(prompt)
        input_tokens = estimate_tokens(prompt)
        expected_out = int(kwargs.get("expected_output_tokens", self.default_output_tokens))
        sigma = self.provider_profile.output_tokens_sigma
        noise = float(rng.lognormal(-0.5 * sigma**2, sigma)) if sigma else 1.0
        output_tokens = max(1, round(expected_out * noise))
        latency = self.provider_profile.sample_latency(
            rng, input_tokens, output_tokens, self._active
        )
        if waited:
            latency += self.provider_profile.rate_limit_penalty_s
        if rng.random() < self.failure_rate:
            return latency * self.time_scale, None

        tag = hashlib.sha256(prompt.encode()).hexdigest()[:8]
        last = messages[-1].text if messages else ""
        content = _filler(f"[mock {tag}] Response to: {last[:120]}", output_tokens, rng)
        message = AIMessage(
            content=content,
            usage_metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            response_metadata={"model_name": "mock", "simulated_latency_s": latency},
        )
        return latency * self.time_scale, message

    def _enter(self) -> None:
        with self._lock:
            self._active += 1
            self._peak_active = max(self._peak_active, self._active)

    def _exit(self) -> None:
        with self._lock:
            self._active -= 1

    # ------------------------------------------------------------------ LangChain interface
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._enter()
        try:
            delay, message = self._plan_call(messages, kwargs, waited=False)
            if delay > 0:
                time.sleep(delay)
        finally:
            self._exit()
        return _result(message)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        loop_id = id(asyncio.get_running_loop())
        semaphore = self._semaphores.setdefault(
            loop_id, asyncio.Semaphore(self.provider_profile.max_concurrency)
        )
        waited = semaphore.locked()
        async with semaphore:
            self._enter()
            try:
                delay, message = self._plan_call(messages, kwargs, waited=waited)
                if delay > 0:
                    await asyncio.sleep(delay)
            finally:
                self._exit()
        return _result(message)


_WORDS = (
    "analysis", "result", "data", "evidence", "estimate", "trend", "growth", "market", "risk",
    "factor", "summary", "detail", "source", "figure", "comparison", "outcome", "finding",
    "context", "assumption", "scenario", "metric",
)  # fmt: skip


def _filler(header: str, output_tokens: int, rng: np.random.Generator) -> str:
    """Text of roughly ``output_tokens`` tokens, so downstream prompts grow realistically."""
    target_chars = 4 * output_tokens
    words = [header]
    length = len(header)
    while length < target_chars:
        word = _WORDS[int(rng.integers(len(_WORDS)))]
        words.append(word)
        length += len(word) + 1
    return " ".join(words)[: max(target_chars, len(header))]


def _result(message: AIMessage | None) -> ChatResult:
    # Failures surface only after the simulated latency elapsed, like a real provider error.
    if message is None:
        raise MockLLMError("simulated transient provider error")
    return ChatResult(generations=[ChatGeneration(message=message)])
