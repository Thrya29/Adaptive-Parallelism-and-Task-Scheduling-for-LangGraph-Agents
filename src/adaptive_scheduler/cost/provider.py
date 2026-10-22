"""LLM provider performance and pricing model.

Latency of one call that reads ``n_in`` tokens and writes ``n_out`` tokens while ``a`` calls are
active at the provider::

    base    = ttft + n_in / prefill_tps + n_out / decode_tps
    latency = base * noise * tail * (1 + contention * (a - 1))

where ``noise`` is log-normal with mean 1, and ``tail`` equals ``tail_multiplier`` with probability
``tail_prob`` (otherwise 1) to reproduce the heavy right tail seen in production LLM APIs.

The provider serves at most ``max_concurrency`` calls at once; excess calls wait (modelling
HTTP 429 + client back-off) and pay ``rate_limit_penalty`` seconds of wasted back-off on top.

Default values are illustrative assumptions for experiments, not measurements of a specific model;
they are calibrated online by :class:`~adaptive_scheduler.cost.estimator.CostEstimator`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# USD per million (input, output) tokens, Anthropic first-party API, as of 2026-10.
# Haiku 5.5 prices apply to prompts up to 100K tokens. Unknown models fall back to the profile
# defaults, which are illustrative values for simulation.
MODEL_PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}


@dataclass(frozen=True, slots=True)
class ProviderProfile:
    ttft_s: float = 0.6
    prefill_tps: float = 5_000.0
    decode_tps: float = 80.0
    noise_sigma: float = 0.35
    output_tokens_sigma: float = 0.3  # log-normal spread of actual vs expected output length
    tail_prob: float = 0.05
    tail_multiplier: float = 3.0
    contention: float = 0.03
    max_concurrency: int = 8
    rate_limit_penalty_s: float = 1.0
    usd_per_mtok_in: float = 1.0
    usd_per_mtok_out: float = 5.0
    fusion_quality_decay: float = 0.98  # success multiplier per extra task packed into one call

    def __post_init__(self) -> None:
        positive = {
            "prefill_tps": self.prefill_tps,
            "decode_tps": self.decode_tps,
            "tail_multiplier": self.tail_multiplier,
        }
        for name, value in positive.items():
            if not value > 0:
                raise ValueError(f"{name} must be > 0")
        non_negative = {
            "ttft_s": self.ttft_s,
            "noise_sigma": self.noise_sigma,
            "output_tokens_sigma": self.output_tokens_sigma,
            "contention": self.contention,
            "rate_limit_penalty_s": self.rate_limit_penalty_s,
            "usd_per_mtok_in": self.usd_per_mtok_in,
            "usd_per_mtok_out": self.usd_per_mtok_out,
        }
        for name, value in non_negative.items():
            if not (math.isfinite(value) and value >= 0):
                raise ValueError(f"{name} must be finite and >= 0")
        if not 0 <= self.tail_prob <= 1:
            raise ValueError("tail_prob must be in [0, 1]")
        if not 0 < self.fusion_quality_decay <= 1:
            raise ValueError("fusion_quality_decay must be in (0, 1]")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")

    def base_latency(self, input_tokens: float, output_tokens: float) -> float:
        return self.ttft_s + input_tokens / self.prefill_tps + output_tokens / self.decode_tps

    def expected_latency(self, input_tokens: float, output_tokens: float) -> float:
        """Expected latency of an uncontended call (noise has mean 1)."""
        tail = 1 - self.tail_prob + self.tail_prob * self.tail_multiplier
        return self.base_latency(input_tokens, output_tokens) * tail

    def sample_latency(
        self,
        rng: np.random.Generator,
        input_tokens: float,
        output_tokens: float,
        active_calls: int = 1,
    ) -> float:
        sigma = self.noise_sigma
        noise = float(rng.lognormal(-0.5 * sigma * sigma, sigma)) if sigma > 0 else 1.0
        tail = self.tail_multiplier if rng.random() < self.tail_prob else 1.0
        slowdown = 1 + self.contention * max(active_calls - 1, 0)
        return self.base_latency(input_tokens, output_tokens) * noise * tail * slowdown

    @classmethod
    def for_model(cls, model: str, **overrides: float) -> ProviderProfile:
        """Profile with the model's published prices (other fields default / overridden)."""
        prices = MODEL_PRICES_USD_PER_MTOK.get(model)
        fields: dict[str, float] = dict(overrides)
        if prices is not None:
            fields.setdefault("usd_per_mtok_in", prices[0])
            fields.setdefault("usd_per_mtok_out", prices[1])
        return cls(**fields)  # type: ignore[arg-type]

    def cost_usd(self, input_tokens: float, output_tokens: float) -> float:
        return (input_tokens * self.usd_per_mtok_in + output_tokens * self.usd_per_mtok_out) / 1e6
