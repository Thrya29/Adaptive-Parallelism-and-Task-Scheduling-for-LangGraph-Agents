"""Online calibration of the cost model from observed LLM calls.

* Output tokens and success rate per task type: exponentially weighted moving averages seeded
  with the priors, so early estimates are sane and later ones track drift.
* Latency: decayed *weighted* least-squares fit of ``latency = a + b * output_tokens``, the
  dominant linear structure of LLM latency (``a`` ~ time-to-first-token, ``b`` ~ 1 / decode
  throughput). LLM latency noise is multiplicative (its standard deviation grows with the mean),
  so each observation is weighted by ``1 / expected_latency**2`` under the prior; ordinary least
  squares would let long, noisy generations dominate the intercept. Five decayed sufficient
  statistics make each update ``O(1)``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from adaptive_scheduler.cost.profiles import OUTPUT_TOKEN_PRIORS, SUCCESS_PRIORS
from adaptive_scheduler.cost.provider import ProviderProfile
from adaptive_scheduler.graph.model import TaskType


@dataclass
class _LinearFit:
    decay: float
    n: float = 0.0
    sx: float = 0.0
    sy: float = 0.0
    sxx: float = 0.0
    sxy: float = 0.0

    count: float = 0.0

    def update(self, x: float, y: float, w: float = 1.0) -> None:
        d = self.decay
        self.count = d * self.count + 1
        self.n = d * self.n + w
        self.sx = d * self.sx + w * x
        self.sy = d * self.sy + w * y
        self.sxx = d * self.sxx + w * x * x
        self.sxy = d * self.sxy + w * x * y

    def solve(self) -> tuple[float, float] | None:
        """Return ``(intercept, slope)``, or ``None`` until the fit is well-conditioned."""
        denom = self.n * self.sxx - self.sx * self.sx
        if self.count < 3 or denom <= 1e-9 * max(1e-12, self.n * self.sxx):
            return None
        slope = (self.n * self.sxy - self.sx * self.sy) / denom
        intercept = (self.sy - slope * self.sx) / self.n
        return intercept, slope


@dataclass
class CostEstimator:
    prior: ProviderProfile = field(default_factory=ProviderProfile)
    alpha: float = 0.1  # EWMA weight of a new observation
    decay: float = 0.995  # forgetting factor of the latency fit
    output_tokens: dict[TaskType, float] = field(default_factory=lambda: dict(OUTPUT_TOKEN_PRIORS))
    success: dict[TaskType, float] = field(default_factory=lambda: dict(SUCCESS_PRIORS))
    observations: int = 0
    _fit: _LinearFit = field(init=False)

    def __post_init__(self) -> None:
        if not 0 < self.alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if not 0 < self.decay <= 1:
            raise ValueError("decay must be in (0, 1]")
        self._fit = _LinearFit(self.decay)

    def observe(
        self,
        task_type: TaskType,
        *,
        latency_s: float,
        input_tokens: float,
        output_tokens: float,
        succeeded: bool,
        update_type_stats: bool = True,
    ) -> None:
        if update_type_stats:
            a = self.alpha
            self.output_tokens[task_type] = (1 - a) * self.output_tokens[
                task_type
            ] + a * output_tokens
            # Kept away from 0/1 so a single observation never makes the estimate degenerate.
            target = 0.999 if succeeded else 0.001
            self.success[task_type] = (1 - a) * self.success[task_type] + a * target
        # E[latency] = tail_factor * (ttft + n_in / prefill + n_out / decode). Divide out the tail
        # factor and the (known) prefill time so the fit isolates ttft and 1 / decode speed.
        y = latency_s / self._tail_factor() - input_tokens / self.prior.prefill_tps
        weight = 1.0 / self.prior.base_latency(input_tokens, output_tokens) ** 2
        self._fit.update(output_tokens, y, weight)
        self.observations += 1

    def _tail_factor(self) -> float:
        p = self.prior
        return 1 - p.tail_prob + p.tail_prob * p.tail_multiplier

    def provider(self) -> ProviderProfile:
        """Current provider beliefs: the prior with fitted ttft and decode speed, if available."""
        solved = self._fit.solve()
        if solved is None:
            return self.prior
        intercept, slope = solved
        if intercept < 0 or slope <= 0:
            return self.prior  # fit not physically meaningful yet
        return replace(self.prior, ttft_s=intercept, decode_tps=1 / slope)

    def output_token_estimates(self) -> Mapping[TaskType, float]:
        return dict(self.output_tokens)

    def success_estimates(self) -> Mapping[TaskType, float]:
        return dict(self.success)
