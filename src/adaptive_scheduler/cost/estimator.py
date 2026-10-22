"""Online calibration of the cost model from observed LLM calls.

Real LLM latency is close to deterministic *given* the number of output tokens (in the live
validation, corr(latency, output tokens) = 0.93 and run-level predictions made with the actual
token counts correlate 0.98 with measured makespans). Run-to-run variability therefore comes
mostly from *how long the output is*, not from noise on top of it. The estimator models the two
separately:

* **Latency given tokens** - decayed weighted least squares of
  ``latency - n_in / prefill = ttft + n_out / decode`` (weights ``1 / expected_latency**2``,
  because noise is multiplicative), plus the spread of the multiplicative residual, which becomes
  the provider's ``noise_sigma``. Once fitted, the provider's assumed heavy tail is shrunk away:
  any genuine tail shows up in the residual spread and the output-length spread instead.
* **Output length per task type** - decayed mean and log-spread with a prior pseudo-count, so a
  few long outputs cannot swing the estimate and the simulator can sample realistic lengths.
* **Success rate per task type** - exponentially weighted moving average seeded with the prior.

All fitted quantities are shrunk towards the prior with weight ``n / (n + prior_strength)``, where
``n`` is the *total* number of observations: noisy early fits cannot overcorrect, and a poor prior
fades completely as evidence accumulates. (An earlier version used the decayed count, which caps
at ``1 / (1 - decay)``, so a bad prior kept ~13% weight forever - the live run exposed it.)
"""

from __future__ import annotations

import math
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
class _Moments:
    """Decayed count, mean and log-moments of a positive quantity."""

    decay: float
    n: float = 0.0
    sum_x: float = 0.0
    sum_log: float = 0.0
    sum_log2: float = 0.0

    def update(self, x: float) -> None:
        d, lx = self.decay, math.log(max(x, 1e-9))
        self.n = d * self.n + 1
        self.sum_x = d * self.sum_x + x
        self.sum_log = d * self.sum_log + lx
        self.sum_log2 = d * self.sum_log2 + lx * lx

    def log_var(self) -> float | None:
        if self.n < 3:
            return None
        mean = self.sum_log / self.n
        return max(self.sum_log2 / self.n - mean * mean, 0.0)


@dataclass
class CostEstimator:
    prior: ProviderProfile = field(default_factory=ProviderProfile)
    alpha: float = 0.1  # EWMA weight of a new observation (success rates)
    decay: float = 0.995  # forgetting factor of the latency fit
    token_decay: float = 0.98  # forgetting factor of per-type output length (~50-call window)
    prior_strength: float = 10.0  # pseudo-observations backing the prior latency parameters
    token_prior_strength: float = 3.0  # pseudo-observations backing each output-length prior
    output_tokens: dict[TaskType, float] = field(default_factory=lambda: dict(OUTPUT_TOKEN_PRIORS))
    success: dict[TaskType, float] = field(default_factory=lambda: dict(SUCCESS_PRIORS))
    observations: int = 0
    _fit: _LinearFit = field(init=False)
    _residual: _Moments = field(init=False)
    _tokens: dict[TaskType, _Moments] = field(init=False)
    _token_priors: dict[TaskType, float] = field(init=False)

    def __post_init__(self) -> None:
        if not 0 < self.alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if not 0 < self.decay <= 1 or not 0 < self.token_decay <= 1:
            raise ValueError("decay factors must be in (0, 1]")
        if self.prior_strength < 0 or self.token_prior_strength < 0:
            raise ValueError("prior strengths must be >= 0")
        self._fit = _LinearFit(self.decay)
        self._residual = _Moments(self.decay)
        self._tokens = {t: _Moments(self.token_decay) for t in TaskType}
        self._token_priors = dict(self.output_tokens)

    # ------------------------------------------------------------------ updates
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
            self._tokens[task_type].update(output_tokens)
            self.output_tokens[task_type] = self._token_mean(task_type)
            # Kept away from 0/1 so a single observation never makes the estimate degenerate.
            target = 0.999 if succeeded else 0.001
            a = self.alpha
            self.success[task_type] = (1 - a) * self.success[task_type] + a * target

        prefill = input_tokens / self.prior.prefill_tps
        # Residual against the *current* mean model, before this point updates it.
        predicted = self._mean_latency(input_tokens, output_tokens)
        if predicted > 0 and latency_s > 0:
            self._residual.update(latency_s / predicted)
        weight = 1.0 / self.prior.base_latency(input_tokens, output_tokens) ** 2
        self._fit.update(output_tokens, latency_s - prefill, weight)
        self.observations += 1

    # ------------------------------------------------------------------ estimates
    def _weight(self) -> float:
        n = self.observations
        return n / (n + self.prior_strength) if n + self.prior_strength > 0 else 1.0

    def _latency_params(self) -> tuple[float, float] | None:
        """Shrunk ``(ttft_s, seconds_per_output_token)`` of the *mean* latency, if fitted."""
        solved = self._fit.solve()
        if solved is None:
            return None
        intercept, slope = solved
        if intercept < 0 or slope <= 0:
            return None  # not physically meaningful yet
        p = self.prior
        tail = 1 - p.tail_prob + p.tail_prob * p.tail_multiplier
        w = self._weight()
        # The prior's *mean* latency includes its assumed tail factor.
        ttft = w * intercept + (1 - w) * p.ttft_s * tail
        sec_per_token = w * slope + (1 - w) * tail / p.decode_tps
        return ttft, sec_per_token

    def _mean_latency(self, input_tokens: float, output_tokens: float) -> float:
        params = self._latency_params()
        if params is None:
            return self.prior.expected_latency(input_tokens, output_tokens)
        ttft, spt = params
        return ttft + input_tokens / self.prior.prefill_tps + output_tokens * spt

    def _token_mean(self, task_type: TaskType) -> float:
        m, k = self._tokens[task_type], self.token_prior_strength
        return (k * self._token_priors[task_type] + m.sum_x) / (k + m.n)

    def _token_sigma(self) -> float:
        """Pooled log-spread of output lengths across task types, shrunk to the prior."""
        num = den = 0.0
        for m in self._tokens.values():
            var = m.log_var()
            if var is not None:
                num += m.n * var
                den += m.n
        if den == 0:
            return self.prior.output_tokens_sigma
        k = self.token_prior_strength
        var = (num + k * self.prior.output_tokens_sigma**2) / (den + k)
        return math.sqrt(var)

    def provider(self) -> ProviderProfile:
        """Current provider beliefs.

        ``ttft_s`` / ``decode_tps`` describe the mean latency given tokens. Because they are fitted
        on raw latencies (which already contain whatever tail the provider has), the prior's tail
        factor is shrunk away and the measured residual spread becomes ``noise_sigma``.
        """
        params = self._latency_params()
        sigma_out = self._token_sigma()
        if params is None:
            return replace(self.prior, output_tokens_sigma=sigma_out)
        ttft, spt = params
        w = self._weight()
        noise = self.prior.noise_sigma
        log_var = self._residual.log_var()
        if log_var is not None:
            noise = math.sqrt(w * log_var + (1 - w) * self.prior.noise_sigma**2)
        return replace(
            self.prior,
            ttft_s=ttft,
            decode_tps=1 / spt,
            tail_prob=(1 - w) * self.prior.tail_prob,
            noise_sigma=noise,
            output_tokens_sigma=sigma_out,
        )

    def output_token_estimates(self) -> Mapping[TaskType, float]:
        return dict(self.output_tokens)

    def success_estimates(self) -> Mapping[TaskType, float]:
        return dict(self.success)
