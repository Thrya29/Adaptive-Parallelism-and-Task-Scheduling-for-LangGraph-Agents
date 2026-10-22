"""Model-based strategy selection: predict, per request, how to execute its task graph.

For every candidate execution config (concurrency ``k`` x fusion on/off) the selector runs ``M``
Monte Carlo simulations with the *estimated* cost model and predicts mean latency, P95 latency,
token cost, number of model calls and success rate. It then picks the candidate minimizing

    J = mean_latency + alpha * p95_latency + usd_weight * cost_usd

among candidates whose success probability is within ``success_tolerance`` of the best.

Success probability is computed *analytically*, not from the Monte Carlo runs: in the cost model
a task's attempts succeed independently of timing, so a plan succeeds with probability
``prod_t (1 - (1 - p_t) ** (retries + 1))``. It depends on the configuration only through task
fusion (which changes the set of calls and their success probabilities). Using the Monte Carlo
estimate instead let sampling noise (standard error ~2% at p = 0.97 with 64 samples) exclude
good configurations - a flaw the benchmark exposed under the ``unreliable`` condition.
Near-ties (within ``parsimony``) go to the simplest architecture - fewest workers, then fewest
calls - so an inherently sequential plan is executed sequentially rather than paying
coordination overhead for nothing.

All candidates are evaluated on the same random seeds (common random numbers), which reduces the
variance of their *differences* - what the comparison actually depends on.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from adaptive_scheduler.cost.profiles import TaskProfile
from adaptive_scheduler.cost.provider import ProviderProfile
from adaptive_scheduler.graph.model import TaskGraph
from adaptive_scheduler.simulation.engine import (
    ExecutionConfig,
    RetryPolicy,
    RunResult,
    prepare,
    simulate,
)

DEFAULT_WORKER_OPTIONS: tuple[int, ...] = (1, 2, 4, 8, 16)


@dataclass(frozen=True, slots=True)
class Objective:
    alpha_p95: float = 0.5
    usd_weight: float = 100.0  # seconds of latency one is willing to trade for $1
    success_tolerance: float = 0.01
    # Among candidates scoring within this relative margin of the best, prefer the simplest
    # (fewest workers, then fewest calls): no extra concurrency for noise-level gains.
    parsimony: float = 0.01

    def score(self, mean_latency: float, p95_latency: float, cost_usd: float) -> float:
        return mean_latency + self.alpha_p95 * p95_latency + self.usd_weight * cost_usd


@dataclass(frozen=True, slots=True)
class Prediction:
    config: ExecutionConfig
    mean_latency: float
    p95_latency: float
    mean_cost_usd: float
    mean_calls: float
    success_rate: float  # Monte Carlo estimate (reported)
    expected_success: float  # exact probability under the cost model (used for selection)
    score: float
    p50_latency: float = 0.0

    @classmethod
    def from_runs(
        cls,
        config: ExecutionConfig,
        runs: Sequence[RunResult],
        objective: Objective,
        expected_success: float,
    ) -> Prediction:
        latencies = np.array([r.makespan for r in runs])
        mean, p95 = float(latencies.mean()), float(np.percentile(latencies, 95))
        cost = float(np.mean([r.cost_usd for r in runs]))
        return cls(
            config=config,
            mean_latency=mean,
            p95_latency=p95,
            mean_cost_usd=cost,
            mean_calls=float(np.mean([r.model_calls for r in runs])),
            success_rate=float(np.mean([r.success for r in runs])),
            expected_success=expected_success,
            score=objective.score(mean, p95, cost),
            p50_latency=float(np.percentile(latencies, 50)),
        )


@dataclass(frozen=True)
class Selection:
    chosen: Prediction
    candidates: tuple[Prediction, ...]
    selection_time_s: float


def candidate_configs(
    graph: TaskGraph, worker_options: Sequence[int] = DEFAULT_WORKER_OPTIONS
) -> list[ExecutionConfig]:
    """Concurrency levels up to the graph's maximum width (more workers can never be used)."""
    useful = sorted({min(k, graph.max_width) for k in worker_options if k >= 1})
    return [
        ExecutionConfig(name=f"adaptive[k={k}{',fused' if fuse else ''}]", workers=k, fuse=fuse)
        for k in useful
        for fuse in (False, True)
    ]


def predict(
    graph: TaskGraph,
    profiles: Mapping[str, TaskProfile],
    provider: ProviderProfile,
    config: ExecutionConfig,
    *,
    samples: int,
    seed: int,
    objective: Objective,
    per_call_context_tokens: float = 0.0,
    retry: RetryPolicy | None = None,
) -> Prediction:
    retry = retry or RetryPolicy()
    plan = prepare(graph, profiles, provider, config, per_call_context_tokens)
    runs = [
        simulate(plan, provider, config, np.random.default_rng([seed, i]), retry)
        for i in range(samples)
    ]
    return Prediction.from_runs(
        config, runs, objective, plan_success_probability(plan.profiles.values(), retry)
    )


def plan_success_probability(profiles: Iterable[TaskProfile], retry: RetryPolicy) -> float:
    """Probability that every task eventually succeeds within its retry budget."""
    attempts = retry.max_retries + 1
    return math.prod(1 - (1 - p.success_prob) ** attempts for p in profiles)


def select_strategy(
    graph: TaskGraph,
    profiles: Mapping[str, TaskProfile],
    provider: ProviderProfile,
    *,
    objective: Objective | None = None,
    candidates: Sequence[ExecutionConfig] | None = None,
    samples: int = 64,
    seed: int = 0,
    per_call_context_tokens: float = 0.0,
    retry: RetryPolicy | None = None,
) -> Selection:
    if samples < 1:
        raise ValueError("samples must be >= 1")
    objective = objective or Objective()
    configs = list(candidates) if candidates is not None else candidate_configs(graph)
    if not configs:
        raise ValueError("no candidate configs")

    started = time.perf_counter()
    predictions = tuple(
        predict(
            graph,
            profiles,
            provider,
            cfg,
            samples=samples,
            seed=seed,
            objective=objective,
            per_call_context_tokens=per_call_context_tokens,
            retry=retry,
        )
        for cfg in configs
    )
    best_success = max(p.expected_success for p in predictions)
    feasible = [
        p for p in predictions if p.expected_success >= best_success - objective.success_tolerance
    ]
    best_score = min(p.score for p in feasible)
    acceptable = [p for p in feasible if p.score <= best_score * (1 + objective.parsimony)]

    def simplicity(p: Prediction) -> tuple[float, float, float]:
        workers = math.inf if p.config.workers is None else p.config.workers
        return (workers, p.mean_calls, p.score)

    chosen = min(acceptable, key=simplicity)
    return Selection(chosen, predictions, time.perf_counter() - started)
