"""Experiment runner: every strategy on every (workload, condition) unit, many seeds.

Protocol per unit
-----------------
1. **Plan with beliefs, evaluate against truth.** Priorities, fusion and the selector use the
   condition's *belief* provider; every run is simulated with the *true* provider.
2. **Separate randomness.** The selector predicts on its own seeds; all strategies are then scored
   on shared evaluation seeds (common random numbers, so comparisons are paired).
3. **Strategies.**

   * fixed baselines - ``sequential``, ``static_parallel`` (level barriers), ``eager_parallel``
     (unbounded greedy, LLMCompiler-style), ``blind_parallel`` (dependency-blind agents + a
     decision agent);
   * ``adaptive`` - Monte Carlo selector over ``k`` x fusion with critical-path priority;
   * ablations - ``adaptive_fifo`` (FIFO priority), ``adaptive_nofusion`` (no fusion),
     ``fixed_k_cap`` (no cost model: ``k = min(provider cap, width)``, critical-path priority).

   The selector's wall-clock time is added to the latency of every strategy that uses it.
4. **Hindsight check.** Every candidate is also scored on a large, separate *oracle* sample
   (``oracle_runs`` seeds), so we know which configuration was actually best and how much the
   selector's choice gave up. A small oracle is itself noisy: under heavy-tailed latency the mean
   of 30 runs has a ~±30% confidence interval, which made correct choices look like 24% regret.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from adaptive_scheduler.bench.conditions import CONDITIONS, Condition, blind_variant
from adaptive_scheduler.bench.workloads import CONTEXT, FAMILIES, Workload, stable_seed, suite
from adaptive_scheduler.cost.profiles import TaskProfile, expected_durations
from adaptive_scheduler.cost.provider import ProviderProfile
from adaptive_scheduler.graph.analysis import profile as graph_profile
from adaptive_scheduler.graph.model import TaskGraph
from adaptive_scheduler.scheduling.simulator import PriorityPolicy
from adaptive_scheduler.simulation.engine import ExecutionConfig, RunResult, prepare, simulate
from adaptive_scheduler.simulation.selector import Objective, candidate_configs, select_strategy
from adaptive_scheduler.simulation.strategies import BASELINES, Strategy

FIXED = ("sequential", "static_parallel", "eager_parallel", "blind_parallel")
ADAPTIVE = ("adaptive", "adaptive_fifo", "adaptive_nofusion", "fixed_k_cap")
ALL_STRATEGIES = FIXED + ADAPTIVE


@dataclass(frozen=True)
class BenchConfig:
    families: tuple[str, ...] = (*FAMILIES, "realistic")
    instances: int = 4
    conditions: tuple[str, ...] = tuple(CONDITIONS)
    eval_runs: int = 100
    oracle_runs: int = 200  # per candidate, for hindsight scoring only
    selector_samples: int = 64
    blind_quality: float = 0.8  # chance a task that ran without its inputs is still correct
    heterogeneity: float = 0.6
    seed: int = 0
    objective: Objective = field(default_factory=Objective)

    def __post_init__(self) -> None:
        unknown = set(self.conditions) - set(CONDITIONS)
        if unknown:
            raise ValueError(f"unknown conditions: {sorted(unknown)}")
        if not 0 < self.blind_quality <= 1:
            raise ValueError("blind_quality must be in (0, 1]")
        if self.eval_runs < 2 or self.oracle_runs < 2:
            raise ValueError("eval_runs and oracle_runs must be >= 2")


PROFILES: dict[str, BenchConfig] = {
    "quick": BenchConfig(
        families=("chain", "fan_out", "realistic"),
        instances=1,
        conditions=("baseline", "tight_limit"),
        eval_runs=6,
        oracle_runs=20,
        selector_samples=12,
    ),
    "full": BenchConfig(),
}


@dataclass
class UnitResult:
    runs: list[dict[str, Any]]
    selections: list[dict[str, Any]]
    blind: list[dict[str, Any]]


def _evaluate(
    graph: TaskGraph,
    profiles: dict[str, TaskProfile],
    config: ExecutionConfig,
    *,
    belief: ProviderProfile,
    truth: ProviderProfile,
    seeds: Sequence[int],
) -> list[RunResult]:
    plan = prepare(graph, profiles, belief, config, CONTEXT.per_call)
    return [simulate(plan, truth, config, np.random.default_rng(s)) for s in seeds]


def _score(runs: Sequence[RunResult], objective: Objective) -> float:
    lat = np.array([r.makespan for r in runs])
    cost = float(np.mean([r.cost_usd for r in runs]))
    return objective.score(float(lat.mean()), float(np.percentile(lat, 95)), cost)


def run_unit(workload: Workload, condition: Condition, cfg: BenchConfig) -> UnitResult:
    graph = workload.graph
    profiles = condition.adjust(workload.profiles)
    truth, belief = condition.true_provider, condition.beliefs
    seeds = [
        stable_seed("eval", workload.name, condition.name, cfg.seed, r)
        for r in range(cfg.eval_runs)
    ]
    selector_seed = stable_seed("select", workload.name, condition.name, cfg.seed)
    gp = graph_profile(graph, expected_durations(profiles, truth))
    base = {
        "workload": workload.name,
        "family": workload.family,
        "n_tasks": len(graph),
        "parallelism": gp.parallelism,
        "condition": condition.name,
    }

    runs: list[dict[str, Any]] = []

    def record(
        strategy: str,
        config_name: str,
        results: Sequence[RunResult],
        overhead: float = 0.0,
        quality_ok: Sequence[bool] | None = None,
    ) -> None:
        for i, r in enumerate(results):
            ok = r.success and (quality_ok[i] if quality_ok is not None else True)
            runs.append(
                base
                | {
                    "strategy": strategy,
                    "config": config_name,
                    "seed_index": i,
                    "latency_s": r.makespan + overhead,
                    "cost_usd": r.cost_usd,
                    "model_calls": r.model_calls,
                    "rate_limited": r.rate_limited_calls,
                    "success": ok,
                    "overhead_s": overhead,
                }
            )

    # Fixed baselines.
    for name in ("sequential", "static_parallel", "eager_parallel"):
        config = BASELINES[Strategy(name)]
        record(
            name,
            config.name,
            _evaluate(graph, profiles, config, belief=belief, truth=truth, seeds=seeds),
        )

    blind = blind_variant(graph, profiles, CONTEXT.per_call)
    blind_cfg = ExecutionConfig("blind_parallel", workers=None, policy=PriorityPolicy.FIFO)
    blind_runs = _evaluate(
        blind.graph, blind.profiles, blind_cfg, belief=belief, truth=truth, seeds=seeds
    )
    quality_p = cfg.blind_quality**blind.blind_tasks
    quality_rng = np.random.default_rng(
        stable_seed("quality", workload.name, condition.name, cfg.seed)
    )
    record(
        "blind_parallel",
        blind_cfg.name,
        blind_runs,
        quality_ok=[bool(quality_rng.random() < quality_p) for _ in blind_runs],
    )
    exec_success = float(np.mean([r.success for r in blind_runs]))
    blind_rows = [
        base
        | {
            "blind_tasks": blind.blind_tasks,
            "quality": q,
            "exec_success": exec_success,
            "expected_success": exec_success * q**blind.blind_tasks,
        }
        for q in (0.6, 0.7, 0.8, 0.9, 0.95, 1.0)
    ]

    # Candidate configurations. Every candidate is scored on the large oracle sample (hindsight);
    # only chosen ones are run on the shared evaluation seeds for the strategy comparison.
    cp = candidate_configs(graph)
    fifo = [
        replace(c, name=c.name.replace("adaptive[", "fifo["), policy=PriorityPolicy.FIFO)
        for c in cp
    ]
    oracle_seeds = [
        stable_seed("oracle", workload.name, condition.name, cfg.seed, r)
        for r in range(cfg.oracle_runs)
    ]
    oracle = {
        c.name: _evaluate(graph, profiles, c, belief=belief, truth=truth, seeds=oracle_seeds)
        for c in (*cp, *fifo)
    }
    by_name = {c.name: c for c in (*cp, *fifo)}

    cap_k = min(belief.max_concurrency, graph.max_width)
    cap_cfg = ExecutionConfig(f"fixed[k={cap_k}]", workers=cap_k)
    record(
        "fixed_k_cap",
        cap_cfg.name,
        _evaluate(graph, profiles, cap_cfg, belief=belief, truth=truth, seeds=seeds),
    )

    selections: list[dict[str, Any]] = []
    variants: dict[str, list[ExecutionConfig]] = {
        "adaptive": cp,
        "adaptive_nofusion": [c for c in cp if not c.fuse],
        "adaptive_fifo": fifo,
    }
    for variant, candidates in variants.items():
        sel = select_strategy(
            graph,
            profiles,
            belief,
            objective=cfg.objective,
            candidates=candidates,
            samples=cfg.selector_samples,
            seed=selector_seed,
            per_call_context_tokens=CONTEXT.per_call,
        )
        chosen = sel.chosen.config.name
        chosen_runs = _evaluate(
            graph, profiles, by_name[chosen], belief=belief, truth=truth, seeds=seeds
        )
        record(variant, chosen, chosen_runs, overhead=sel.selection_time_s)
        scores = {c.name: _score(oracle[c.name], cfg.objective) for c in candidates}
        best = min(scores, key=lambda n: scores[n])
        selections.append(
            base
            | {
                "variant": variant,
                "chosen": chosen,
                "hindsight_best": best,
                "chosen_score": scores[chosen],
                "best_score": scores[best],
                "regret": scores[chosen] / scores[best] - 1 if scores[best] > 0 else 0.0,
                "exact": chosen == best,
                "selection_time_s": sel.selection_time_s,
                "predicted_mean_s": sel.chosen.mean_latency,
                "realized_mean_s": float(np.mean([r.makespan for r in oracle[chosen]])),
            }
        )
    return UnitResult(runs, selections, blind_rows)


def _run_unit_args(args: tuple[Workload, str, BenchConfig]) -> UnitResult:
    workload, condition, cfg = args
    return run_unit(workload, CONDITIONS[condition], cfg)


def run(
    cfg: BenchConfig,
    jobs: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> UnitResult:
    """Run the full benchmark; ``jobs > 1`` distributes units over processes."""
    workloads = suite(cfg.families, cfg.instances, cfg.seed, cfg.heterogeneity)
    units = [(w, c, cfg) for c in cfg.conditions for w in workloads]
    total = UnitResult([], [], [])
    results: Iterable[UnitResult]
    if jobs > 1:
        pool = ProcessPoolExecutor(max_workers=jobs)
        results = pool.map(
            _run_unit_args, units, chunksize=max(1, math.ceil(len(units) / (jobs * 8)))
        )
    else:
        pool = None
        results = map(_run_unit_args, units)
    try:
        for i, res in enumerate(results, 1):
            total.runs += res.runs
            total.selections += res.selections
            total.blind += res.blind
            if progress:
                progress(i, len(units))
    finally:
        if pool is not None:
            pool.shutdown()
    return total
