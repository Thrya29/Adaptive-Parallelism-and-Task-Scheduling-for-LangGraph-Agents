"""Live validation: run the scheduler against a real LLM API and check the simulation's claims.

Protocol
--------
1. **Plan once per request** with the real planner, then execute that same plan under every
   strategy, so plan variance does not contaminate the strategy comparison.
2. **Repetitions.** Repetition 0 runs on the prior cost model; every later repetition plans with
   the estimator calibrated on all calls observed so far. Each run's latency is *predicted before
   it executes*, so every prediction is out-of-sample.
3. **Shuffled order.** Strategy order is shuffled per (request, repetition) so drift in API load
   over time cannot systematically favour one strategy.
4. **Budget guard.** The run stops (keeping everything measured so far) once estimated spend
   reaches ``budget_usd``.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel

from adaptive_scheduler.agent import prompts
from adaptive_scheduler.agent.executor import ExecutionReport, TaskExecutor, call_kwargs_for
from adaptive_scheduler.agent.planning import Planner, PlanningError
from adaptive_scheduler.agent.workflow import learn_from_report
from adaptive_scheduler.bench.workloads import REQUESTS
from adaptive_scheduler.cost.estimator import CostEstimator
from adaptive_scheduler.cost.profiles import ContextSize, build_profiles, estimate_tokens
from adaptive_scheduler.cost.provider import ProviderProfile
from adaptive_scheduler.graph.analysis import profile as graph_profile
from adaptive_scheduler.graph.model import PlanLimitError, TaskGraph
from adaptive_scheduler.simulation.engine import ExecutionConfig, RetryPolicy, prepare
from adaptive_scheduler.simulation.selector import Objective, predict, select_strategy
from adaptive_scheduler.simulation.strategies import BASELINES, Strategy


class BudgetExceededError(RuntimeError):
    pass


@dataclass(frozen=True)
class LiveConfig:
    requests: tuple[str, ...] = REQUESTS
    strategies: tuple[Strategy, ...] = tuple(Strategy)
    reps: int = 3
    budget_usd: float = 2.0
    selector_samples: int = 64
    max_plan_tasks: int = 16
    seed: int = 0
    objective: Objective = field(default_factory=Objective)

    def __post_init__(self) -> None:
        if self.reps < 1:
            raise ValueError("reps must be >= 1")
        if self.budget_usd <= 0:
            raise ValueError("budget_usd must be > 0")


@dataclass
class LiveResult:
    plans: list[dict[str, Any]] = field(default_factory=list)
    runs: list[dict[str, Any]] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    calibration: dict[str, Any] = field(default_factory=dict)
    spent_usd: float = 0.0
    stopped_early: str | None = None


def context_for(request: str) -> ContextSize:
    return ContextSize(
        system_tokens=estimate_tokens(prompts.WORKER_SYSTEM),
        shared_context_tokens=estimate_tokens(prompts.worker_request(request, "", "", "", {})),
    )


def _call_rows(
    request_id: int,
    rep: int,
    strategy: str,
    graph: TaskGraph,
    report: ExecutionReport,
    *,
    time_scale: float = 1.0,
) -> list[dict[str, Any]]:
    """One row per executed task: latency, tokens and how many tasks were running at its start."""
    recs = [r for r in report.records.values() if r.start_s is not None and r.end_s is not None]
    rows = []
    for r in recs:
        if not r.call_latencies_s:
            continue
        start = r.start_s if r.start_s is not None else 0.0
        concurrent = sum(
            1
            for o in recs
            if o.start_s is not None and o.end_s is not None and o.start_s <= start < o.end_s
        )
        rows.append(
            {
                "request_id": request_id,
                "rep": rep,
                "strategy": strategy,
                "task_id": r.task_id,
                "task_type": graph.task(r.task_id).task_type.value,
                "status": r.status.value,
                "attempts": r.attempts,
                "latency_s": r.call_latencies_s[-1] / time_scale,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "concurrent_at_start": concurrent,
            }
        )
    return rows


async def _plan(
    planner: Planner, request: str, max_tasks: int
) -> tuple[TaskGraph, int, float, int, int]:
    """Plan with one repair attempt; returns (graph, attempts, seconds, input, output tokens)."""
    feedback: str | None = None
    tokens_in = tokens_out = 0
    started = time.perf_counter()
    for attempt in (1, 2):
        try:
            spec, usage = await planner.propose(request, feedback)
            tokens_in += usage.input_tokens
            tokens_out += usage.output_tokens
            graph = spec.to_graph()
            graph.check_limits(max_tasks=max_tasks, max_depth=12, max_fan_in=16)
            elapsed = time.perf_counter() - started
            return graph.transitive_reduction(), attempt, elapsed, tokens_in, tokens_out
        except (PlanningError, PlanLimitError, ValueError) as exc:
            if isinstance(exc, PlanningError) and exc.usage:
                tokens_in += exc.usage.input_tokens
                tokens_out += exc.usage.output_tokens
            feedback = str(exc)
    raise PlanningError(f"planning failed twice: {feedback}")


async def run_live(
    cfg: LiveConfig,
    llm: BaseChatModel,
    planner: Planner,
    provider_prior: ProviderProfile,
    *,
    retry: RetryPolicy | None = None,
    timeout_s: float = 120.0,
    time_scale: float = 1.0,
    estimator: CostEstimator | None = None,
    progress: Callable[[str], None] | None = None,
) -> LiveResult:
    """``time_scale`` is wall-clock seconds per model second (1.0 for a real API; the mock LLM can
    run faster). All reported latencies are in model seconds."""
    retry = retry or RetryPolicy()
    say = progress or (lambda _msg: None)
    # A warm-started estimator (already calibrated on earlier data) makes every run "calibrated".
    warm = estimator is not None and estimator.observations > 0
    estimator = estimator or CostEstimator(prior=provider_prior)
    executor = TaskExecutor(
        llm, retry=retry, timeout_s=timeout_s, seed=cfg.seed, call_kwargs=call_kwargs_for(llm)
    )
    rng = random.Random(cfg.seed)  # noqa: S311 - shuffles run order; not security-relevant
    result = LiveResult()

    def charge(tokens_in: float, tokens_out: float) -> float:
        cost = provider_prior.cost_usd(tokens_in, tokens_out)
        result.spent_usd += cost
        return cost

    graphs: list[TaskGraph | None] = []
    for i, request in enumerate(cfg.requests):
        try:
            graph, attempts, seconds, t_in, t_out = await _plan(
                planner, request, cfg.max_plan_tasks
            )
        except PlanningError as exc:
            say(f"plan {i}: FAILED ({exc})")
            graphs.append(None)
            result.plans.append({"request_id": i, "request": request, "error": str(exc)})
            continue
        charge(t_in, t_out)
        graphs.append(graph)
        result.plans.append(
            {
                "request_id": i,
                "request": request,
                "attempts": attempts,
                "planning_s": seconds,
                "input_tokens": t_in,
                "output_tokens": t_out,
                "num_tasks": len(graph),
                "depth": graph.depth,
                "max_width": graph.max_width,
                "tasks": [t.model_dump(mode="json") for t in graph.tasks],
            }
        )
        say(f"plan {i}: {len(graph)} tasks, width {graph.max_width}, depth {graph.depth}")

    try:
        for rep in range(cfg.reps):
            for i, request in enumerate(cfg.requests):
                planned = graphs[i]
                if planned is None:
                    continue
                order = list(cfg.strategies)
                rng.shuffle(order)
                for strategy in order:
                    if result.spent_usd >= cfg.budget_usd:
                        raise BudgetExceededError(
                            f"budget ${cfg.budget_usd:.2f} reached after ${result.spent_usd:.4f}"
                        )
                    row = await _run_one(
                        cfg,
                        executor=executor,
                        estimator=estimator,
                        request=request,
                        request_id=i,
                        rep=rep,
                        strategy=strategy,
                        graph=planned,
                        retry=retry,
                        result=result,
                        time_scale=time_scale,
                        warm=warm,
                    )
                    row["cost_usd"] = charge(row["input_tokens"], row["output_tokens"])
                    result.runs.append(row)
                    say(
                        f"rep {rep} req {i} {strategy.value:15s} {row['config']:22s} "
                        f"{row['latency_s']:6.1f}s (predicted {row['predicted_mean_s']:6.1f}s) "
                        f"spent ${result.spent_usd:.4f}"
                    )
    except BudgetExceededError as exc:
        result.stopped_early = str(exc)
        say(f"stopped: {exc}")

    fitted = estimator.provider()
    result.calibration = {
        "prior": {"ttft_s": provider_prior.ttft_s, "decode_tps": provider_prior.decode_tps},
        "fitted": {"ttft_s": fitted.ttft_s, "decode_tps": fitted.decode_tps},
        "observations": estimator.observations,
        "output_tokens_by_type": {k.value: v for k, v in estimator.output_tokens.items()},
        "success_by_type": {k.value: v for k, v in estimator.success.items()},
    }
    return result


async def _run_one(
    cfg: LiveConfig,
    *,
    executor: TaskExecutor,
    estimator: CostEstimator,
    request: str,
    request_id: int,
    rep: int,
    strategy: Strategy,
    graph: TaskGraph,
    retry: RetryPolicy,
    result: LiveResult,
    time_scale: float,
    warm: bool = False,
) -> dict[str, Any]:
    ctx = context_for(request)
    beliefs = estimator.provider()
    profiles = build_profiles(
        graph, ctx, estimator.output_token_estimates(), estimator.success_estimates()
    )
    overhead = 0.0
    if strategy is Strategy.ADAPTIVE:
        sel = select_strategy(
            graph,
            profiles,
            beliefs,
            objective=cfg.objective,
            samples=cfg.selector_samples,
            seed=cfg.seed + rep,
            per_call_context_tokens=ctx.per_call,
            retry=retry,
        )
        config: ExecutionConfig = sel.chosen.config
        prediction = sel.chosen
        overhead = sel.selection_time_s
    else:
        config = BASELINES[strategy]
        prediction = predict(
            graph,
            profiles,
            beliefs,
            config,
            samples=cfg.selector_samples,
            seed=cfg.seed + rep,
            objective=cfg.objective,
            per_call_context_tokens=ctx.per_call,
            retry=retry,
        )

    plan = prepare(graph, profiles, beliefs, config, ctx.per_call)
    report = await executor.run(request, plan, config)
    learn_from_report(estimator, plan.graph, report, time_scale)
    result.calls.extend(
        _call_rows(request_id, rep, strategy.value, plan.graph, report, time_scale=time_scale)
    )

    durations = {
        t: beliefs.expected_latency(p.input_tokens, p.output_tokens) for t, p in profiles.items()
    }
    gp = graph_profile(graph, durations)
    return {
        "request_id": request_id,
        "rep": rep,
        "strategy": strategy.value,
        "config": config.name,
        "calibrated": rep > 0 or warm,
        "num_tasks": len(graph),
        "tasks_executed": len(plan.graph),
        "parallelism_est": gp.parallelism,
        "latency_s": report.makespan_s / time_scale + overhead,
        "selection_s": overhead,
        "predicted_mean_s": prediction.mean_latency,
        "predicted_p50_s": prediction.p50_latency,
        "predicted_p95_s": prediction.p95_latency,
        "success": report.success,
        "model_calls": report.model_calls,
        "input_tokens": report.input_tokens,
        "output_tokens": report.output_tokens,
        "peak_concurrency": report.peak_concurrency,
    }


def summarize_strategies(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    from adaptive_scheduler.bench.stats import paired_ratio_ci  # noqa: PLC0415

    by: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}
    for r in runs:
        by.setdefault(r["strategy"], {})[(r["request_id"], r["rep"])] = r
    seq = by.get("sequential", {})
    out = []
    for strategy, cells in by.items():
        keys = sorted(k for k in cells if k in seq)
        lat = [cells[k]["latency_s"] for k in keys]
        speed, lo, hi = (
            paired_ratio_ci([seq[k]["latency_s"] for k in keys], lat)
            if keys
            else (float("nan"), float("nan"), float("nan"))
        )
        rows = list(cells.values())
        out.append(
            {
                "strategy": strategy,
                "runs": len(rows),
                "mean_latency_s": sum(r["latency_s"] for r in rows) / len(rows),
                "p95_latency_s": sorted(r["latency_s"] for r in rows)[
                    max(0, int(0.95 * len(rows)) - 1)
                ],
                "mean_cost_usd": sum(r["cost_usd"] for r in rows) / len(rows),
                "mean_calls": sum(r["model_calls"] for r in rows) / len(rows),
                "success_rate": sum(r["success"] for r in rows) / len(rows),
                "speedup": speed,
                "speedup_lo": lo,
                "speedup_hi": hi,
            }
        )
    order = [s.value for s in Strategy]
    return sorted(out, key=lambda s: order.index(s["strategy"]) if s["strategy"] in order else 99)
