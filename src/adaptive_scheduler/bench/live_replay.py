"""Offline re-analysis of a recorded live run, including a prequential replay of the estimator.

``load_live`` rebuilds a :class:`LiveResult` from the files ``write_live`` produced, so analysis
can be repeated without spending anything. ``replay`` walks through the recorded runs *in their
original order* and, for each one, predicts its latency with the current estimator state and only
*then* feeds that run's calls to the estimator - a prequential (predict-then-update) evaluation.
Every prediction is therefore out-of-sample, exactly as in the live run, which allows a changed
estimator to be evaluated on the same real data at no cost.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from adaptive_scheduler.bench.live import LiveResult, context_for
from adaptive_scheduler.cost.estimator import CostEstimator
from adaptive_scheduler.cost.profiles import build_profiles
from adaptive_scheduler.cost.provider import ProviderProfile
from adaptive_scheduler.graph.model import Task, TaskGraph, TaskType
from adaptive_scheduler.simulation.engine import ExecutionConfig
from adaptive_scheduler.simulation.selector import Objective, predict
from adaptive_scheduler.simulation.strategies import BASELINES, Strategy

_INT = {
    "request_id",
    "rep",
    "num_tasks",
    "tasks_executed",
    "model_calls",
    "input_tokens",
    "output_tokens",
    "peak_concurrency",
    "attempts",
    "concurrent_at_start",
}
_BOOL = {"calibrated", "success"}


def _typed(row: Mapping[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in row.items():
        if k in _BOOL:
            out[k] = v == "True"
        elif k in _INT:
            out[k] = int(float(v))
        else:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def load_live(directory: Path) -> LiveResult:
    def rows(name: str) -> list[dict[str, Any]]:
        path = directory / name
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as fh:
            return [_typed(r) for r in csv.DictReader(fh)]

    analysis = json.loads((directory / "analysis.json").read_text(encoding="utf-8"))
    return LiveResult(
        plans=json.loads((directory / "plans.json").read_text(encoding="utf-8")),
        runs=rows("runs.csv"),
        calls=rows("calls.csv"),
        calibration={
            k: analysis["calibration"][k]
            for k in ("prior", "fitted", "observations", "output_tokens_by_type", "success_by_type")
        },
        spent_usd=analysis["spent_usd"],
        stopped_early=analysis.get("stopped_early"),
    )


def _graph(plan: Mapping[str, Any]) -> TaskGraph:
    return TaskGraph(tasks=tuple(Task(**t) for t in plan["tasks"]))


def warm_estimator(result: LiveResult, prior: ProviderProfile) -> CostEstimator:
    """An estimator that has already observed every call of a recorded run (in memory)."""
    est = CostEstimator(prior=prior)
    for call in result.calls:
        task_type = TaskType(call["task_type"])
        est.observe(
            task_type,
            latency_s=call["latency_s"],
            input_tokens=call["input_tokens"],
            output_tokens=call["output_tokens"],
            succeeded=call["status"] == "completed",
            update_type_stats=task_type is not TaskType.COMPOSITE,
        )
    return est


def config_from_run(run: Mapping[str, Any]) -> ExecutionConfig:
    strategy = Strategy(run["strategy"])
    if strategy is not Strategy.ADAPTIVE:
        return BASELINES[strategy]
    name: str = run["config"]
    workers = int(name.split("k=")[1].split(",", maxsplit=1)[0].rstrip("]"))
    return ExecutionConfig(name, workers=workers, fuse="fused" in name)


def replay(
    result: LiveResult,
    prior: ProviderProfile,
    *,
    estimator: CostEstimator | None = None,
    samples: int = 64,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Prequential replay; returns run rows with the replayed predictions substituted."""
    est = estimator or CostEstimator(prior=prior)
    plans = {p["request_id"]: p for p in result.plans if "tasks" in p}
    calls: dict[tuple[int, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for c in result.calls:
        calls[(c["request_id"], c["rep"], c["strategy"])].append(c)

    out = []
    for run in result.runs:
        plan = plans.get(run["request_id"])
        if plan is None:
            continue
        graph = _graph(plan)
        ctx = context_for(plan["request"])
        profiles = build_profiles(graph, ctx, est.output_token_estimates(), est.success_estimates())
        pred = predict(
            graph,
            profiles,
            est.provider(),
            config_from_run(run),
            samples=samples,
            seed=seed,
            objective=Objective(),
            per_call_context_tokens=ctx.per_call,
        )
        overhead = run.get("selection_s", 0.0)
        out.append(
            dict(run)
            | {
                "predicted_mean_s": pred.mean_latency + overhead,
                "predicted_p50_s": pred.p50_latency + overhead,
                "predicted_p95_s": pred.p95_latency + overhead,
            }
        )
        for call in calls[(run["request_id"], run["rep"], run["strategy"])]:
            task_type = TaskType(call["task_type"])
            est.observe(
                task_type,
                latency_s=call["latency_s"],
                input_tokens=call["input_tokens"],
                output_tokens=call["output_tokens"],
                succeeded=call["status"] == "completed",
                update_type_stats=task_type is not TaskType.COMPOSITE,
            )
    return out


def replay_summary(
    result: LiveResult, prior: ProviderProfile, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    from adaptive_scheduler.bench.live_report import _errors  # noqa: PLC0415

    return {
        "prior_model": _errors([r for r in rows if not r["calibrated"]]),
        "calibrated_model": _errors([r for r in rows if r["calibrated"]]),
        "late_runs": _errors([r for r in rows if r["rep"] == max(x["rep"] for x in rows)]),
    }
