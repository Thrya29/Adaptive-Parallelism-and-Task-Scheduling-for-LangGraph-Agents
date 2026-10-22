"""Analysis and report for the live validation run."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from adaptive_scheduler.bench.live import LiveResult, summarize_strategies
from adaptive_scheduler.cost.provider import ProviderProfile


def _errors(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Evaluate the latency *distribution* forecast, not single draws.

    LLM latency is noisy and right-skewed, so one run versus the predicted mean mostly measures
    intrinsic variance (a correct model even shows a positive median error, because the typical
    run is faster than the mean). The calibration checks are therefore:

    * aggregate bias = sum(predicted mean) / sum(actual) - 1   (0 for an unbiased mean)
    * P50 / P95 coverage = share of runs at or below the predicted quantile (ideal 50% / 95%)
    """
    rows = [r for r in rows if r["latency_s"] > 0]
    nan = float("nan")
    if not rows:
        return {
            "n": 0,
            "aggregate_bias": nan,
            "p50_coverage": nan,
            "p95_coverage": nan,
            "mape": nan,
        }
    actual = np.array([r["latency_s"] for r in rows], dtype=float)
    mean = np.array([r["predicted_mean_s"] for r in rows], dtype=float)
    p50 = np.array([r.get("predicted_p50_s", nan) for r in rows], dtype=float)
    p95 = np.array([r["predicted_p95_s"] for r in rows], dtype=float)
    return {
        "n": len(rows),
        "aggregate_bias": float(np.sum(mean)) / float(np.sum(actual)) - 1,
        "p50_coverage": float(np.mean(actual <= p50)),
        "p95_coverage": float(np.mean(actual <= p95)),
        "mape": float(np.mean(np.abs(mean - actual) / actual)),
    }


def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def analyse_live(result: LiveResult, prior: ProviderProfile) -> dict[str, Any]:
    runs = result.runs
    calibrated = [r for r in runs if r["calibrated"]]

    # Ranking: per request, order strategies by mean predicted vs mean actual (calibrated reps).
    ranking: list[dict[str, Any]] = []
    by_req: dict[int, dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in calibrated:
        by_req[r["request_id"]][r["strategy"]].append(r)
    for rid, strategies in sorted(by_req.items()):
        # Rank only where the strategies can actually differ: a plan with no parallelism runs
        # identically under every strategy, so its "ranking" would be pure noise.
        if any(x["parallelism_est"] <= 1.0 + 1e-9 for runs_ in strategies.values() for x in runs_):
            continue
        names = sorted(strategies)
        actual = [statistics.fmean(x["latency_s"] for x in strategies[n]) for n in names]
        predicted = [statistics.fmean(x["predicted_mean_s"] for x in strategies[n]) for n in names]
        spread = (max(actual) - min(actual)) / min(actual) if min(actual) > 0 else 0.0
        ranking.append(
            {
                "request_id": rid,
                "spearman": _spearman(predicted, actual),
                "predicted_best": names[int(np.argmin(predicted))],
                "actual_best": names[int(np.argmin(actual))],
                "actual_spread": spread,
            }
        )
    rhos = [x["spearman"] for x in ranking if not np.isnan(x["spearman"])]

    # Call-level latency model and contention.
    fitted = ProviderProfile(
        ttft_s=result.calibration["fitted"]["ttft_s"],
        decode_tps=result.calibration["fitted"]["decode_tps"],
        tail_prob=prior.tail_prob,
        tail_multiplier=prior.tail_multiplier,
        prefill_tps=prior.prefill_tps,
    )
    calls = [c for c in result.calls if c["status"] == "completed" and c["attempts"] == 1]

    def call_mape(p: ProviderProfile) -> float:
        errs = [
            abs(p.expected_latency(c["input_tokens"], c["output_tokens"]) - c["latency_s"])
            / c["latency_s"]
            for c in calls
            if c["latency_s"] > 0
        ]
        return float(np.mean(errs)) if errs else float("nan")

    contention = float("nan")
    if calls:
        x = np.array([c["concurrent_at_start"] - 1 for c in calls], dtype=float)
        base = np.array([fitted.base_latency(c["input_tokens"], c["output_tokens"]) for c in calls])
        y = np.array([c["latency_s"] for c in calls]) / base - 1
        if x.std() > 0:
            contention = float(np.polyfit(x, y, 1)[0])
    decode = [c["output_tokens"] / c["latency_s"] for c in calls if c["latency_s"] > 0]

    return {
        "spent_usd": result.spent_usd,
        "stopped_early": result.stopped_early,
        "requests": len(result.plans),
        "plans": [{k: v for k, v in p.items() if k != "tasks"} for p in result.plans],
        "runs": len(runs),
        "calls": len(result.calls),
        "strategies": summarize_strategies(runs) if runs else [],
        "prediction": {
            "prior_model": _errors([r for r in runs if not r["calibrated"]]),
            "calibrated_model": _errors(calibrated),
        },
        "ranking": {
            "per_request": ranking,
            "mean_spearman": float(np.mean(rhos)) if rhos else float("nan"),
            "top1_agreement": float(
                np.mean([x["predicted_best"] == x["actual_best"] for x in ranking])
            )
            if ranking
            else float("nan"),
        },
        "calibration": result.calibration
        | {
            "call_mape_prior": call_mape(prior),
            "call_mape_fitted": call_mape(fitted),
            "contention_per_extra_call": contention,
            "median_tokens_per_s_end_to_end": float(np.median(decode)) if decode else float("nan"),
        },
    }


def _fmt(x: float, spec: str = ".2f") -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else format(x, spec)


def render_live_markdown(a: Mapping[str, Any], model: str) -> str:
    cal, pred, rank = a["calibration"], a["prediction"], a["ranking"]
    lines = [
        "# Live validation results",
        "",
        f"Model `{model}`; {a['requests']} requests, {a['runs']} executions, {a['calls']} task calls; "
        f"estimated spend ${a['spent_usd']:.4f}.",
    ]
    if a["stopped_early"]:
        lines += ["", f"**Stopped early:** {a['stopped_early']}"]
    lines += [
        "",
        "## Strategies (measured)",
        "",
        "| strategy | runs | success | mean (s) | p95 (s) | cost ($) | calls | speedup vs seq [95% CI] |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in a["strategies"]:
        lines.append(
            f"| {s['strategy']} | {s['runs']} | {s['success_rate']:.2f} | {s['mean_latency_s']:.1f} | "
            f"{s['p95_latency_s']:.1f} | {s['mean_cost_usd']:.5f} | {s['mean_calls']:.1f} | "
            f"{_fmt(s['speedup'])}x [{_fmt(s['speedup_lo'])}, {_fmt(s['speedup_hi'])}] |"
        )
    lines += [
        "",
        "## Does the cost model predict real latency?",
        "",
        "Latency is noisy and right-skewed, so the forecast is judged as a distribution: aggregate",
        "bias of the predicted mean (ideal 0%) and coverage of the predicted P50 / P95 (ideal 50% / 95%).",
        "Single-run error is shown for reference only; it is dominated by the intrinsic run-to-run",
        "variance of LLM calls rather than by model error.",
        "",
        "| model used for the prediction | runs | aggregate bias | P50 coverage | P95 coverage | single-run abs. error |",
        "|---|---|---|---|---|---|",
        *(
            f"| {label} | {e['n']} | {_fmt(e['aggregate_bias'], '+.1%')} | {_fmt(e['p50_coverage'], '.0%')} | "
            f"{_fmt(e['p95_coverage'], '.0%')} | {_fmt(e['mape'], '.0%')} |"
            for label, e in (
                ("prior (before any observation)", pred["prior_model"]),
                ("calibrated online", pred["calibrated_model"]),
            )
        ),
        "",
        f"Strategy ranking per request (calibrated reps): mean Spearman rho = "
        f"{_fmt(rank['mean_spearman'])}; predicted fastest = actual fastest in "
        f"{_fmt(rank['top1_agreement'], '.0%')} of requests.",
        "",
        "## Calibration",
        "",
        "| parameter | prior | fitted from real calls |",
        "|---|---|---|",
        f"| time to first token + fixed overhead (s) | {cal['prior']['ttft_s']:.2f} | {cal['fitted']['ttft_s']:.2f} |",
        f"| decode throughput (tokens/s) | {cal['prior']['decode_tps']:.0f} | {cal['fitted']['decode_tps']:.0f} |",
        f"| per-call latency error | {_fmt(cal['call_mape_prior'], '.1%')} | {_fmt(cal['call_mape_fitted'], '.1%')} |",
        "",
        f"Measured contention: each additional concurrent call changed latency by "
        f"{_fmt(cal['contention_per_extra_call'], '+.1%')} (slope of latency / predicted vs. concurrent calls).",
        "",
        "## Plans produced by the planner",
        "",
        "| request | tasks | width | depth | planning attempts |",
        "|---|---|---|---|---|",
    ]
    for p in a["plans"]:
        if "error" in p:
            lines.append(f"| {p['request_id']} | planning failed | | | |")
        else:
            lines.append(
                f"| {p['request_id']} | {p['num_tasks']} | {p['max_width']} | {p['depth']} | {p['attempts']} |"
            )
    return "\n".join(lines) + "\n"


def write_live(result: LiveResult, analysis: Mapping[str, Any], out: Path, model: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("runs.csv", result.runs), ("calls.csv", result.calls)):
        if rows:
            with (out / name).open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    (out / "plans.json").write_text(json.dumps(result.plans, indent=2), encoding="utf-8")
    (out / "analysis.json").write_text(
        json.dumps(analysis, indent=2, default=str), encoding="utf-8"
    )
    (out / "LIVE_RESULTS.md").write_text(render_live_markdown(analysis, model), encoding="utf-8")
