"""Turn raw benchmark rows into hypothesis verdicts, tables and a Markdown report."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

import numpy as np

from adaptive_scheduler.bench.runner import BenchConfig, UnitResult
from adaptive_scheduler.bench.stats import (
    Summary,
    holm,
    paired_latencies,
    summarize,
    wilcoxon_p,
)

BASELINE_STRATEGIES = ("sequential", "static_parallel", "eager_parallel")
SUCCESS_TOLERANCE = 0.02


def _cell_map(summaries: Sequence[Summary]) -> dict[tuple[str, str], dict[str, Summary]]:
    cells: dict[tuple[str, str], dict[str, Summary]] = defaultdict(dict)
    for s in summaries:
        cells[(s.family, s.condition)][s.strategy] = s
    return cells


def _score(s: Summary, cfg: BenchConfig) -> float:
    return cfg.objective.score(s.mean_latency_s, s.p95_latency_s, s.mean_cost_usd)


def analyse(result: UnitResult, cfg: BenchConfig) -> dict[str, Any]:
    rows = result.runs
    summaries = summarize(rows)
    cells = _cell_map(summaries)

    # H1 - adaptive vs the best fixed strategy per (family, condition), by the selector objective.
    h1_cells = []
    for (family, condition), by in sorted(cells.items()):
        ad = by["adaptive"]
        eligible = [
            by[s]
            for s in BASELINE_STRATEGIES
            if by[s].success_rate >= ad.success_rate - SUCCESS_TOLERANCE
        ]
        # If no baseline matches adaptive's success rate, adaptive is strictly more reliable in
        # this cell; compare latency/cost against all baselines and flag it.
        more_reliable = not eligible
        best = min(eligible or [by[s] for s in BASELINE_STRATEGIES], key=lambda s: _score(s, cfg))
        h1_cells.append(
            {
                "family": family,
                "condition": condition,
                "best_fixed": best.strategy + (" (less reliable)" if more_reliable else ""),
                "adaptive_score": _score(ad, cfg),
                "best_fixed_score": _score(best, cfg),
                "relative": _score(ad, cfg) / _score(best, cfg) - 1,
                "adaptive_latency_s": ad.mean_latency_s,
                "best_fixed_latency_s": best.mean_latency_s,
            }
        )
    rel = np.array([c["relative"] for c in h1_cells])

    # Paired significance: adaptive vs each baseline, per condition, Holm-corrected.
    tests: list[dict[str, Any]] = []
    for condition in cfg.conditions:
        for other in (*BASELINE_STRATEGIES, "fixed_k_cap", "adaptive_fifo", "adaptive_nofusion"):
            a, b = paired_latencies(rows, "adaptive", other, condition=condition)
            if a:
                tests.append(
                    {
                        "condition": condition,
                        "vs": other,
                        "pairs": len(a),
                        "adaptive_mean_s": float(np.mean(a)),
                        "other_mean_s": float(np.mean(b)),
                        "p_value": wilcoxon_p(a, b),
                    }
                )
    for t, adj in zip(tests, holm([t["p_value"] for t in tests]), strict=True):
        t["p_holm"] = adj

    # H2 - dependency-blind fan-out: success collapses on dependent plans; sensitivity to quality.
    blind_by_family: dict[str, list[float]] = defaultdict(list)
    seq_by_family: dict[str, list[float]] = defaultdict(list)
    for s in summaries:
        if s.condition == "baseline":
            if s.strategy == "blind_parallel":
                blind_by_family[s.family].append(s.success_rate)
            if s.strategy == "sequential":
                seq_by_family[s.family].append(s.success_rate)
    sensitivity: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in result.blind:
        if row["condition"] == "baseline":
            sensitivity[(row["family"], row["quality"])].append(row["expected_success"])

    # H3 - critical-path vs FIFO priority (same selector otherwise), and H4 - fusion.
    def ablation(other: str) -> dict[str, Any]:
        a, b = paired_latencies(rows, "adaptive", other)
        ad = [r for r in rows if r["strategy"] == "adaptive"]
        ot = [r for r in rows if r["strategy"] == other]
        return {
            "pairs": len(a),
            "adaptive_mean_s": float(np.mean(a)),
            "other_mean_s": float(np.mean(b)),
            "adaptive_p95_s": float(np.percentile(a, 95)),
            "other_p95_s": float(np.percentile(b, 95)),
            "adaptive_calls": float(np.mean([r["model_calls"] for r in ad])),
            "other_calls": float(np.mean([r["model_calls"] for r in ot])),
            "adaptive_cost": float(np.mean([r["cost_usd"] for r in ad])),
            "other_cost": float(np.mean([r["cost_usd"] for r in ot])),
            "adaptive_success": float(np.mean([r["success"] for r in ad])),
            "other_success": float(np.mean([r["success"] for r in ot])),
            "p_value": wilcoxon_p(a, b),
        }

    sel = [s for s in result.selections if s["variant"] == "adaptive"]
    selection = {
        "units": len(sel),
        "exact_rate": float(np.mean([s["exact"] for s in sel])),
        "within_1pct_rate": float(np.mean([s["regret"] <= 0.01 for s in sel])),
        "within_5pct_rate": float(np.mean([s["regret"] <= 0.05 for s in sel])),
        "mean_regret": float(np.mean([s["regret"] for s in sel])),
        "max_regret": float(np.max([s["regret"] for s in sel])),
        "mean_selection_ms": 1000 * float(np.mean([s["selection_time_s"] for s in sel])),
        "prediction_error_by_condition": {
            c: float(
                np.mean(
                    [
                        s["predicted_mean_s"] / s["realized_mean_s"] - 1
                        for s in sel
                        if s["condition"] == c
                    ]
                )
            )
            for c in cfg.conditions
        },
    }

    return {
        "config": {k: v for k, v in asdict(cfg).items() if k != "objective"}
        | {"objective": asdict(cfg.objective)},
        "summaries": [asdict(s) for s in summaries],
        "pooled": [asdict(s) for s in summarize(rows, group_family=False)],
        "h1": {
            "cells": h1_cells,
            "within_5pct_rate": float(np.mean(rel <= 0.05)),
            "better_or_equal_rate": float(np.mean(rel <= 0.0)),
            "median_relative": float(np.median(rel)),
            "worst_relative": float(rel.max()),
        },
        "tests": tests,
        "h2": {
            "blind_success": {f: float(np.mean(v)) for f, v in sorted(blind_by_family.items())},
            "sequential_success": {f: float(np.mean(v)) for f, v in sorted(seq_by_family.items())},
            "sensitivity": {
                f"{f}@{q}": float(np.mean(v)) for (f, q), v in sorted(sensitivity.items())
            },
        },
        "h3": ablation("adaptive_fifo"),
        "h4": ablation("adaptive_nofusion"),
        "no_cost_model": ablation("fixed_k_cap"),
        "selection": selection,
    }


# ---------------------------------------------------------------------------- markdown
def _fmt_p(p: float) -> str:
    return "<0.001" if p < 1e-3 else f"{p:.3f}"


def render_markdown(analysis: Mapping[str, Any]) -> str:
    cfg = analysis["config"]
    lines = [
        "# Benchmark results",
        "",
        "Generated by `apsched bench`. Simulation with the calibrated provider model; "
        "latencies are",
        "model seconds (selector overhead included for adaptive strategies). "
        f"{cfg['eval_runs']} evaluation seeds per (workload, condition, strategy); "
        f"selector uses {cfg['selector_samples']} Monte Carlo samples per candidate "
        "on separate seeds. "
        f"Blind-agent quality q = {cfg['blind_quality']}.",
        "",
        "## Pooled over all workload families",
        "",
        "| condition | strategy | success | mean (s) | p95 (s) | cost ($) | calls "
        "| speedup vs seq [95% CI] |",
        "|---|---|---|---|---|---|---|---|",
    ]
    order = [
        "sequential",
        "static_parallel",
        "eager_parallel",
        "blind_parallel",
        "fixed_k_cap",
        "adaptive_fifo",
        "adaptive_nofusion",
        "adaptive",
    ]
    pooled = sorted(analysis["pooled"], key=lambda s: (s["condition"], order.index(s["strategy"])))
    for s in pooled:
        name = f"**{s['strategy']}**" if s["strategy"] == "adaptive" else s["strategy"]
        lines.append(
            f"| {s['condition']} | {name} | {s['success_rate']:.3f} | {s['mean_latency_s']:.1f} | "
            f"{s['p95_latency_s']:.1f} | {s['mean_cost_usd']:.4f} | {s['mean_calls']:.1f} | "
            f"{s['speedup']:.2f}x [{s['speedup_lo']:.2f}, {s['speedup_hi']:.2f}] |"
        )

    h1 = analysis["h1"]
    lines += [
        "",
        "## H1 - adaptive vs the best fixed strategy (per family x condition)",
        "",
        f"Objective J = mean + {cfg['objective']['alpha_p95']}·P95 + "
        f"{cfg['objective']['usd_weight']}·cost; "
        f"fixed strategies must match adaptive's success rate within {SUCCESS_TOLERANCE:.0%}.",
        "",
        "- adaptive is within 5% of the best fixed strategy in "
        f"**{h1['within_5pct_rate']:.0%}** of cells, "
        f"and better or equal in **{h1['better_or_equal_rate']:.0%}**",
        f"- median relative J: {h1['median_relative']:+.1%}; "
        f"worst cell: {h1['worst_relative']:+.1%}",
        "",
        "| family | condition | best fixed | adaptive J | best fixed J | relative |",
        "|---|---|---|---|---|---|",
    ]
    for c in h1["cells"]:
        lines.append(
            f"| {c['family']} | {c['condition']} | {c['best_fixed']} | {c['adaptive_score']:.1f} | "
            f"{c['best_fixed_score']:.1f} | {c['relative']:+.1%} |"
        )

    lines += [
        "",
        "### Paired Wilcoxon tests on latency (adaptive vs X, Holm-corrected)",
        "",
        "| condition | vs | pairs | adaptive mean (s) | other mean (s) | p (Holm) |",
        "|---|---|---|---|---|---|",
    ]
    for t in analysis["tests"]:
        lines.append(
            f"| {t['condition']} | {t['vs']} | {t['pairs']} | {t['adaptive_mean_s']:.1f} | "
            f"{t['other_mean_s']:.1f} | {_fmt_p(t['p_holm'])} |"
        )

    h2 = analysis["h2"]
    lines += [
        "",
        "## H2 - dependency-blind multi-agent fan-out (baseline condition)",
        "",
        f"| family | sequential success | blind success (q = {cfg['blind_quality']}) |",
        "|---|---|---|",
    ]
    for f, v in h2["blind_success"].items():
        lines.append(f"| {f} | {h2['sequential_success'][f]:.3f} | {v:.3f} |")
    families = sorted({k.split("@")[0] for k in h2["sensitivity"]})
    qualities = sorted({float(k.split("@")[1]) for k in h2["sensitivity"]})
    lines += [
        "",
        "Sensitivity of blind success to the per-task quality q (expected success):",
        "",
        "| family | " + " | ".join(f"q={q:g}" for q in qualities) + " |",
        "|---|" + "---|" * len(qualities),
    ]
    for f in families:
        cells = " | ".join(f"{h2['sensitivity'][f'{f}@{q}']:.3f}" for q in qualities)
        lines.append(f"| {f} | {cells} |")

    for key, title, other in (
        ("h3", "H3 - critical-path priority vs FIFO (ablation)", "FIFO"),
        ("h4", "H4 - task fusion (ablation: no fusion)", "no fusion"),
        ("no_cost_model", "Ablation - no cost model (k = provider cap)", "k = cap"),
    ):
        a = analysis[key]
        lines += [
            "",
            f"## {title}",
            "",
            f"| metric | adaptive | {other} |",
            "|---|---|---|",
            f"| mean latency (s) | {a['adaptive_mean_s']:.2f} | {a['other_mean_s']:.2f} |",
            f"| p95 latency (s) | {a['adaptive_p95_s']:.2f} | {a['other_p95_s']:.2f} |",
            f"| model calls | {a['adaptive_calls']:.2f} | {a['other_calls']:.2f} |",
            f"| cost ($) | {a['adaptive_cost']:.4f} | {a['other_cost']:.4f} |",
            f"| success | {a['adaptive_success']:.3f} | {a['other_success']:.3f} |",
            "",
            f"Paired Wilcoxon on latency: p = {_fmt_p(a['p_value'])} over {a['pairs']} pairs.",
        ]

    s = analysis["selection"]
    lines += [
        "",
        "## Selector accuracy (hindsight)",
        "",
        f"Over {s['units']} (workload, condition) units the selector picked the configuration "
        f"that was best on the oracle sample ({cfg['oracle_runs']} runs per candidate) in "
        f"**{s['exact_rate']:.0%}** of units; "
        "its choice was within 1% "
        f"of the best in {s['within_1pct_rate']:.0%} and within 5% in {s['within_5pct_rate']:.0%} "
        f"(mean regret {s['mean_regret']:.2%}, max {s['max_regret']:.1%}). Mean selection time "
        f"{s['mean_selection_ms']:.0f} ms.",
        "",
        "Mean prediction error of the chosen configuration's latency (predicted / realized - 1):",
        "",
        "| condition | error |",
        "|---|---|",
        *(f"| {c} | {e:+.1%} |" for c, e in s["prediction_error_by_condition"].items()),
    ]
    return "\n".join(lines) + "\n"
