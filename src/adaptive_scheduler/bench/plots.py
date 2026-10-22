"""Static figures for the report (matplotlib, PNG).

Style: one fixed categorical colour per strategy (validated reference palette, in its validated
order), thin marks, recessive grid, one y-axis per chart, a legend on every multi-series chart.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

STRATEGY_COLORS = {
    "adaptive": "#2a78d6",
    "sequential": "#eb6834",
    "static_parallel": "#1baf7a",
    "eager_parallel": "#eda100",
    "blind_parallel": "#e87ba4",
}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COMPARED = ("sequential", "static_parallel", "eager_parallel", "adaptive")


def _style(ax: Any, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=10)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.tick_params(colors=INK_2, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def speedup_by_family(summaries: Sequence[Mapping[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    rows = [s for s in summaries if s["condition"] == "baseline" and s["strategy"] in COMPARED]
    families = sorted(
        {s["family"] for s in rows},
        key=lambda f: -max(s["speedup"] for s in rows if s["family"] == f),
    )
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(families) + 1.6), facecolor=SURFACE)
    offsets = np.linspace(-0.27, 0.27, len(COMPARED))
    for off, strategy in zip(offsets, COMPARED, strict=True):
        pts = {s["family"]: s for s in rows if s["strategy"] == strategy}
        ys = [i + off for i, f in enumerate(families)]
        xs = [pts[f]["speedup"] for f in families]
        err = [
            [pts[f]["speedup"] - pts[f]["speedup_lo"] for f in families],
            [pts[f]["speedup_hi"] - pts[f]["speedup"] for f in families],
        ]
        ax.errorbar(
            xs,
            ys,
            xerr=err,
            fmt="o",
            ms=7,
            color=STRATEGY_COLORS[strategy],
            ecolor=STRATEGY_COLORS[strategy],
            elinewidth=1.5,
            capsize=0,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            label=strategy,
        )
    ax.axvline(1.0, color=INK_2, linewidth=1, linestyle=(0, (3, 3)))
    ax.set_yticks(range(len(families)), families)
    ax.invert_yaxis()
    _style(ax, "Speedup over sequential execution (baseline condition, 95% CI)", "speedup (x)", "")
    ax.legend(frameon=False, loc="lower right", labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def p95_by_condition(pooled: Sequence[Mapping[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    rows = [s for s in pooled if s["strategy"] in COMPARED]
    conditions = sorted({s["condition"] for s in rows})
    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=SURFACE)
    width = 0.8 / len(COMPARED)
    for j, strategy in enumerate(COMPARED):
        vals = {s["condition"]: s["p95_latency_s"] for s in rows if s["strategy"] == strategy}
        xs = np.arange(len(conditions)) + (j - (len(COMPARED) - 1) / 2) * width
        ax.bar(
            xs,
            [vals[c] for c in conditions],
            width=width - 0.02,
            color=STRATEGY_COLORS[strategy],
            edgecolor=SURFACE,
            linewidth=2,
            label=strategy,
        )
    ax.set_xticks(range(len(conditions)), conditions)
    top = max(s["p95_latency_s"] for s in rows)
    ax.set_ylim(0, top * 1.22)  # headroom so the legend never covers a bar
    _style(ax, "P95 latency by environment condition (all workloads pooled)", "", "P95 latency (s)")
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, ncols=len(COMPARED), loc="upper left", labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def latency_cdf(
    runs: Sequence[Mapping[str, Any]],
    path: Path,
    family: str = "layered",
    condition: str = "baseline",
) -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    by: dict[str, list[float]] = defaultdict(list)
    for r in runs:
        if r["family"] == family and r["condition"] == condition and r["strategy"] in COMPARED:
            by[r["strategy"]].append(r["latency_s"])
    if not by:
        return
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    for strategy in COMPARED:
        xs = np.sort(by[strategy])
        ax.step(
            xs,
            np.arange(1, len(xs) + 1) / len(xs),
            where="post",
            linewidth=2,
            color=STRATEGY_COLORS[strategy],
            label=strategy,
        )
    ax.axhline(0.95, color=INK_2, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(ax.get_xlim()[1], 0.95, " P95", va="center", ha="left", color=INK_2, fontsize=9)
    _style(
        ax,
        f"Latency distribution - {family} workloads, {condition} condition",
        "end-to-end latency (s)",
        "fraction of runs",
    )
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, loc="lower right", labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_all(
    analysis: Mapping[str, Any], runs: Sequence[Mapping[str, Any]], out: Path
) -> list[Path]:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    paths = [out / "speedup_by_family.png", out / "p95_by_condition.png", out / "latency_cdf.png"]
    speedup_by_family(analysis["summaries"], paths[0])
    p95_by_condition(analysis["pooled"], paths[1])
    latency_cdf(runs, paths[2])
    return [p for p in paths if p.exists()]
