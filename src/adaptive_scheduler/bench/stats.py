"""Statistics for the benchmark: summaries, bootstrap CIs, paired tests, multiple-testing control.

All comparisons are *paired*: strategies are evaluated on the same (workload, seed) draws, so
uncertainty is estimated by resampling those pairs, and significance by the Wilcoxon signed-rank
test on per-pair latency differences, with Holm's step-down correction across comparisons.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

Row = Mapping[str, Any]


def bootstrap_ci(
    values: Sequence[float],
    stat: Callable[[np.ndarray], float] = lambda x: float(np.mean(x)),
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_resamples, len(x)))
    stats = np.array([stat(x[i]) for i in idx])
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def paired_ratio_ci(
    numerator: Sequence[float],
    denominator: Sequence[float],
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Ratio of means ``mean(num) / mean(den)`` with a paired bootstrap CI (e.g. speedup)."""
    a, b = np.asarray(numerator, float), np.asarray(denominator, float)
    if len(a) != len(b) or len(a) == 0:
        raise ValueError("paired samples must be non-empty and of equal length")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_resamples, len(a)))
    ratios = a[idx].mean(axis=1) / b[idx].mean(axis=1)
    point = float(a.mean() / b.mean())
    return point, float(np.quantile(ratios, alpha / 2)), float(np.quantile(ratios, 1 - alpha / 2))


def wilcoxon_p(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sided Wilcoxon signed-rank p-value for paired samples (1.0 if all pairs are equal)."""
    from scipy.stats import wilcoxon  # noqa: PLC0415 - optional 'experiments' dependency

    diff = np.asarray(a, float) - np.asarray(b, float)
    if np.allclose(diff, 0.0):
        return 1.0
    return float(wilcoxon(diff, zero_method="wilcox").pvalue)


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (controls the family-wise error rate)."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


# ---------------------------------------------------------------------------- summaries
@dataclass(frozen=True)
class Summary:
    family: str
    condition: str
    strategy: str
    n: int
    success_rate: float
    mean_latency_s: float
    p50_latency_s: float
    p95_latency_s: float
    p99_latency_s: float
    mean_cost_usd: float
    mean_calls: float
    speedup: float  # mean(sequential latency) / mean(strategy latency), paired
    speedup_lo: float
    speedup_hi: float


def _paired(rows: Iterable[Row]) -> dict[tuple[str, str, str], dict[tuple[str, int], Row]]:
    """(family, condition, strategy) -> {(workload, seed_index): row}."""
    out: dict[tuple[str, str, str], dict[tuple[str, int], Row]] = defaultdict(dict)
    for r in rows:
        out[(r["family"], r["condition"], r["strategy"])][(r["workload"], r["seed_index"])] = r
    return out


def summarize(rows: Sequence[Row], group_family: bool = True) -> list[Summary]:
    """Per (family, condition, strategy) metrics; ``group_family=False`` pools all families."""
    pooled = [dict(r) | ({} if group_family else {"family": "all"}) for r in rows]
    cells = _paired(pooled)
    out = []
    for (family, condition, strategy), runs in sorted(cells.items()):
        keys = sorted(runs)
        lat = np.array([runs[k]["latency_s"] for k in keys])
        seq = cells[(family, condition, "sequential")]
        seq_lat = [seq[k]["latency_s"] for k in keys]
        speed, lo, hi = paired_ratio_ci(seq_lat, lat.tolist())
        out.append(
            Summary(
                family=family,
                condition=condition,
                strategy=strategy,
                n=len(keys),
                success_rate=float(np.mean([runs[k]["success"] for k in keys])),
                mean_latency_s=float(lat.mean()),
                p50_latency_s=float(np.percentile(lat, 50)),
                p95_latency_s=float(np.percentile(lat, 95)),
                p99_latency_s=float(np.percentile(lat, 99)),
                mean_cost_usd=float(np.mean([runs[k]["cost_usd"] for k in keys])),
                mean_calls=float(np.mean([runs[k]["model_calls"] for k in keys])),
                speedup=speed,
                speedup_lo=lo,
                speedup_hi=hi,
            )
        )
    return out


def paired_latencies(
    rows: Sequence[Row], a: str, b: str, *, condition: str | None = None, family: str | None = None
) -> tuple[list[float], list[float]]:
    """Latencies of strategies ``a`` and ``b`` on the same (workload, condition, seed) draws."""
    index: dict[tuple[str, str, int], dict[str, float]] = defaultdict(dict)
    for r in rows:
        if (condition and r["condition"] != condition) or (family and r["family"] != family):
            continue
        if r["strategy"] in (a, b):
            index[(r["workload"], r["condition"], r["seed_index"])][r["strategy"]] = r["latency_s"]
    pairs = [v for v in index.values() if a in v and b in v]
    return [v[a] for v in pairs], [v[b] for v in pairs]
