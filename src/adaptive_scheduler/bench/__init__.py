"""Benchmark suite: workloads, environment conditions, experiment runner, statistics, report."""

from __future__ import annotations

import csv
import gzip
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from adaptive_scheduler.bench.analysis import analyse, render_markdown
from adaptive_scheduler.bench.runner import PROFILES, BenchConfig, UnitResult, run


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark(
    cfg: BenchConfig,
    out_dir: Path,
    *,
    jobs: int = 1,
    plots: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Run, analyse and write ``runs.csv.gz`` (every run), ``selections.csv``,
    ``blind_sensitivity.csv``, ``analysis.json``, ``RESULTS.md`` and figures into ``out_dir``.
    Returns the analysis dictionary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run(cfg, jobs=jobs, progress=progress)
    # Persist raw results before analysis so no compute is lost if analysis fails.
    _write_csv(out_dir / "runs.csv.gz", result.runs)
    _write_csv(out_dir / "selections.csv", result.selections)
    _write_csv(out_dir / "blind_sensitivity.csv", result.blind)
    analysis = analyse(result, cfg)
    (out_dir / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    markdown = render_markdown(analysis)
    if plots:
        from adaptive_scheduler.bench.plots import write_all  # noqa: PLC0415

        figures = write_all(analysis, result.runs, out_dir)
        if figures:
            markdown += (
                "\n## Figures\n\n" + "\n".join(f"![{p.stem}]({p.name})" for p in figures) + "\n"
            )
    (out_dir / "RESULTS.md").write_text(markdown, encoding="utf-8")
    return analysis


__all__ = [
    "PROFILES",
    "BenchConfig",
    "UnitResult",
    "analyse",
    "render_markdown",
    "run",
    "run_benchmark",
]
