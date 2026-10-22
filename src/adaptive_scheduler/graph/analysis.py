"""Critical-path analysis and graph profiling.

Notation used throughout the project:

* ``d(v)``  - estimated duration of task ``v``
* ``W``     - total work, ``sum d(v)``; equals the sequential makespan
* ``S``     - span, the length of the critical (longest) path
* ``W / S`` - available parallelism; ~1 means the plan is inherently sequential
* ``max(S, W / k)`` - lower bound on the makespan of any schedule on ``k`` workers
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from dataclasses import dataclass

from adaptive_scheduler.graph.model import TaskGraph

Durations = Mapping[str, float]


def check_durations(graph: TaskGraph, durations: Durations) -> None:
    missing = [tid for tid in graph.ids if tid not in durations]
    if missing:
        raise ValueError(f"missing durations for tasks: {missing}")
    bad = [tid for tid in graph.ids if not (math.isfinite(durations[tid]) and durations[tid] >= 0)]
    if bad:
        raise ValueError(f"durations must be finite and non-negative: {bad}")


def earliest_finish(graph: TaskGraph, durations: Durations) -> dict[str, float]:
    """Earliest finish time of each task with unlimited workers (forward pass)."""
    check_durations(graph, durations)
    finish: dict[str, float] = {}
    for tid in graph.topological_order:
        start = max((finish[d] for d in graph.dependency_map[tid]), default=0.0)
        finish[tid] = start + durations[tid]
    return finish


def upward_rank(graph: TaskGraph, durations: Durations) -> dict[str, float]:
    """HEFT upward rank: ``rank(v) = d(v) + max(rank(s) for s in successors(v))``.

    The rank is the length of the longest path from ``v`` to any exit, so scheduling the ready task
    with the highest rank first prioritizes the critical path.
    """
    check_durations(graph, durations)
    rank: dict[str, float] = {}
    for tid in reversed(graph.topological_order):
        rank[tid] = durations[tid] + max((rank[s] for s in graph.successor_map[tid]), default=0.0)
    return rank


@dataclass(frozen=True, slots=True)
class CriticalPath:
    length: float
    tasks: tuple[str, ...]


def critical_path(graph: TaskGraph, durations: Durations) -> CriticalPath:
    finish = earliest_finish(graph, durations)
    order = graph.topological_order
    # Ties are broken towards the earliest task in topological order for determinism.
    last = max(order, key=lambda t: (finish[t], -graph.topological_index[t]))
    path = [last]
    while preds := graph.dependency_map[path[-1]]:
        path.append(max(preds, key=lambda t: (finish[t], -graph.topological_index[t])))
    return CriticalPath(length=finish[last], tasks=tuple(reversed(path)))


def slack(graph: TaskGraph, durations: Durations) -> dict[str, float]:
    """How long each task can be delayed without extending the critical path (0 = critical)."""
    finish = earliest_finish(graph, durations)
    rank = upward_rank(graph, durations)
    span = max(finish.values())
    return {t: max(0.0, span - (finish[t] - durations[t]) - rank[t]) for t in graph.ids}


@dataclass(frozen=True, slots=True)
class GraphProfile:
    """Structural and cost features of a task graph; also the feature vector for strategy choice."""

    num_tasks: int
    num_edges: int
    depth: int
    max_width: int
    total_work: float
    span: float
    critical_path: tuple[str, ...]
    duration_cv: float  # coefficient of variation of task durations (cost heterogeneity)

    @property
    def parallelism(self) -> float:
        return self.total_work / self.span if self.span > 0 else 1.0

    @property
    def edge_density(self) -> float:
        n = self.num_tasks
        return self.num_edges / (n * (n - 1) / 2) if n > 1 else 0.0

    def lower_bound(self, workers: int) -> float:
        """No schedule on ``workers`` workers can finish faster than this."""
        _check_workers(workers)
        return max(self.span, self.total_work / workers)

    def graham_bound(self, workers: int) -> float:
        """Upper bound for any greedy list schedule: ``W/k + (1 - 1/k) * S`` (Graham, 1969)."""
        _check_workers(workers)
        return self.total_work / workers + (1 - 1 / workers) * self.span


def profile(graph: TaskGraph, durations: Durations) -> GraphProfile:
    cp = critical_path(graph, durations)
    values = [durations[t] for t in graph.ids]
    mean = statistics.fmean(values)
    cv = statistics.pstdev(values) / mean if mean > 0 else 0.0
    return GraphProfile(
        num_tasks=len(graph),
        num_edges=graph.num_edges,
        depth=graph.depth,
        max_width=graph.max_width,
        total_work=math.fsum(values),
        span=cp.length,
        critical_path=cp.tasks,
        duration_cv=cv,
    )


def _check_workers(workers: int) -> None:
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
