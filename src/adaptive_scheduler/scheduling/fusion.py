"""Task fusion: merge consecutive chain tasks into a single LLM call when it pays off.

Every LLM call pays a fixed overhead ``o`` (network round-trip, time-to-first-token, re-sent
system prompt). For a task with estimated latency ``d`` we model its useful work as
``max(d - o, 0)``. Fusing a segment of ``L >= 2`` consecutive chain tasks into one call costs

    latency(segment) = o + sum(work)
    objective(segment) = latency(segment) + penalty * (L - 1) ** 2

where the convex penalty is a latency-equivalent price for the quality risk of packing more steps
into one prompt. A single task keeps its original latency ``d``. Segments are limited by
``max_tasks_per_call`` and ``max_work_per_call`` (output / timeout limits).

Only *chain* edges ``u -> v`` with ``out_degree(u) == 1`` and ``in_degree(v) == 1`` are fused.
Contracting such edges cannot remove any parallelism and cannot create a cycle, so as long as each
chosen segment is no slower than its parts (guaranteed by optimality and ``penalty >= 0``), both
total work ``W`` and the critical path ``S`` can only decrease. (A greedy list schedule's makespan
can still change non-monotonically - Graham's timing anomalies - which the strategy selector
accounts for by simulating both variants.)

The optimal segmentation of a chain of length ``n`` is found by dynamic programming in
``O(n * max_tasks_per_call)``::

    best[0] = 0
    best[i] = min over feasible L of best[i - L] + objective(chain[i-L : i])
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from adaptive_scheduler.graph.analysis import Durations, check_durations
from adaptive_scheduler.graph.model import Task, TaskGraph, TaskType

_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class FusionParams:
    call_overhead: float
    max_tasks_per_call: int = 4
    max_work_per_call: float = math.inf
    penalty_per_extra_task: float = 0.0

    def __post_init__(self) -> None:
        if not (math.isfinite(self.call_overhead) and self.call_overhead >= 0):
            raise ValueError("call_overhead must be finite and >= 0")
        if self.max_tasks_per_call < 1:
            raise ValueError("max_tasks_per_call must be >= 1")
        if self.max_work_per_call <= 0:
            raise ValueError("max_work_per_call must be > 0")
        if self.penalty_per_extra_task < 0:
            raise ValueError("penalty_per_extra_task must be >= 0")


def fused_latency(durations: Sequence[float], params: FusionParams) -> float:
    if len(durations) == 1:
        return durations[0]
    o = params.call_overhead
    return o + math.fsum(max(d - o, 0.0) for d in durations)


def segment_feasible(durations: Sequence[float], params: FusionParams) -> bool:
    if len(durations) == 1:
        return True
    o = params.call_overhead
    work = math.fsum(max(d - o, 0.0) for d in durations)
    return len(durations) <= params.max_tasks_per_call and work <= params.max_work_per_call


def segment_objective(durations: Sequence[float], params: FusionParams) -> float:
    extra = len(durations) - 1
    return fused_latency(durations, params) + params.penalty_per_extra_task * extra * extra


@dataclass(frozen=True, slots=True)
class ChainPlan:
    segments: tuple[tuple[int, int], ...]  # half-open [start, end) index ranges
    objective: float


def optimal_chain_segmentation(durations: Sequence[float], params: FusionParams) -> ChainPlan:
    """Minimum-objective partition of a chain into contiguous feasible segments.

    Ties prefer shorter final segments, i.e. fuse only when it strictly helps.
    """
    n = len(durations)
    best = [0.0] + [math.inf] * n
    choice = [0] * (n + 1)
    o = params.call_overhead
    for i in range(1, n + 1):
        work = 0.0
        for length in range(1, min(params.max_tasks_per_call, i) + 1):
            j = i - length
            work += max(durations[j] - o, 0.0)
            if length > 1 and work > params.max_work_per_call:
                break  # work only grows with longer segments
            cost = best[j] + segment_objective(durations[j:i], params)
            if cost < best[i] - _EPS:
                best[i], choice[i] = cost, length
    segments = []
    i = n
    while i > 0:
        segments.append((i - choice[i], i))
        i -= choice[i]
    return ChainPlan(tuple(reversed(segments)), best[n])


def maximal_chains(graph: TaskGraph) -> list[tuple[str, ...]]:
    """Partition the tasks into maximal chains of fusible edges (singletons included)."""
    preds, succ = graph.dependency_map, graph.successor_map

    def continues_chain(tid: str) -> bool:
        return len(preds[tid]) == 1 and len(succ[preds[tid][0]]) == 1

    chains = []
    for head in graph.topological_order:
        if continues_chain(head):
            continue
        chain = [head]
        while len(succ[chain[-1]]) == 1 and len(preds[succ[chain[-1]][0]]) == 1:
            chain.append(succ[chain[-1]][0])
        chains.append(tuple(chain))
    return chains


@dataclass(frozen=True)
class FusionResult:
    graph: TaskGraph
    durations: dict[str, float]
    groups: dict[str, tuple[str, ...]]  # new task id -> original task ids

    @property
    def calls_saved(self) -> int:
        return sum(len(members) - 1 for members in self.groups.values())


def fuse_chains(graph: TaskGraph, durations: Durations, params: FusionParams) -> FusionResult:
    check_durations(graph, durations)
    segments: list[tuple[str, ...]] = []
    for chain in maximal_chains(graph):
        plan = optimal_chain_segmentation([durations[t] for t in chain], params)
        segments.extend(chain[a:b] for a, b in plan.segments)

    group_of: dict[str, str] = {}
    groups: dict[str, tuple[str, ...]] = {}
    for members in segments:
        gid = members[0] if len(members) == 1 else f"fused:{members[0]}..{members[-1]}"
        groups[gid] = members
        for tid in members:
            group_of[tid] = gid

    tasks = []
    new_durations = {}
    # Keep the original topological position of each segment's first task.
    for gid, members in sorted(groups.items(), key=lambda g: graph.topological_index[g[1][0]]):
        # Only a segment's first task can have dependencies outside the segment.
        deps = tuple(dict.fromkeys(group_of[d] for d in graph.dependency_map[members[0]]))
        if len(members) == 1:
            tasks.append(graph.task(gid).model_copy(update={"depends_on": deps}))
        else:
            originals = [graph.task(t) for t in members]
            types = {t.task_type for t in originals}
            tasks.append(
                Task(
                    id=gid,
                    instruction="\n".join(
                        f"Step {i}/{len(members)} ({t.id}): {t.instruction}"
                        for i, t in enumerate(originals, 1)
                    ),
                    task_type=types.pop() if len(types) == 1 else TaskType.COMPOSITE,
                    depends_on=deps,
                    fused_from=members,
                )
            )
        new_durations[gid] = fused_latency([durations[t] for t in members], params)

    return FusionResult(TaskGraph(tasks=tuple(tasks)), new_durations, groups)
