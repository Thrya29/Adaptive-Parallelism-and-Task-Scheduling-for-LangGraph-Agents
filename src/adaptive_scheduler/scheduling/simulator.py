"""Discrete-event schedulers for a task graph with known (or sampled) durations.

* :func:`list_schedule` - greedy list scheduling on ``k`` workers driven by :class:`ReadyQueue`.
  With ``CRITICAL_PATH`` priority this is HEFT-style scheduling on identical workers; any greedy
  list schedule satisfies Graham's bound ``makespan <= W/k + (1 - 1/k) * S``.
* :func:`sequential_schedule` - one task at a time in topological order (makespan = ``W``).
* :func:`level_parallel_schedule` - naive static fan-out: run each dependency level concurrently
  and wait for the whole level (a barrier) before starting the next.
"""

from __future__ import annotations

import heapq
from enum import StrEnum

from adaptive_scheduler.graph.analysis import Durations, check_durations, upward_rank
from adaptive_scheduler.graph.model import TaskGraph
from adaptive_scheduler.scheduling.ready_queue import ReadyQueue
from adaptive_scheduler.scheduling.schedule import Schedule, ScheduledTask


class PriorityPolicy(StrEnum):
    CRITICAL_PATH = "critical_path"  # highest upward rank first
    FIFO = "fifo"  # topological order


def priorities(graph: TaskGraph, durations: Durations, policy: PriorityPolicy) -> dict[str, float]:
    if policy is PriorityPolicy.CRITICAL_PATH:
        return upward_rank(graph, durations)
    return {tid: -float(i) for tid, i in graph.topological_index.items()}


def list_schedule(
    graph: TaskGraph,
    durations: Durations,
    workers: int,
    policy: PriorityPolicy = PriorityPolicy.CRITICAL_PATH,
    *,
    strategy: str | None = None,
) -> Schedule:
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    check_durations(graph, durations)

    queue = ReadyQueue(graph, priorities(graph, durations, policy))
    free = list(range(workers))  # min-heap: lowest free worker id is reused first
    running: list[tuple[float, int, str, int]] = []  # (finish, seq, task, worker)
    entries: list[ScheduledTask] = []
    now = 0.0

    while True:
        while queue and free:
            tid = queue.pop()
            worker = heapq.heappop(free)
            finish = now + durations[tid]
            entries.append(ScheduledTask(tid, worker, now, finish))
            heapq.heappush(running, (finish, len(entries), tid, worker))
        if not running:
            break
        # Advance to the next completion and retire everything finishing at that instant, so the
        # priority policy chooses among all simultaneously released tasks.
        now = running[0][0]
        while running and running[0][0] <= now:
            _, _, tid, worker = heapq.heappop(running)
            heapq.heappush(free, worker)
            queue.complete(tid)

    return Schedule(strategy or f"list[{policy.value}]", workers, tuple(entries))


def sequential_schedule(graph: TaskGraph, durations: Durations) -> Schedule:
    return list_schedule(graph, durations, 1, PriorityPolicy.FIFO, strategy="sequential")


def level_parallel_schedule(
    graph: TaskGraph, durations: Durations, workers: int | None = None
) -> Schedule:
    """Static level-synchronous fan-out. ``workers=None`` means one worker per task of the widest
    level, i.e. unbounded fan-out."""
    check_durations(graph, durations)
    k = graph.max_width if workers is None else workers
    if k < 1:
        raise ValueError(f"workers must be >= 1, got {k}")

    by_level: dict[int, list[str]] = {}
    for tid in graph.topological_order:
        by_level.setdefault(graph.levels[tid], []).append(tid)

    entries: list[ScheduledTask] = []
    barrier = 0.0
    for level in sorted(by_level):
        slots = [(barrier, w) for w in range(k)]  # (free_at, worker)
        level_end = barrier
        for tid in by_level[level]:
            start, worker = heapq.heappop(slots)
            finish = start + durations[tid]
            entries.append(ScheduledTask(tid, worker, start, finish))
            heapq.heappush(slots, (finish, worker))
            level_end = max(level_end, finish)
        barrier = level_end

    return Schedule("level_parallel", k, tuple(entries))
