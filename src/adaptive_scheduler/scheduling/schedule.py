"""Schedule representation and an independent validity checker."""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property

from adaptive_scheduler.graph.analysis import Durations
from adaptive_scheduler.graph.model import TaskGraph


@dataclass(frozen=True, slots=True)
class ScheduledTask:
    task_id: str
    worker: int
    start: float
    finish: float


@dataclass(frozen=True)
class Schedule:
    strategy: str
    workers: int
    entries: tuple[ScheduledTask, ...]

    @property
    def makespan(self) -> float:
        return max((e.finish for e in self.entries), default=0.0)

    @cached_property
    def by_task(self) -> dict[str, ScheduledTask]:
        return {e.task_id: e for e in self.entries}

    @property
    def dispatch_order(self) -> tuple[str, ...]:
        return tuple(e.task_id for e in self.entries)


class InvalidScheduleError(ValueError):
    pass


def validate_schedule(
    schedule: Schedule, graph: TaskGraph, durations: Durations, *, tol: float = 1e-9
) -> None:
    """Check every invariant a schedule must satisfy; raise :class:`InvalidScheduleError` if not.

    Deliberately written independently of the schedulers so tests do not trust the code under test.
    """
    ids = [e.task_id for e in schedule.entries]
    if sorted(ids) != sorted(graph.ids):
        raise InvalidScheduleError("schedule must contain every task exactly once")

    entry = schedule.by_task
    for e in schedule.entries:
        if not 0 <= e.worker < schedule.workers:
            raise InvalidScheduleError(f"{e.task_id}: worker {e.worker} out of range")
        if e.start < -tol:
            raise InvalidScheduleError(f"{e.task_id}: negative start time")
        if abs((e.finish - e.start) - durations[e.task_id]) > tol * max(1.0, e.finish):
            raise InvalidScheduleError(f"{e.task_id}: finish - start != duration")
        for dep in graph.dependency_map[e.task_id]:
            if e.start < entry[dep].finish - tol:
                raise InvalidScheduleError(f"{e.task_id} starts before dependency {dep} finishes")

    per_worker: dict[int, list[ScheduledTask]] = defaultdict(list)
    for e in schedule.entries:
        per_worker[e.worker].append(e)
    for worker, items in per_worker.items():
        items.sort(key=lambda x: (x.start, x.finish))
        for prev, cur in itertools.pairwise(items):
            if cur.start < prev.finish - tol:
                raise InvalidScheduleError(
                    f"worker {worker} runs {prev.task_id} and {cur.task_id} concurrently"
                )
