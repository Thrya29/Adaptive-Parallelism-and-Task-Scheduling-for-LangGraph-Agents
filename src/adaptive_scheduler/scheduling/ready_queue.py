"""Dependency-aware priority queue.

This is the single dispatch primitive of the project: the discrete-event simulator and the live
LangGraph executor both drive the same queue, so what is benchmarked is exactly what runs.
"""

from __future__ import annotations

import heapq
from collections import deque
from collections.abc import Mapping

from adaptive_scheduler.graph.model import TaskGraph


class ReadyQueue:
    """Releases tasks once all dependencies completed; pops the highest-priority ready task.

    Ties are broken by topological position, so dispatch order is fully deterministic.
    Operations are ``O(log n)`` per push/pop.
    """

    def __init__(self, graph: TaskGraph, priority: Mapping[str, float]) -> None:
        missing = [t for t in graph.ids if t not in priority]
        if missing:
            raise ValueError(f"missing priorities for tasks: {missing}")
        self._graph = graph
        self._priority = priority
        self._waiting = {t: len(deps) for t, deps in graph.dependency_map.items()}
        self._heap: list[tuple[float, int, str]] = []
        self._dispatched: set[str] = set()
        self._completed: set[str] = set()
        self._cancelled: set[str] = set()
        for tid in graph.topological_order:
            if self._waiting[tid] == 0:
                self._push(tid)

    def _push(self, tid: str) -> None:
        heapq.heappush(self._heap, (-self._priority[tid], self._graph.topological_index[tid], tid))

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)

    def peek(self) -> str:
        return self._heap[0][2]

    def pop(self) -> str:
        if not self._heap:
            raise IndexError("no ready task")
        tid = heapq.heappop(self._heap)[2]
        self._dispatched.add(tid)
        return tid

    def complete(self, task_id: str) -> list[str]:
        """Mark a dispatched task as done; return the tasks that became ready (in push order)."""
        self._require_dispatched(task_id)
        self._completed.add(task_id)
        released = []
        for nxt in self._graph.successor_map[task_id]:
            self._waiting[nxt] -= 1
            if self._waiting[nxt] == 0 and nxt not in self._cancelled:
                self._push(nxt)
                released.append(nxt)
        return released

    def fail(self, task_id: str) -> list[str]:
        """Mark a dispatched task as failed; cancel and return all of its transitive dependents.

        Independent branches are unaffected, which gives failure isolation with partial results.
        """
        self._require_dispatched(task_id)
        self._completed.add(task_id)
        cancelled = []
        frontier = deque(self._graph.successor_map[task_id])
        while frontier:
            nxt = frontier.popleft()
            if nxt in self._cancelled:
                continue
            self._cancelled.add(nxt)
            cancelled.append(nxt)
            frontier.extend(self._graph.successor_map[nxt])
        return cancelled

    @property
    def finished(self) -> bool:
        return len(self._completed) + len(self._cancelled) == len(self._graph)

    @property
    def cancelled(self) -> frozenset[str]:
        return frozenset(self._cancelled)

    def _require_dispatched(self, task_id: str) -> None:
        if task_id not in self._dispatched:
            raise ValueError(f"task {task_id!r} was not dispatched")
        if task_id in self._completed:
            raise ValueError(f"task {task_id!r} already finished")
