"""Task and task-graph data model.

A :class:`TaskGraph` is immutable and structurally valid by construction: ids are unique, every
dependency exists, and the graph is acyclic. Invalid plans raise ``pydantic.ValidationError`` with
a precise message (e.g. the exact cycle), which the planner feeds back to the LLM for repair.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from enum import StrEnum
from functools import cached_property

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from adaptive_scheduler.graph import algorithms

TASK_ID_PATTERN = r"^[A-Za-z0-9_][A-Za-z0-9_.:\-]{0,255}$"


class TaskType(StrEnum):
    RETRIEVAL = "retrieval"
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    GENERATION = "generation"
    SYNTHESIS = "synthesis"
    COMPOSITE = "composite"  # produced by task fusion


class Task(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=TASK_ID_PATTERN)
    instruction: str = ""
    task_type: TaskType = TaskType.REASONING
    depends_on: tuple[str, ...] = ()
    fused_from: tuple[str, ...] = Field(
        default=(), description="Original task ids when this task is the fusion of several."
    )

    @field_validator("depends_on")
    @classmethod
    def _no_duplicate_dependencies(cls, deps: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(deps)) != len(deps):
            raise ValueError(f"duplicate dependencies: {list(deps)}")
        return deps


class PlanLimitError(ValueError):
    """Raised when a plan exceeds configured size limits (guardrail against runaway plans)."""


class TaskGraph(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tasks: tuple[Task, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_structure(self) -> TaskGraph:
        counts = Counter(t.id for t in self.tasks)
        dupes = sorted(i for i, c in counts.items() if c > 1)
        if dupes:
            raise ValueError(f"duplicate task ids: {dupes}")
        for task in self.tasks:
            for dep in task.depends_on:
                if dep == task.id:
                    raise ValueError(f"task {task.id!r} depends on itself")
                if dep not in counts:
                    raise ValueError(f"task {task.id!r} depends on unknown task {dep!r}")
        cycle = algorithms.find_cycle(self.dependency_map)
        if cycle:
            raise ValueError("dependency cycle: " + " -> ".join(cycle))
        return self

    # ------------------------------------------------------------------ construction helpers
    @classmethod
    def from_dependencies(
        cls,
        deps: Mapping[str, Sequence[str]],
        task_type: TaskType = TaskType.REASONING,
    ) -> TaskGraph:
        """Build a graph from ``{task_id: [dependency ids]}`` (mainly for tests and tooling)."""
        return cls(
            tasks=tuple(
                Task(id=tid, instruction=tid, task_type=task_type, depends_on=tuple(d))
                for tid, d in deps.items()
            )
        )

    # ------------------------------------------------------------------ structure
    def __len__(self) -> int:
        return len(self.tasks)

    @cached_property
    def ids(self) -> tuple[str, ...]:
        return tuple(t.id for t in self.tasks)

    @cached_property
    def _by_id(self) -> dict[str, Task]:
        return {t.id: t for t in self.tasks}

    def task(self, task_id: str) -> Task:
        return self._by_id[task_id]

    @cached_property
    def dependency_map(self) -> dict[str, tuple[str, ...]]:
        return {t.id: t.depends_on for t in self.tasks}

    @cached_property
    def successor_map(self) -> dict[str, tuple[str, ...]]:
        return {k: tuple(v) for k, v in algorithms.successors_of(self.dependency_map).items()}

    @cached_property
    def topological_order(self) -> tuple[str, ...]:
        return tuple(algorithms.topological_order(self.dependency_map))

    @cached_property
    def topological_index(self) -> dict[str, int]:
        return {tid: i for i, tid in enumerate(self.topological_order)}

    @cached_property
    def levels(self) -> dict[str, int]:
        return algorithms.levels(self.dependency_map)

    @property
    def num_edges(self) -> int:
        return sum(len(t.depends_on) for t in self.tasks)

    @property
    def depth(self) -> int:
        return 1 + max(self.levels.values())

    @property
    def max_width(self) -> int:
        return max(Counter(self.levels.values()).values())

    # ------------------------------------------------------------------ transformations
    def transitive_reduction(self) -> TaskGraph:
        reduced = algorithms.transitive_reduction(self.dependency_map)
        return TaskGraph(
            tasks=tuple(t.model_copy(update={"depends_on": reduced[t.id]}) for t in self.tasks)
        )

    def check_limits(self, *, max_tasks: int, max_depth: int, max_fan_in: int) -> None:
        if len(self) > max_tasks:
            raise PlanLimitError(f"plan has {len(self)} tasks; limit is {max_tasks}")
        if self.depth > max_depth:
            raise PlanLimitError(f"plan depth is {self.depth}; limit is {max_depth}")
        widest = max(len(t.depends_on) for t in self.tasks)
        if widest > max_fan_in:
            raise PlanLimitError(f"a task has {widest} dependencies; limit is {max_fan_in}")
