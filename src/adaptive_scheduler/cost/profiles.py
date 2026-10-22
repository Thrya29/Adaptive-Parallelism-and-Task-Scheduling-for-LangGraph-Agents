"""Per-task token and success estimates.

Information flows along DAG edges: a task's prompt contains the system prompt, the shared request
context, its own instruction, and the outputs of the tasks it depends on. That is why fan-in tasks
are expensive and why fusing a chain saves input tokens - the shared context is sent once and the
intermediate outputs never leave the call.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from adaptive_scheduler.cost.provider import ProviderProfile
from adaptive_scheduler.graph.model import TaskGraph, TaskType

# Priors for expected output tokens and per-attempt success probability by task type.
OUTPUT_TOKEN_PRIORS: dict[TaskType, float] = {
    TaskType.RETRIEVAL: 300.0,
    TaskType.REASONING: 400.0,
    TaskType.TOOL_CALL: 150.0,
    TaskType.GENERATION: 600.0,
    TaskType.SYNTHESIS: 500.0,
    TaskType.COMPOSITE: 800.0,
}
SUCCESS_PRIORS: dict[TaskType, float] = {
    TaskType.RETRIEVAL: 0.98,
    TaskType.REASONING: 0.96,
    TaskType.TOOL_CALL: 0.97,
    TaskType.GENERATION: 0.97,
    TaskType.SYNTHESIS: 0.97,
    TaskType.COMPOSITE: 0.95,
}


def estimate_tokens(text: str) -> int:
    """Rough token count (~4 characters per token for English); exact counts come from usage."""
    return max(1, math.ceil(len(text) / 4))


@dataclass(frozen=True, slots=True)
class TaskProfile:
    task_type: TaskType
    input_tokens: float
    output_tokens: float
    success_prob: float

    def __post_init__(self) -> None:
        if self.input_tokens < 0 or self.output_tokens < 0:
            raise ValueError("token counts must be >= 0")
        if not 0 < self.success_prob <= 1:
            raise ValueError("success_prob must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class ContextSize:
    system_tokens: float = 400.0
    shared_context_tokens: float = 600.0

    @property
    def per_call(self) -> float:
        return self.system_tokens + self.shared_context_tokens


def build_profiles(
    graph: TaskGraph,
    context: ContextSize,
    output_tokens: Mapping[TaskType, float] = OUTPUT_TOKEN_PRIORS,
    success: Mapping[TaskType, float] = SUCCESS_PRIORS,
    per_task_output: Mapping[str, float] | None = None,
) -> dict[str, TaskProfile]:
    """Profiles for every task; ``per_task_output`` overrides the per-type output estimate."""
    overrides = per_task_output or {}
    profiles: dict[str, TaskProfile] = {}
    for tid in graph.topological_order:
        task = graph.task(tid)
        dep_tokens = sum(profiles[d].output_tokens for d in task.depends_on)
        profiles[tid] = TaskProfile(
            task_type=task.task_type,
            input_tokens=context.per_call + estimate_tokens(task.instruction) + dep_tokens,
            output_tokens=overrides.get(tid, output_tokens[task.task_type]),
            success_prob=success[task.task_type],
        )
    return profiles


def expected_durations(
    profiles: Mapping[str, TaskProfile], provider: ProviderProfile
) -> dict[str, float]:
    return {
        tid: provider.expected_latency(p.input_tokens, p.output_tokens)
        for tid, p in profiles.items()
    }
