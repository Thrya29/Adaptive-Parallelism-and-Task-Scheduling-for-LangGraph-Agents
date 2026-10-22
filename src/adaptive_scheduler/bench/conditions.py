"""Environment conditions and the dependency-blind multi-agent baseline.

A condition fixes the *true* provider behaviour (what the evaluation simulates) and the provider
*beliefs* the selector plans with. They are equal except in ``misestimated``, which tests how the
adaptive scheduler degrades when its cost model is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from adaptive_scheduler.cost.profiles import TaskProfile
from adaptive_scheduler.cost.provider import ProviderProfile
from adaptive_scheduler.graph.model import Task, TaskGraph, TaskType


@dataclass(frozen=True)
class Condition:
    name: str
    description: str
    true_provider: ProviderProfile = field(default_factory=ProviderProfile)
    belief_provider: ProviderProfile | None = None  # None = beliefs are correct
    failure_multiplier: float = 1.0  # scales every task's per-attempt failure probability

    @property
    def beliefs(self) -> ProviderProfile:
        return self.belief_provider or self.true_provider

    def adjust(self, profiles: dict[str, TaskProfile]) -> dict[str, TaskProfile]:
        if self.failure_multiplier == 1.0:
            return profiles
        return {
            t: replace(
                p, success_prob=max(1e-6, 1 - (1 - p.success_prob) * self.failure_multiplier)
            )
            for t, p in profiles.items()
        }


_DEFAULT = ProviderProfile()

CONDITIONS: dict[str, Condition] = {
    c.name: c
    for c in (
        Condition("baseline", "default provider: cap 8, mild contention, 5% heavy tail"),
        Condition(
            "tight_limit",
            "provider allows only 2 concurrent calls; 2 s wasted per rate-limited call",
            true_provider=replace(_DEFAULT, max_concurrency=2, rate_limit_penalty_s=2.0),
        ),
        Condition(
            "heavy_tail",
            "15% of calls are 4x slower and noise is larger",
            true_provider=replace(_DEFAULT, tail_prob=0.15, tail_multiplier=4.0, noise_sigma=0.5),
        ),
        Condition(
            "unreliable",
            "per-attempt failure probability x4",
            failure_multiplier=4.0,
        ),
        Condition(
            "misestimated",
            "true provider is slower and more contended than the selector believes",
            true_provider=replace(_DEFAULT, ttft_s=1.0, decode_tps=60.0, contention=0.08),
            belief_provider=_DEFAULT,
        ),
    )
}


# ---------------------------------------------------------------------------- blind fan-out
@dataclass(frozen=True)
class BlindVariant:
    graph: TaskGraph
    profiles: dict[str, TaskProfile]
    blind_tasks: int  # tasks that needed upstream outputs but ran without them


def blind_variant(
    graph: TaskGraph, profiles: dict[str, TaskProfile], context_tokens: float | None = None
) -> BlindVariant:
    """Dependency-blind multi-agent execution: every task runs at once as an independent agent
    (without the upstream outputs it needed), then one decision agent aggregates all outputs.

    Inputs shrink (no upstream outputs) and an aggregation call is added. The quality cost of
    running blind is *not* an execution failure - retries cannot fix a wrong answer - so it is
    applied separately by the runner as a silent quality check.
    """
    tasks = [t.model_copy(update={"depends_on": ()}) for t in graph.tasks]
    new_profiles: dict[str, TaskProfile] = {}
    for t in graph.tasks:
        p = profiles[t.id]
        upstream = sum(profiles[d].output_tokens for d in t.depends_on)
        new_profiles[t.id] = replace(p, input_tokens=p.input_tokens - upstream)
    context = (
        context_tokens
        if context_tokens is not None
        else min(p.input_tokens for p in new_profiles.values())
    )
    agg = "decision_agent"
    tasks.append(
        Task(
            id=agg,
            instruction="Aggregate all agent outputs into the final answer.",
            task_type=TaskType.SYNTHESIS,
            depends_on=graph.ids,
        )
    )
    new_profiles[agg] = TaskProfile(
        task_type=TaskType.SYNTHESIS,
        input_tokens=context + sum(p.output_tokens for p in new_profiles.values()),
        output_tokens=max(p.output_tokens for p in profiles.values()),
        success_prob=min(p.success_prob for p in profiles.values()),
    )
    blind = sum(1 for t in graph.tasks if t.depends_on)
    return BlindVariant(TaskGraph(tasks=tuple(tasks)), new_profiles, blind)


__all__ = ["CONDITIONS", "BlindVariant", "Condition", "blind_variant"]
