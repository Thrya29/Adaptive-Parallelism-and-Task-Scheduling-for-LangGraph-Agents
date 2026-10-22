"""Stochastic discrete-event simulation of executing a task graph against an LLM provider.

Two nested concurrency limits are modelled:

1. **Client slots** (``ExecutionConfig.workers``): how many tasks the executor runs at once. A
   task holds its slot across retries, exactly like an ``asyncio.Semaphore``-guarded worker.
2. **Provider slots** (``ProviderProfile.max_concurrency``): calls beyond it wait FIFO and pay a
   rate-limit penalty (the wasted back-off after an HTTP 429).

Per-call latency is sampled at call start, using the number of calls active at the provider for
contention. Failed attempts are retried with jittered exponential back-off; a task that exhausts
its retries fails and its transitive dependents are cancelled (independent branches continue).

Dispatch uses the same :class:`ReadyQueue` as the live executor. With the noise, tail, contention,
failures and provider limit switched off, a run reproduces the deterministic schedulers of
:mod:`adaptive_scheduler.scheduling.simulator` exactly (tested).
"""

from __future__ import annotations

import heapq
import math
from collections import Counter, deque
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np

from adaptive_scheduler.cost.profiles import TaskProfile, expected_durations
from adaptive_scheduler.cost.provider import ProviderProfile
from adaptive_scheduler.graph.model import TaskGraph
from adaptive_scheduler.scheduling.fusion import FusionParams, fuse_chains
from adaptive_scheduler.scheduling.ready_queue import ReadyQueue
from adaptive_scheduler.scheduling.simulator import PriorityPolicy, priorities


class Mode(StrEnum):
    LIST = "list"  # greedy: start any ready task as soon as a slot frees up
    LEVEL_SYNC = "level_sync"  # static fan-out: barrier between dependency levels


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_retries: int = 2
    backoff_base_s: float = 0.5
    backoff_cap_s: float = 8.0

    def backoff(self, attempt: int, rng: np.random.Generator) -> float:
        """Exponential back-off with full jitter in ``[0.5, 1.5)`` of the nominal delay."""
        nominal = min(self.backoff_cap_s, self.backoff_base_s * 2.0**attempt)
        return nominal * (0.5 + float(rng.random()))


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    name: str
    workers: int | None = None  # None = unbounded fan-out
    policy: PriorityPolicy = PriorityPolicy.CRITICAL_PATH
    mode: Mode = Mode.LIST
    fuse: bool = False
    fusion_max_tasks: int = 4
    fusion_penalty_s: float = 0.25

    def __post_init__(self) -> None:
        if self.workers is not None and self.workers < 1:
            raise ValueError("workers must be >= 1 or None")


@dataclass(frozen=True, slots=True)
class RunResult:
    makespan: float
    model_calls: int
    rate_limited_calls: int
    input_tokens: float
    output_tokens: float
    cost_usd: float
    completed: int
    failed: int
    cancelled: int
    num_tasks: int  # tasks actually executed (after fusion)

    @property
    def success(self) -> bool:
        return self.failed == 0 and self.cancelled == 0


@dataclass(frozen=True)
class PreparedPlan:
    """A graph ready to execute under a config: fused if requested, with priorities computed."""

    graph: TaskGraph
    profiles: dict[str, TaskProfile]
    priority: dict[str, float]
    groups: dict[str, tuple[str, ...]]


def fuse_profiles(
    graph: TaskGraph,
    profiles: Mapping[str, TaskProfile],
    provider: ProviderProfile,
    config: ExecutionConfig,
    per_call_context_tokens: float,
) -> tuple[TaskGraph, dict[str, TaskProfile], dict[str, tuple[str, ...]]]:
    durations = expected_durations(profiles, provider)
    overhead = provider.expected_latency(per_call_context_tokens, 0.0)
    params = FusionParams(
        call_overhead=overhead,
        max_tasks_per_call=config.fusion_max_tasks,
        penalty_per_extra_task=config.fusion_penalty_s,
    )
    fused = fuse_chains(graph, durations, params)
    new_profiles: dict[str, TaskProfile] = {}
    for gid, members in fused.groups.items():
        if len(members) == 1:
            new_profiles[gid] = profiles[members[0]]
            continue
        ps = [profiles[m] for m in members]
        # Context is sent once; outputs of earlier members stay inside the call.
        internal = sum(p.output_tokens for p in ps[:-1])
        input_tokens = sum(p.input_tokens for p in ps) - (len(ps) - 1) * per_call_context_tokens
        success = math.prod(p.success_prob for p in ps) * provider.fusion_quality_decay ** (
            len(ps) - 1
        )
        new_profiles[gid] = TaskProfile(
            task_type=fused.graph.task(gid).task_type,
            input_tokens=max(input_tokens - internal, per_call_context_tokens),
            output_tokens=sum(p.output_tokens for p in ps),
            success_prob=success,
        )
    return fused.graph, new_profiles, fused.groups


def prepare(
    graph: TaskGraph,
    profiles: Mapping[str, TaskProfile],
    provider: ProviderProfile,
    config: ExecutionConfig,
    per_call_context_tokens: float = 0.0,
) -> PreparedPlan:
    groups: dict[str, tuple[str, ...]] = {t: (t,) for t in graph.ids}
    plan_profiles = dict(profiles)
    if config.fuse:
        graph, plan_profiles, groups = fuse_profiles(
            graph, profiles, provider, config, per_call_context_tokens
        )
    if config.mode is Mode.LEVEL_SYNC:
        # Lower levels strictly first; within a level, topological order.
        n = len(graph)
        prio = {t: -float(graph.levels[t] * n + graph.topological_index[t]) for t in graph.ids}
    else:
        prio = priorities(graph, expected_durations(plan_profiles, provider), config.policy)
    return PreparedPlan(graph, plan_profiles, prio, groups)


def simulate(
    plan: PreparedPlan,
    provider: ProviderProfile,
    config: ExecutionConfig,
    rng: np.random.Generator,
    retry: RetryPolicy | None = None,
) -> RunResult:
    return _Simulation(plan, provider, config, rng, retry or RetryPolicy()).run()


class _Simulation:
    """State of one simulated run. Events: ``done`` (a call returned) and ``retry`` (back-off
    elapsed). Each heap entry is ``(time, seq, kind, task, attempt, ok)``; ``seq`` keeps ordering
    deterministic for simultaneous events."""

    def __init__(
        self,
        plan: PreparedPlan,
        provider: ProviderProfile,
        config: ExecutionConfig,
        rng: np.random.Generator,
        retry: RetryPolicy,
    ) -> None:
        self.graph, self.profiles = plan.graph, plan.profiles
        self.provider, self.rng, self.retry = provider, rng, retry
        self.queue = ReadyQueue(self.graph, plan.priority)
        self.client_free = math.inf if config.workers is None else config.workers

        self.level_sync = config.mode is Mode.LEVEL_SYNC
        self.level_size = Counter(self.graph.levels.values())
        self.level_done: Counter[int] = Counter()
        self.current_level = 0

        self.provider_active = 0
        self.provider_wait: deque[tuple[str, int, float]] = deque()  # (task, attempt, enqueued)
        self.events: list[tuple[float, int, str, str, int, bool]] = []
        self.seq = 0
        self.now = 0.0
        self.calls = self.rate_limited = self.completed = self.failed = self.cancelled = 0
        self.tokens_in = self.tokens_out = 0.0

    def run(self) -> RunResult:
        self._dispatch()
        while self.events:
            self.now = self.events[0][0]
            # Retire every event at this instant before dispatching, so the priority policy
            # chooses among all simultaneously released tasks (same semantics as list_schedule).
            while self.events and self.events[0][0] == self.now:
                _, _, kind, tid, attempt, ok = heapq.heappop(self.events)
                self._handle(kind, tid, attempt, ok)
            self._dispatch()

        if not self.queue.finished:  # pragma: no cover - would indicate a simulator bug
            raise RuntimeError("simulation ended with unfinished tasks")
        return RunResult(
            makespan=self.now,
            model_calls=self.calls,
            rate_limited_calls=self.rate_limited,
            input_tokens=self.tokens_in,
            output_tokens=self.tokens_out,
            cost_usd=self.provider.cost_usd(self.tokens_in, self.tokens_out),
            completed=self.completed,
            failed=self.failed,
            cancelled=self.cancelled,
            num_tasks=len(self.graph),
        )

    def _push(self, at: float, kind: str, tid: str, attempt: int, ok: bool = False) -> None:
        self.seq += 1
        heapq.heappush(self.events, (at, self.seq, kind, tid, attempt, ok))

    def _dispatch(self) -> None:
        """Hand ready tasks to free client slots, then start calls the provider can accept."""
        while self.queue and self.client_free > 0:
            if self.level_sync and self.graph.levels[self.queue.peek()] > self.current_level:
                break
            self.client_free -= 1
            self.provider_wait.append((self.queue.pop(), 0, self.now))
        self._start_waiting_calls()

    def _start_waiting_calls(self) -> None:
        prov, rng = self.provider, self.rng
        while self.provider_wait and self.provider_active < prov.max_concurrency:
            tid, attempt, enqueued_at = self.provider_wait.popleft()
            self.provider_active += 1
            p = self.profiles[tid]
            sigma = prov.output_tokens_sigma
            out = p.output_tokens * (float(rng.lognormal(-0.5 * sigma**2, sigma)) if sigma else 1)
            latency = prov.sample_latency(rng, p.input_tokens, out, self.provider_active)
            if self.now > enqueued_at:  # had to wait for a provider slot: 429 + back-off
                latency += prov.rate_limit_penalty_s
                self.rate_limited += 1
            self.calls += 1
            self.tokens_in += p.input_tokens
            self.tokens_out += out
            self._push(
                self.now + latency, "done", tid, attempt, bool(rng.random() < p.success_prob)
            )

    def _handle(self, kind: str, tid: str, attempt: int, ok: bool) -> None:
        if kind == "retry":
            self.provider_wait.append((tid, attempt, self.now))
            return
        self.provider_active -= 1
        if ok:
            self.completed += 1
            self.queue.complete(tid)
            self._finish(tid, [])
        elif attempt < self.retry.max_retries:
            self._push(self.now + self.retry.backoff(attempt, self.rng), "retry", tid, attempt + 1)
        else:
            self.failed += 1
            dropped = self.queue.fail(tid)
            self.cancelled += len(dropped)
            self._finish(tid, dropped)

    def _finish(self, tid: str, dropped: list[str]) -> None:
        self.client_free += 1
        for t in (tid, *dropped):
            self.level_done[self.graph.levels[t]] += 1
        size = self.level_size
        while (
            self.current_level in size
            and self.level_done[self.current_level] == size[self.current_level]
        ):
            self.current_level += 1


def deterministic(provider: ProviderProfile) -> ProviderProfile:
    """The provider with every source of randomness and interference switched off."""
    return replace(
        provider,
        noise_sigma=0.0,
        output_tokens_sigma=0.0,
        tail_prob=0.0,
        contention=0.0,
        max_concurrency=10**9,
    )
