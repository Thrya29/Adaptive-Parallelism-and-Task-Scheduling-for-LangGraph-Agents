"""Live asynchronous task executor.

The executor is the real-world twin of :mod:`adaptive_scheduler.simulation.engine`: it drives the
same :class:`ReadyQueue` with the same priorities, concurrency limit and level-synchronous mode,
but each "duration" is an actual LLM call. Per task it

* builds the prompt from the request, the task instruction and the outputs of its dependencies,
* enforces a timeout per attempt and retries transient errors with jittered exponential back-off
  (the task keeps its worker slot while backing off, like a semaphore-guarded worker),
* on final failure cancels only the task's transitive dependents; independent branches continue,
* records start/end time, attempts, per-call latencies and token usage for metrics and learning.
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import Any

import numpy as np
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from adaptive_scheduler.agent import prompts
from adaptive_scheduler.cost.profiles import OUTPUT_TOKEN_PRIORS, TaskProfile, estimate_tokens
from adaptive_scheduler.events import AgentEvent, EventSink, EventType, emit
from adaptive_scheduler.graph.model import TaskType
from adaptive_scheduler.scheduling.ready_queue import ReadyQueue
from adaptive_scheduler.simulation.engine import ExecutionConfig, Mode, PreparedPlan, RetryPolicy

# Extra keyword arguments for a model call, e.g. an output-length hint for the mock LLM.
CallKwargs = Callable[[TaskProfile], dict[str, Any]]


class TaskStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskRecord(BaseModel):
    task_id: str
    status: TaskStatus
    attempts: int = 0
    start_s: float | None = None
    end_s: float | None = None
    call_latencies_s: list[float] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    output: str = ""
    error: str | None = None

    @property
    def duration_s(self) -> float:
        if self.start_s is None or self.end_s is None:
            return 0.0
        return self.end_s - self.start_s


class ExecutionReport(BaseModel):
    config_name: str
    workers: int | None
    makespan_s: float
    records: dict[str, TaskRecord]

    @property
    def success(self) -> bool:
        return all(r.status is TaskStatus.COMPLETED for r in self.records.values())

    @property
    def model_calls(self) -> int:
        return sum(r.attempts for r in self.records.values())

    @property
    def input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.records.values())

    @property
    def output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.records.values())

    @property
    def peak_concurrency(self) -> int:
        events = sorted(
            [(r.start_s, 1) for r in self.records.values() if r.start_s is not None]
            + [(r.end_s, -1) for r in self.records.values() if r.end_s is not None]
        )
        peak = active = 0
        for _, delta in events:  # ends sort before starts at equal times (-1 < 1)
            active += delta
            peak = max(peak, active)
        return peak


def call_kwargs_for(llm: BaseChatModel) -> CallKwargs | None:
    """Per-call extras for a model. Only the mock understands the output-length hint; a real API
    would reject the unknown parameter.

    The hint is the task type's *fixed* prior length, not the estimator's current belief: the mock
    plays the role of the real model, whose true output lengths do not depend on what the scheduler
    believes. (Feeding beliefs back into the "truth" made estimates drift like a random walk.)
    """
    from adaptive_scheduler.llm.mock import MockChatModel  # noqa: PLC0415 - avoid import cycle

    if not isinstance(llm, MockChatModel):
        return None

    def hint(p: TaskProfile) -> dict[str, Any]:
        tokens = (
            p.output_tokens
            if p.task_type is TaskType.COMPOSITE
            else OUTPUT_TOKEN_PRIORS[p.task_type]
        )
        return {"expected_output_tokens": round(tokens)}

    return hint


class TaskExecutor:
    def __init__(
        self,
        llm: BaseChatModel,
        *,
        retry: RetryPolicy | None = None,
        timeout_s: float = 120.0,
        seed: int = 0,
        call_kwargs: CallKwargs | None = None,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        self._llm = llm
        self._retry = retry or RetryPolicy()
        self._timeout_s = timeout_s
        self._rng = np.random.default_rng(seed)
        self._call_kwargs = call_kwargs or (lambda _profile: {})

    async def run(
        self,
        request: str,
        plan: PreparedPlan,
        config: ExecutionConfig,
        *,
        on_event: EventSink | None = None,
        run_id: str = "",
    ) -> ExecutionReport:
        graph = plan.graph
        queue = ReadyQueue(graph, plan.priority)
        limit = config.workers if config.workers is not None else len(graph)
        level_sync = config.mode is Mode.LEVEL_SYNC
        level_size = Counter(graph.levels.values())
        level_done: Counter[int] = Counter()
        current_level = 0

        outputs: dict[str, str] = {}
        records: dict[str, TaskRecord] = {}
        running: dict[asyncio.Task[TaskRecord], str] = {}
        t0 = time.perf_counter()

        try:
            while True:
                while queue and len(running) < limit:
                    if level_sync and graph.levels[queue.peek()] > current_level:
                        break
                    tid = queue.pop()
                    inputs = {d: outputs[d] for d in graph.dependency_map[tid]}
                    coro = self._run_task(
                        request, plan, tid, inputs, t0, on_event=on_event, run_id=run_id
                    )
                    running[asyncio.create_task(coro, name=f"task:{tid}")] = tid
                if not running:
                    break
                done, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
                for fut in sorted(done, key=lambda f: graph.topological_index[running[f]]):
                    tid = running.pop(fut)
                    record = fut.result()
                    records[tid] = record
                    finished = [tid]
                    if record.status is TaskStatus.COMPLETED:
                        outputs[tid] = record.output
                        queue.complete(tid)
                    else:
                        for dep in queue.fail(tid):
                            records[dep] = TaskRecord(
                                task_id=dep,
                                status=TaskStatus.CANCELLED,
                                error=f"upstream task {tid!r} failed",
                            )
                            finished.append(dep)
                            await emit(on_event, _finished_event(run_id, records[dep], graph))
                    for t in finished:
                        level_done[graph.levels[t]] += 1
                    while level_done[current_level] == level_size.get(current_level, -1):
                        current_level += 1
        except BaseException:
            for fut in running:
                fut.cancel()
            await asyncio.gather(*running, return_exceptions=True)
            raise

        ordered = {t: records[t] for t in graph.topological_order}
        return ExecutionReport(
            config_name=config.name,
            workers=config.workers,
            makespan_s=time.perf_counter() - t0,
            records=ordered,
        )

    async def _run_task(
        self,
        request: str,
        plan: PreparedPlan,
        tid: str,
        inputs: Mapping[str, str],
        t0: float,
        *,
        on_event: EventSink | None = None,
        run_id: str = "",
    ) -> TaskRecord:
        task = plan.graph.task(tid)
        messages = [
            SystemMessage(prompts.WORKER_SYSTEM),
            HumanMessage(
                prompts.worker_request(request, tid, task.task_type.value, task.instruction, inputs)
            ),
        ]
        kwargs = self._call_kwargs(plan.profiles[tid])
        record = TaskRecord(task_id=tid, status=TaskStatus.FAILED, start_s=time.perf_counter() - t0)
        await emit(
            on_event,
            AgentEvent(
                type=EventType.TASK_STARTED,
                run_id=run_id,
                data={"task_id": tid, "task_type": task.task_type.value, "inputs": list(inputs)},
            ),
        )

        for attempt in range(self._retry.max_retries + 1):
            record.attempts += 1
            started = time.perf_counter()
            try:
                reply = await asyncio.wait_for(
                    self._llm.ainvoke(messages, **kwargs), timeout=self._timeout_s
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                record.call_latencies_s.append(time.perf_counter() - started)
                record.error = f"{type(exc).__name__}: {exc}"
                if attempt < self._retry.max_retries:
                    await asyncio.sleep(self._retry.backoff(attempt, self._rng))
                continue

            record.call_latencies_s.append(time.perf_counter() - started)
            usage: Mapping[str, Any] = reply.usage_metadata or {}
            record.output = reply.text
            record.input_tokens += int(
                usage.get("input_tokens", estimate_tokens(str(messages[-1].content)))
            )
            record.output_tokens += int(usage.get("output_tokens", estimate_tokens(record.output)))
            record.status = TaskStatus.COMPLETED
            record.error = None
            break

        record.end_s = time.perf_counter() - t0
        await emit(on_event, _finished_event(run_id, record, plan.graph))
        return record


def _finished_event(run_id: str, record: TaskRecord, graph: Any) -> AgentEvent:
    return AgentEvent(
        type=EventType.TASK_FINISHED,
        run_id=run_id,
        data={
            "task_id": record.task_id,
            "task_type": graph.task(record.task_id).task_type.value,
            "status": record.status.value,
            "attempts": record.attempts,
            "duration_s": record.duration_s,
            "input_tokens": record.input_tokens,
            "output_tokens": record.output_tokens,
            "error": record.error,
        },
    )
