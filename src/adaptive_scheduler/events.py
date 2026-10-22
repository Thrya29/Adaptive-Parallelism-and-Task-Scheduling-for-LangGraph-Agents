"""Typed run events: the single source for streaming, metrics and tracing.

The agent emits an :class:`AgentEvent` at every lifecycle step of a run. Consumers (the SSE
endpoint, the Prometheus collector, the OpenTelemetry span recorder) subscribe through an
:data:`EventSink`, so the live stream, the metrics and the traces can never disagree.
"""

from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EventType(StrEnum):
    RUN_STARTED = "run_started"
    PLAN_READY = "plan_ready"
    DECISION = "decision"
    TASK_STARTED = "task_started"
    TASK_FINISHED = "task_finished"
    RUN_FINISHED = "run_finished"
    RUN_FAILED = "run_failed"


class AgentEvent(BaseModel):
    type: EventType
    run_id: str
    ts: float = Field(default_factory=time.time, description="Unix time (seconds)")
    data: dict[str, Any] = Field(default_factory=dict)


EventSink = Callable[[AgentEvent], Awaitable[None] | None]


async def emit(sink: EventSink | None, event: AgentEvent) -> None:
    """Deliver an event. A failing consumer is logged and never breaks the run."""
    if sink is None:
        return
    try:
        result = sink(event)
        if inspect.isawaitable(result):
            await result
    except Exception:
        logger.exception("event sink failed for %s", event.type)


def fan_out(sinks: Sequence[EventSink | None]) -> EventSink:
    """Combine several sinks into one (each delivery isolated from the others)."""
    active = [s for s in sinks if s is not None]

    async def sink(event: AgentEvent) -> None:
        for s in active:
            await emit(s, event)

    return sink
