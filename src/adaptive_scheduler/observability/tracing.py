"""OpenTelemetry spans derived from the agent event stream.

One ``agent.run`` span per run with a child ``agent.task`` span per executed task, so a trace
viewer shows the actual schedule as a Gantt chart. Plan and decision details are attached as span
events / attributes. Tracing is optional: without the ``observability`` extra, or without a
configured exporter, the recorder is a no-op.
"""

from __future__ import annotations

import os
from typing import Any

from adaptive_scheduler.events import AgentEvent

try:  # optional dependency
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
except ImportError:  # pragma: no cover - exercised only without the extra
    trace = None  # type: ignore[assignment]


def setup_tracing(service_name: str, *, console: bool = False) -> Any:
    """Install a tracer provider. Exports to the console when ``console`` is set, and to an OTLP
    collector when ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set and the OTLP exporter is installed.
    Returns the provider (or ``None`` when OpenTelemetry is unavailable)."""
    if trace is None:  # pragma: no cover
        return None
    from opentelemetry.sdk.resources import Resource  # noqa: PLC0415
    from opentelemetry.sdk.trace import TracerProvider  # noqa: PLC0415
    from opentelemetry.sdk.trace.export import (  # noqa: PLC0415
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if console:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # noqa: PLC0415
                OTLPSpanExporter,
            )
        except ImportError:  # pragma: no cover - optional exporter package
            pass
        else:
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    return provider


class SpanRecorder:
    """An event sink that turns run events into OpenTelemetry spans."""

    def __init__(self, tracer: Any = None) -> None:
        if tracer is None and trace is not None:
            tracer = trace.get_tracer("adaptive_scheduler")
        self._tracer = tracer
        self._runs: dict[str, Any] = {}
        self._tasks: dict[tuple[str, str], Any] = {}

    def __call__(self, event: AgentEvent) -> None:
        if self._tracer is None or trace is None:  # pragma: no cover
            return
        handler = getattr(self, f"_on_{event.type.value}", None)
        if handler is not None:
            handler(event, event.data, int(event.ts * 1e9))

    def _on_run_started(self, event: AgentEvent, d: dict[str, Any], ts: int) -> None:
        span = self._tracer.start_span("agent.run", start_time=ts)
        span.set_attribute("apsched.run_id", event.run_id)
        span.set_attribute("apsched.strategy", d["strategy"])
        self._runs[event.run_id] = span

    def _on_plan_ready(self, event: AgentEvent, d: dict[str, Any], ts: int) -> None:
        if run := self._runs.get(event.run_id):
            attrs = {"tasks": len(d["tasks"]), "attempts": d["attempts"], "fallback": d["fallback"]}
            run.add_event("plan_ready", attrs, timestamp=ts)

    def _on_decision(self, event: AgentEvent, d: dict[str, Any], ts: int) -> None:
        if run := self._runs.get(event.run_id):
            run.set_attribute("apsched.config", d["config_name"])
            run.set_attribute("apsched.parallelism", d["parallelism"])
            if d.get("workers") is not None:
                run.set_attribute("apsched.workers", d["workers"])

    def _on_task_started(self, event: AgentEvent, d: dict[str, Any], ts: int) -> None:
        if run := self._runs.get(event.run_id):
            ctx = trace.set_span_in_context(run)
            span = self._tracer.start_span("agent.task", context=ctx, start_time=ts)
            span.set_attribute("apsched.task_id", d["task_id"])
            span.set_attribute("apsched.task_type", d["task_type"])
            self._tasks[(event.run_id, d["task_id"])] = span

    def _on_task_finished(self, event: AgentEvent, d: dict[str, Any], ts: int) -> None:
        span = self._tasks.pop((event.run_id, d["task_id"]), None)
        if span is None:  # cancelled tasks never started
            return
        span.set_attribute("apsched.attempts", d["attempts"])
        span.set_attribute("apsched.input_tokens", d["input_tokens"])
        span.set_attribute("apsched.output_tokens", d["output_tokens"])
        if d["status"] != "completed":
            span.set_status(Status(StatusCode.ERROR, d.get("error") or d["status"]))
        span.end(end_time=ts)

    def _on_run_finished(self, event: AgentEvent, d: dict[str, Any], ts: int) -> None:
        if run := self._runs.get(event.run_id):
            run.set_attribute("apsched.success", d["metrics"]["success"])
        self._close(event.run_id, ts)

    def _on_run_failed(self, event: AgentEvent, d: dict[str, Any], ts: int) -> None:
        if run := self._runs.get(event.run_id):
            run.set_status(Status(StatusCode.ERROR, d.get("error", "")))
        self._close(event.run_id, ts)

    def _close(self, run_id: str, ts: int) -> None:
        for key in [k for k in self._tasks if k[0] == run_id]:
            self._tasks.pop(key).end(end_time=ts)
        if run := self._runs.pop(run_id, None):
            run.end(end_time=ts)
