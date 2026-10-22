import json
import logging
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from adaptive_scheduler.agent import (
    AdaptiveAgent,
    AgentOptions,
    HeuristicPlanner,
    PlanSpec,
    TaskSpec,
)
from adaptive_scheduler.agent.planning import CallUsage
from adaptive_scheduler.events import AgentEvent, EventType, emit, fan_out
from adaptive_scheduler.graph import TaskType
from adaptive_scheduler.llm import MockChatModel, MockLLMError
from adaptive_scheduler.observability import (
    JsonFormatter,
    MetricsCollector,
    SpanRecorder,
    request_id_var,
)
from adaptive_scheduler.simulation import RetryPolicy


class FailOn(MockChatModel):
    marker: str = ""

    async def _agenerate(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        if self.marker and self.marker in messages[-1].text:
            raise MockLLMError("boom")
        return await super()._agenerate(messages, *args, **kwargs)


class Plan:
    async def propose(self, request: str, feedback: str | None) -> tuple[PlanSpec, CallUsage]:
        spec = PlanSpec(tasks=[
            TaskSpec(id="a", instruction="a", task_type=TaskType.RETRIEVAL),
            TaskSpec(id="b", instruction="b", task_type=TaskType.RETRIEVAL),
            TaskSpec(id="c", instruction="c", task_type=TaskType.SYNTHESIS, depends_on=["a"]),
        ])  # fmt: skip
        return spec, CallUsage("planning", 0.0, 1, 1)


async def _run_with(sink: Any, llm: MockChatModel | None = None) -> None:
    llm = llm or MockChatModel(time_scale=0.001)
    opts = AgentOptions(time_scale=0.001, selector_samples=4, retry=RetryPolicy(max_retries=0))
    agent = AdaptiveAgent(llm, Plan(), options=opts)
    await agent.arun("anything", on_event=sink)


# ---------------------------------------------------------------------------- events
async def test_failing_sink_never_breaks_the_run(caplog: pytest.LogCaptureFixture) -> None:
    seen: list[EventType] = []

    def broken(_: AgentEvent) -> None:
        raise RuntimeError("consumer bug")

    await _run_with(fan_out([broken, lambda e: seen.append(e.type)]))
    assert seen[0] is EventType.RUN_STARTED
    assert seen[-1] is EventType.RUN_FINISHED
    assert "event sink failed" in caplog.text
    await emit(None, AgentEvent(type=EventType.RUN_STARTED, run_id="x"))  # no-op


# ---------------------------------------------------------------------------- metrics
async def test_metrics_collector_counts_tasks_and_failures() -> None:
    m = MetricsCollector()
    await _run_with(m, FailOn(time_scale=0.001, marker="Your task (a,"))
    value = m.registry.get_sample_value
    assert value("apsched_runs_total", {"strategy": "adaptive", "outcome": "partial"}) == 1
    assert value("apsched_tasks_total", {"task_type": "retrieval", "status": "failed"}) == 1
    assert value("apsched_tasks_total", {"task_type": "retrieval", "status": "completed"}) == 1
    assert value("apsched_tasks_total", {"task_type": "synthesis", "status": "cancelled"}) == 1
    assert value("apsched_runs_in_flight") == 0
    m(AgentEvent(type=EventType.RUN_STARTED, run_id="r", data={"strategy": "x"}))
    m(AgentEvent(type=EventType.RUN_FAILED, run_id="r", data={"strategy": "x", "error": "e"}))
    assert value("apsched_runs_total", {"strategy": "x", "outcome": "error"}) == 1
    assert value("apsched_runs_in_flight") == 0


# ---------------------------------------------------------------------------- tracing
async def test_span_recorder_builds_run_and_task_spans() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    recorder = SpanRecorder(provider.get_tracer("test"))
    await _run_with(recorder, FailOn(time_scale=0.001, marker="Your task (a,"))

    spans = exporter.get_finished_spans()
    runs = [s for s in spans if s.name == "agent.run"]
    tasks = {s.attributes["apsched.task_id"]: s for s in spans if s.name == "agent.task"}
    assert len(runs) == 1
    assert set(tasks) == {"a", "b"}  # c was cancelled and never started
    run = runs[0]
    assert all(
        t.parent is not None and t.parent.span_id == run.context.span_id for t in tasks.values()
    )
    assert tasks["a"].status.status_code is StatusCode.ERROR
    assert tasks["b"].status.status_code is StatusCode.UNSET
    assert run.attributes["apsched.success"] is False
    assert run.attributes["apsched.config"].startswith("adaptive[")
    assert tasks["b"].start_time >= run.start_time
    assert tasks["b"].end_time <= run.end_time


def test_span_recorder_closes_spans_on_failure() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    rec = SpanRecorder(provider.get_tracer("test"))
    rec(AgentEvent(type=EventType.RUN_STARTED, run_id="r", data={"strategy": "adaptive"}))
    rec(
        AgentEvent(
            type=EventType.TASK_STARTED, run_id="r", data={"task_id": "t", "task_type": "reasoning"}
        )
    )
    rec(AgentEvent(type=EventType.RUN_FAILED, run_id="r", data={"error": "x"}))
    assert {s.name for s in exporter.get_finished_spans()} == {"agent.run", "agent.task"}
    rec(AgentEvent(type=EventType.TASK_FINISHED, run_id="unknown", data={"task_id": "t"}))


# ---------------------------------------------------------------------------- logging
def test_json_formatter_includes_request_id_and_extras() -> None:
    record = logging.LogRecord("svc", logging.INFO, __file__, 1, "hello %s", ("world",), None)
    record.path = "/v1/runs"
    token = request_id_var.set("rid-7")
    try:
        payload = json.loads(JsonFormatter().format(record))
    finally:
        request_id_var.reset(token)
    assert payload["message"] == "hello world"
    assert payload["request_id"] == "rid-7"
    assert payload["path"] == "/v1/runs"
    assert payload["level"] == "INFO"
    _ = HeuristicPlanner  # imported for API completeness
