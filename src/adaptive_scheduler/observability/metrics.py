"""Prometheus metrics derived from the agent event stream.

Label cardinality is bounded: strategies, task types, statuses and configuration names
(``adaptive[k=4,fused]`` ...) come from small fixed sets; run and task ids are never labels.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from adaptive_scheduler.events import AgentEvent, EventType

LATENCY_BUCKETS = (0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600)


class MetricsCollector:
    """An event sink that records Prometheus metrics into its own registry."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        r = self.registry
        self.runs = Counter(
            "apsched_runs", "Completed agent runs.", ["strategy", "outcome"], registry=r
        )
        self.in_flight = Gauge("apsched_runs_in_flight", "Runs currently executing.", registry=r)
        self.run_duration = Histogram(
            "apsched_run_duration_seconds", "End-to-end run latency.", ["strategy"],
            buckets=LATENCY_BUCKETS, registry=r,
        )  # fmt: skip
        self.tasks = Counter(
            "apsched_tasks", "Finished tasks.", ["task_type", "status"], registry=r
        )
        self.task_duration = Histogram(
            "apsched_task_duration_seconds", "Task latency including retries.", ["task_type"],
            buckets=LATENCY_BUCKETS, registry=r,
        )  # fmt: skip
        self.model_calls = Counter("apsched_model_calls", "LLM calls (all purposes).", registry=r)
        self.tokens = Counter("apsched_tokens", "LLM tokens.", ["direction"], registry=r)
        self.cost = Counter("apsched_cost_usd", "Estimated LLM spend in USD.", registry=r)
        self.decisions = Counter(
            "apsched_decisions", "Execution configurations chosen.", ["config"], registry=r
        )
        self.parallelism = Histogram(
            "apsched_plan_parallelism", "Estimated W/S of plans.",
            buckets=(1, 1.25, 1.5, 2, 3, 4, 6, 8, 12), registry=r,
        )  # fmt: skip

    def __call__(self, event: AgentEvent) -> None:
        d = event.data
        if event.type is EventType.RUN_STARTED:
            self.in_flight.inc()
        elif event.type is EventType.DECISION:
            self.decisions.labels(config=d["config_name"]).inc()
            self.parallelism.observe(d["parallelism"])
        elif event.type is EventType.TASK_FINISHED:
            self.tasks.labels(task_type=d["task_type"], status=d["status"]).inc()
            if d["status"] != "cancelled":
                self.task_duration.labels(task_type=d["task_type"]).observe(d["duration_s"])
        elif event.type is EventType.RUN_FINISHED:
            m = d["metrics"]
            self.in_flight.dec()
            outcome = "success" if m["success"] else "partial"
            self.runs.labels(strategy=d["strategy"], outcome=outcome).inc()
            self.run_duration.labels(strategy=d["strategy"]).observe(m["wall_clock_s"])
            self.model_calls.inc(m["model_calls"])
            self.tokens.labels(direction="input").inc(m["input_tokens"])
            self.tokens.labels(direction="output").inc(m["output_tokens"])
            self.cost.inc(m["cost_usd"])
        elif event.type is EventType.RUN_FAILED:
            self.in_flight.dec()
            self.runs.labels(strategy=d.get("strategy", "unknown"), outcome="error").inc()
