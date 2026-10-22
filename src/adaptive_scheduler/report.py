"""Human-readable rendering of an agent run (used by the CLI)."""

from __future__ import annotations

from adaptive_scheduler.agent.executor import TaskStatus
from adaptive_scheduler.agent.workflow import AgentResult

_STATUS_MARK = {TaskStatus.COMPLETED: "#", TaskStatus.FAILED: "x", TaskStatus.CANCELLED: "-"}


def gantt(result: AgentResult, width: int = 60) -> list[str]:
    """One text row per executed task showing when it ran relative to the execution window."""
    records = result.report.records
    span = max((r.end_s or 0.0 for r in records.values()), default=0.0) or 1.0
    name_w = max((len(t) for t in records), default=4)
    lines = []
    for tid, rec in records.items():
        if rec.start_s is None or rec.end_s is None:
            bar = " " * width
        else:
            a = round(rec.start_s / span * width)
            b = max(a + 1, round(rec.end_s / span * width))
            bar = " " * a + _STATUS_MARK[rec.status] * (b - a) + " " * (width - b)
        lines.append(f"  {tid:<{name_w}} |{bar}| {rec.duration_s:6.2f}s {rec.status.value}")
    return lines


def render(result: AgentResult, max_answer_chars: int | None = None) -> str:
    d, m = result.decision, result.metrics
    out = [f"Request: {result.request}", ""]

    out.append(f"Plan ({len(result.plan)} tasks, attempts={m.plan_attempts}"
               f"{', FALLBACK' if m.plan_fallback else ''}):")  # fmt: skip
    for tid in result.plan.topological_order:
        task = result.plan.task(tid)
        deps = f" <- {', '.join(task.depends_on)}" if task.depends_on else ""
        out.append(f"  [{task.task_type.value:<10}] {tid}{deps}: {task.instruction[:80]}")
    for err in result.plan_errors:
        out.append(f"  repaired error: {err}")

    out += ["", f"Strategy: {d.strategy.value} -> {d.config_name}"]
    out.append(
        f"  estimated work W={d.total_work_s:.1f}s, critical path S={d.span_s:.1f}s, "
        f"parallelism W/S={d.parallelism:.2f}"
    )
    if d.candidates:
        out.append(
            f"  {'candidate':<24} {'mean':>8} {'p95':>8} {'cost $':>9} {'calls':>6} {'succ':>6}"
        )
        for c in sorted(d.candidates, key=lambda c: c.score):
            mark = "*" if c.name == d.config_name else " "
            out.append(
                f" {mark}{c.name:<24} {c.mean_latency_s:7.1f}s {c.p95_latency_s:7.1f}s "
                f"{c.cost_usd:9.5f} {c.calls:6.1f} {c.success_rate:6.3f}"
            )
        out.append(f"  (model seconds; selection took {d.selection_time_s * 1000:.0f} ms)")

    out += ["", "Execution timeline:", *gantt(result), ""]
    out.append(
        f"Metrics: wall-clock {m.wall_clock_s:.2f}s (plan {m.planning_s:.2f}s, select "
        f"{m.selection_s:.2f}s, execute {m.execution_s:.2f}s, synthesize {m.synthesis_s:.2f}s)"
    )
    out.append(
        f"         {m.model_calls} model calls, {m.input_tokens} in / {m.output_tokens} out "
        f"tokens, ~${m.cost_usd:.5f}, peak concurrency {m.peak_concurrency}, "
        f"success={'yes' if m.success else 'no'}"
    )
    answer = result.answer
    if max_answer_chars is not None and len(answer) > max_answer_chars:
        answer = f"{answer[:max_answer_chars]}... [{len(result.answer)} chars total]"
    out += ["", "Answer:", answer]
    return "\n".join(out)
