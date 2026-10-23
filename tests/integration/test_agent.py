"""End-to-end tests of the LangGraph agent on the mock LLM (no API key required)."""

from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langgraph.checkpoint.memory import InMemorySaver

from adaptive_scheduler.agent import (
    AdaptiveAgent,
    AgentOptions,
    HeuristicPlanner,
    PlanSpec,
    Strategy,
    TaskExecutor,
    TaskSpec,
    TaskStatus,
)
from adaptive_scheduler.agent.planning import CallUsage
from adaptive_scheduler.cli import main
from adaptive_scheduler.config import Settings
from adaptive_scheduler.cost import ContextSize, ProviderProfile, build_profiles
from adaptive_scheduler.graph import TaskGraph, TaskType
from adaptive_scheduler.llm import MockChatModel, MockLLMError
from adaptive_scheduler.simulation import ExecutionConfig, Mode, RetryPolicy, prepare

FAST = 0.002  # wall-clock seconds per simulated second
REQUEST = "Research the EV markets of India, China and Germany, then compare them, then summarize"


def mock(**kw: Any) -> MockChatModel:
    return MockChatModel(time_scale=FAST, **kw)


def agent(llm: MockChatModel | None = None, planner: Any = None, **opts: Any) -> AdaptiveAgent:
    llm = llm or mock()
    options = AgentOptions(
        time_scale=FAST, selector_samples=16, retry=RetryPolicy(backoff_base_s=0.001), **opts
    )
    return AdaptiveAgent(llm, planner or HeuristicPlanner(llm, 32), options=options)


class FailOn(MockChatModel):
    """Mock that always fails for one task id (identified in the worker prompt)."""

    marker: str = ""

    async def _agenerate(
        self, messages: list[BaseMessage], *args: Any, **kwargs: Any
    ) -> ChatResult:
        if self.marker and self.marker in messages[-1].text:
            raise MockLLMError("injected failure")
        return await super()._agenerate(messages, *args, **kwargs)


class ScriptedPlanner:
    """Returns the given plans in order (the last one repeats)."""

    def __init__(self, *plans: PlanSpec) -> None:
        self.plans = list(plans)
        self.feedback: list[str | None] = []

    async def propose(self, request: str, feedback: str | None) -> tuple[PlanSpec, CallUsage]:
        self.feedback.append(feedback)
        plan = self.plans.pop(0) if len(self.plans) > 1 else self.plans[0]
        return plan, CallUsage("planning", 0.0, 10, 10)


def spec(deps: dict[str, list[str]]) -> PlanSpec:
    return PlanSpec(
        tasks=[
            TaskSpec(id=t, instruction=f"do {t}", task_type=TaskType.REASONING, depends_on=d)
            for t, d in deps.items()
        ]
    )


# ---------------------------------------------------------------------------- executor
def _prepared(graph: TaskGraph, config: ExecutionConfig):  # type: ignore[no-untyped-def]
    profiles = build_profiles(graph, ContextSize())
    return prepare(graph, profiles, ProviderProfile(), config)


@pytest.mark.parametrize("workers", [1, 2, 4])
async def test_executor_respects_dependencies_and_concurrency(workers: int) -> None:
    graph = spec(
        {"a": [], "b": [], "c": [], "d": ["a", "b"], "e": ["c"], "f": ["d", "e"]}
    ).to_graph()
    cfg = ExecutionConfig("k", workers=workers)
    report = await TaskExecutor(mock()).run("req", _prepared(graph, cfg), cfg)
    assert report.success
    assert report.model_calls == len(graph)
    assert report.peak_concurrency <= workers
    recs = report.records
    for tid, deps in graph.dependency_map.items():
        for dep in deps:
            assert recs[tid].start_s >= recs[dep].end_s  # type: ignore[operator]
    assert recs["f"].output  # outputs flow downstream
    assert recs["d"].input_tokens > recs["a"].input_tokens  # dependency outputs are in the prompt


async def test_executor_level_sync_waits_for_whole_level() -> None:
    graph = spec({"a": [], "b": [], "c": ["a"]}).to_graph()
    cfg = ExecutionConfig("static", workers=None, mode=Mode.LEVEL_SYNC)
    report = await TaskExecutor(mock()).run("req", _prepared(graph, cfg), cfg)
    r = report.records
    assert r["c"].start_s >= max(r["a"].end_s, r["b"].end_s)  # type: ignore[operator, type-var]


async def test_executor_isolates_failures() -> None:
    graph = spec({"a": [], "b": ["a"], "c": ["b"], "x": []}).to_graph()
    cfg = ExecutionConfig("k2", workers=2)
    llm = FailOn(time_scale=FAST, marker="Your task (a,")
    ex = TaskExecutor(llm, retry=RetryPolicy(max_retries=2, backoff_base_s=0.001))
    report = await ex.run("req", _prepared(graph, cfg), cfg)
    r = report.records
    assert r["a"].status is TaskStatus.FAILED
    assert r["a"].attempts == 3
    assert "injected failure" in (r["a"].error or "")
    assert r["b"].status is TaskStatus.CANCELLED
    assert r["c"].status is TaskStatus.CANCELLED
    assert r["x"].status is TaskStatus.COMPLETED
    assert not report.success


async def test_executor_timeout_counts_as_failure() -> None:
    graph = spec({"a": []}).to_graph()
    cfg = ExecutionConfig("k1", workers=1)
    slow = MockChatModel(time_scale=1.0)  # ~seconds per call
    ex = TaskExecutor(slow, timeout_s=0.01, retry=RetryPolicy(max_retries=0))
    report = await ex.run("req", _prepared(graph, cfg), cfg)
    assert report.records["a"].status is TaskStatus.FAILED
    assert "TimeoutError" in (report.records["a"].error or "")


def test_executor_rejects_bad_timeout() -> None:
    with pytest.raises(ValueError, match="timeout"):
        TaskExecutor(mock(), timeout_s=0)


# ---------------------------------------------------------------------------- workflow
@pytest.mark.parametrize("strategy", list(Strategy))
async def test_agent_runs_end_to_end(strategy: Strategy) -> None:
    a = agent()
    result = await a.arun(REQUEST, strategy=strategy)
    m = result.metrics
    assert m.success
    assert result.answer
    assert result.decision.strategy is strategy
    assert m.model_calls == 1 + m.tasks_executed  # planner + tasks; plan ends in synthesis task
    assert m.input_tokens > 0
    assert m.cost_usd > 0
    assert m.tasks_planned == 6  # 3 parallel lookups -> compare -> summarize -> final
    assert a.estimator.observations == m.tasks_executed
    if strategy is Strategy.SEQUENTIAL:
        assert m.peak_concurrency == 1


async def test_adaptive_decision_is_explained() -> None:
    result = await agent().arun(REQUEST)
    d = result.decision
    assert d.candidates
    assert d.config_name in {c.name for c in d.candidates}
    assert d.predicted_latency_s is not None
    assert d.parallelism > 1
    members = sorted(t for g in result.groups.values() for t in g)
    assert members == sorted(result.plan.ids)


async def test_invalid_plan_is_repaired_once() -> None:
    planner = ScriptedPlanner(spec({"a": ["b"], "b": ["a"]}), spec({"a": [], "b": ["a"]}))
    result = await agent(planner=planner).arun("anything")
    assert result.metrics.plan_attempts == 2
    assert "cycle" in result.plan_errors[0]
    assert planner.feedback[1] is not None
    assert "cycle" in planner.feedback[1]
    assert result.metrics.success


async def test_persistently_invalid_plan_falls_back_to_single_task() -> None:
    planner = ScriptedPlanner(spec({"a": ["zzz"]}))
    result = await agent(planner=planner).arun("anything")
    assert result.metrics.plan_fallback
    assert result.plan.ids == ("answer",)
    assert not result.metrics.success  # degraded runs are not counted as successes


async def test_plan_size_limit_triggers_repair() -> None:
    planner = ScriptedPlanner(spec({f"t{i}": [] for i in range(5)}), spec({"a": []}))
    result = await agent(planner=planner, max_tasks=3).arun("anything")
    assert result.metrics.plan_attempts == 2
    assert "limit" in result.plan_errors[0]


async def test_multiple_sinks_get_a_synthesis_call() -> None:
    planner = ScriptedPlanner(spec({"a": [], "b": []}))
    result = await agent(planner=planner).arun("anything")
    assert result.metrics.model_calls == 1 + 2 + 1  # planner + 2 tasks + synthesis
    assert result.metrics.synthesis_s > 0


async def test_partial_failure_still_answers() -> None:
    planner = ScriptedPlanner(spec({"a": [], "b": []}))
    llm = FailOn(time_scale=FAST, marker="Your task (a,")
    result = await agent(llm=llm, planner=planner).arun("anything")
    assert not result.metrics.success
    assert result.answer  # synthesized from the surviving branch
    assert result.report.records["b"].status is TaskStatus.COMPLETED


async def test_checkpointer_persists_state() -> None:
    llm = mock()
    saver = InMemorySaver()
    a = AdaptiveAgent(
        llm,
        HeuristicPlanner(llm, 32),
        options=AgentOptions(time_scale=FAST, selector_samples=8),
        checkpointer=saver,
    )
    await a.arun(REQUEST, thread_id="t1")
    snapshot = a.graph.get_state({"configurable": {"thread_id": "t1"}})
    assert snapshot.values["answer"]
    assert snapshot.values["report"].success


async def test_empty_request_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        await agent().arun("   ")


# ---------------------------------------------------------------------------- CLI
def test_cli_run_on_mock(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    monkeypatch.setenv("APSCHED_MOCK_TIME_SCALE", "0.002")
    monkeypatch.setenv("APSCHED_SELECTOR_SAMPLES", "8")
    assert main(["--provider", "mock", "run", REQUEST]) == 0
    out = capsys.readouterr().out
    assert "Execution timeline" in out
    assert "Strategy: adaptive" in out
    assert main(["--provider", "mock", "run", "--strategy", "sequential", "--json", "hi"]) == 0
    assert '"strategy": "sequential"' in capsys.readouterr().out


def test_from_settings_warm_starts_estimator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    log = tmp_path / "calls.csv"
    rows = ["task_type,status,latency_s,input_tokens,output_tokens"]
    rows += [f"reasoning,completed,{2 + i % 3},500,{300 + 50 * i}" for i in range(20)]
    log.write_text("\n".join(rows), encoding="utf-8")
    agent = AdaptiveAgent.from_settings(
        Settings(llm_provider="mock", estimator_warm_start=str(log))  # type: ignore[arg-type]
    )
    assert agent.estimator.observations == 20
