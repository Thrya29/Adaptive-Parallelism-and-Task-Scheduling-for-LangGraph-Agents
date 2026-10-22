"""The LangGraph agent: plan -> (repair) -> schedule -> execute -> synthesize.

::

    START -> plan --(invalid & attempts left)--> plan
                  \\-(valid, or fallback)-------> schedule -> execute -> synthesize -> END

* **plan** asks the planner for a :class:`PlanSpec`, validates it into a :class:`TaskGraph` and
  checks size limits. On failure the exact error is fed back for one repair attempt; if the plan
  is still invalid the agent degrades gracefully to a single-task plan.
* **schedule** builds per-task cost profiles from the current estimator beliefs and runs the Monte
  Carlo selector (or applies a fixed baseline strategy), recording every candidate's prediction.
* **execute** runs the plan with :class:`TaskExecutor` and feeds observed calls back into the
  :class:`CostEstimator`, so predictions improve with use.
* **synthesize** returns the single sink task's output directly when possible (no extra call) and
  otherwise combines the available results, flagging anything that failed.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError

from adaptive_scheduler.agent import prompts
from adaptive_scheduler.agent.executor import (
    ExecutionReport,
    TaskExecutor,
    TaskStatus,
    call_kwargs_for,
)
from adaptive_scheduler.agent.planning import (
    CallUsage,
    HeuristicPlanner,
    LLMPlanner,
    Planner,
    PlanningError,
    PlanSpec,
    TaskSpec,
    usage_from,
)
from adaptive_scheduler.config import LLMProvider, Settings
from adaptive_scheduler.cost.estimator import CostEstimator
from adaptive_scheduler.cost.profiles import ContextSize, build_profiles, estimate_tokens
from adaptive_scheduler.cost.provider import ProviderProfile
from adaptive_scheduler.events import AgentEvent, EventSink, EventType, emit
from adaptive_scheduler.graph.analysis import profile as graph_profile
from adaptive_scheduler.graph.model import PlanLimitError, TaskGraph, TaskType
from adaptive_scheduler.llm.factory import create_chat_model
from adaptive_scheduler.llm.mock import MockChatModel
from adaptive_scheduler.scheduling.simulator import PriorityPolicy
from adaptive_scheduler.simulation.engine import (
    ExecutionConfig,
    Mode,
    PreparedPlan,
    RetryPolicy,
    prepare,
)
from adaptive_scheduler.simulation.selector import Objective, select_strategy
from adaptive_scheduler.simulation.strategies import BASELINES, Strategy


# ---------------------------------------------------------------------------- options & results
@dataclass(frozen=True)
class AgentOptions:
    max_plan_attempts: int = 2
    max_tasks: int = 32
    max_depth: int = 12
    max_fan_in: int = 16
    task_timeout_s: float = 120.0
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    selector_samples: int = 64
    objective: Objective = field(default_factory=Objective)
    provider_prior: ProviderProfile = field(default_factory=ProviderProfile)
    # Wall-clock seconds per model second. 1.0 for real models; the mock LLM may run faster.
    time_scale: float = 1.0
    seed: int = 0


class CandidateSummary(BaseModel):
    name: str
    workers: int | None
    fuse: bool
    mean_latency_s: float
    p95_latency_s: float
    cost_usd: float
    calls: float
    success_rate: float
    score: float


class StrategyDecision(BaseModel):
    strategy: Strategy
    config_name: str
    workers: int | None
    fuse: bool
    mode: Mode
    policy: PriorityPolicy
    predicted_latency_s: float | None = None  # model seconds
    predicted_p95_s: float | None = None
    selection_time_s: float = 0.0
    total_work_s: float
    span_s: float
    parallelism: float
    candidates: list[CandidateSummary] = Field(default_factory=list)

    def to_config(self) -> ExecutionConfig:
        return ExecutionConfig(
            self.config_name,
            workers=self.workers,
            policy=self.policy,
            mode=self.mode,
            fuse=self.fuse,
        )


class RunMetrics(BaseModel):
    wall_clock_s: float
    planning_s: float
    selection_s: float
    execution_s: float
    synthesis_s: float
    model_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    success: bool
    plan_attempts: int
    plan_fallback: bool
    tasks_planned: int
    tasks_executed: int
    peak_concurrency: int


class AgentResult(BaseModel):
    run_id: str
    request: str
    answer: str
    plan: TaskGraph
    executed_plan: TaskGraph
    groups: dict[str, tuple[str, ...]]
    decision: StrategyDecision
    report: ExecutionReport
    metrics: RunMetrics
    plan_errors: list[str]


class AgentState(TypedDict, total=False):
    run_id: str
    request: str
    strategy: Strategy
    plan_attempts: int
    plan_errors: list[str]
    plan_fallback: bool
    plan: TaskGraph
    usage: list[CallUsage]
    decision: StrategyDecision
    prepared: PreparedPlan
    report: ExecutionReport
    answer: str
    timings: dict[str, float]


def _fallback_plan(request: str) -> TaskGraph:
    return PlanSpec(
        tasks=[TaskSpec(id="answer", instruction=request, task_type=TaskType.GENERATION)]
    ).to_graph()


# ---------------------------------------------------------------------------- the agent
class AdaptiveAgent:
    def __init__(
        self,
        llm: BaseChatModel,
        planner: Planner,
        *,
        options: AgentOptions | None = None,
        estimator: CostEstimator | None = None,
        checkpointer: BaseCheckpointSaver[Any] | None = None,
    ) -> None:
        self.llm = llm
        self.planner = planner
        self.options = options or AgentOptions()
        self.estimator = estimator or CostEstimator(prior=self.options.provider_prior)
        self.executor = TaskExecutor(
            llm,
            retry=self.options.retry,
            timeout_s=self.options.task_timeout_s,
            seed=self.options.seed,
            call_kwargs=call_kwargs_for(llm),
        )
        self.graph = self._build(checkpointer)

    @classmethod
    def from_settings(
        cls, settings: Settings, *, checkpointer: BaseCheckpointSaver[Any] | None = None
    ) -> AdaptiveAgent:
        llm = create_chat_model(settings)
        provider = settings.resolve_provider()
        scale = settings.mock_time_scale if provider is LLMProvider.MOCK else 1.0
        prior = (
            ProviderProfile.for_model(
                settings.anthropic_model, max_concurrency=settings.max_concurrency
            )
            if provider is LLMProvider.ANTHROPIC
            else ProviderProfile(max_concurrency=settings.max_concurrency)
        )
        if isinstance(llm, MockChatModel):
            llm = llm.model_copy(update={"time_scale": scale, "provider_profile": prior})
        base_retry = RetryPolicy(max_retries=settings.max_retries)
        options = AgentOptions(
            max_tasks=settings.max_plan_tasks,
            task_timeout_s=settings.task_timeout_s,
            retry=RetryPolicy(
                max_retries=base_retry.max_retries,
                backoff_base_s=base_retry.backoff_base_s * scale,
                backoff_cap_s=base_retry.backoff_cap_s * scale,
            ),
            selector_samples=settings.selector_samples,
            provider_prior=prior,
            time_scale=scale,
            seed=settings.mock_seed,
        )
        planner: Planner = (
            HeuristicPlanner(llm, options.max_tasks)
            if provider is LLMProvider.MOCK
            else LLMPlanner(llm, options.max_tasks)
        )
        return cls(llm, planner, options=options, checkpointer=checkpointer)

    # ------------------------------------------------------------------ public API
    async def arun(
        self,
        request: str,
        *,
        strategy: Strategy = Strategy.ADAPTIVE,
        thread_id: str | None = None,
        run_id: str | None = None,
        on_event: EventSink | None = None,
    ) -> AgentResult:
        """Run the pipeline; ``on_event`` receives an :class:`AgentEvent` per lifecycle step."""
        if not request.strip():
            raise ValueError("request must not be empty")
        run_id = run_id or uuid.uuid4().hex
        started = time.perf_counter()
        config: RunnableConfig = {
            "configurable": {"thread_id": thread_id or run_id, "on_event": on_event}
        }
        await emit(
            on_event,
            AgentEvent(
                type=EventType.RUN_STARTED,
                run_id=run_id,
                data={"request": request, "strategy": strategy.value},
            ),
        )
        try:
            state: AgentState = await self.graph.ainvoke(
                {
                    "run_id": run_id,
                    "request": request,
                    "strategy": strategy,
                    "plan_attempts": 0,
                    "plan_errors": [],
                    "usage": [],
                    "timings": {},
                },
                config,
            )
        except Exception as exc:
            await emit(
                on_event,
                AgentEvent(
                    type=EventType.RUN_FAILED,
                    run_id=run_id,
                    data={"strategy": strategy.value, "error": f"{type(exc).__name__}: {exc}"},
                ),
            )
            raise
        result = self._result(state, time.perf_counter() - started)
        await emit(
            on_event,
            AgentEvent(
                type=EventType.RUN_FINISHED,
                run_id=run_id,
                data={
                    "strategy": strategy.value,
                    "config": result.decision.config_name,
                    "metrics": result.metrics.model_dump(),
                },
            ),
        )
        return result

    # ------------------------------------------------------------------ graph construction
    def _build(self, checkpointer: BaseCheckpointSaver[Any] | None) -> Any:
        builder = StateGraph(AgentState)
        builder.add_node("plan", self._plan)
        builder.add_node("schedule", self._schedule)
        builder.add_node("execute", self._execute)
        builder.add_node("synthesize", self._synthesize)
        builder.add_edge(START, "plan")
        builder.add_conditional_edges("plan", self._after_plan, ["plan", "schedule"])
        builder.add_edge("schedule", "execute")
        builder.add_edge("execute", "synthesize")
        builder.add_edge("synthesize", END)
        return builder.compile(checkpointer=checkpointer)

    def _after_plan(self, state: AgentState) -> Literal["plan", "schedule"]:
        return "schedule" if "plan" in state else "plan"

    # ------------------------------------------------------------------ nodes
    async def _plan(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        opts = self.options
        attempts = state["plan_attempts"] + 1
        errors = list(state["plan_errors"])
        usage = list(state["usage"])
        feedback = errors[-1] if errors else None
        started = time.perf_counter()
        try:
            spec, call = await self.planner.propose(state["request"], feedback)
            usage.append(call)
            graph = spec.to_graph()
            graph.check_limits(
                max_tasks=opts.max_tasks, max_depth=opts.max_depth, max_fan_in=opts.max_fan_in
            )
        except (ValidationError, PlanLimitError, PlanningError) as exc:
            if isinstance(exc, PlanningError) and exc.usage:
                usage.append(exc.usage)
            errors.append(_short_error(exc))
            update: dict[str, Any] = {
                "plan_attempts": attempts,
                "plan_errors": errors,
                "usage": usage,
            }
            if attempts >= opts.max_plan_attempts:
                fallback = _fallback_plan(state["request"])
                await self._emit_plan(state, config, fallback, attempts, fallback=True)
                update |= {"plan": fallback, "plan_fallback": True}
            return update | {"timings": _add(state, "planning", started)}
        reduced = graph.transitive_reduction()
        await self._emit_plan(state, config, reduced, attempts, fallback=False)
        return {
            "plan": reduced,
            "plan_attempts": attempts,
            "plan_errors": errors,
            "plan_fallback": False,
            "usage": usage,
            "timings": _add(state, "planning", started),
        }

    async def _emit_plan(
        self,
        state: AgentState,
        config: RunnableConfig,
        graph: TaskGraph,
        attempts: int,
        *,
        fallback: bool,
    ) -> None:
        await emit(
            _sink(config),
            AgentEvent(
                type=EventType.PLAN_READY,
                run_id=state["run_id"],
                data={
                    "attempts": attempts,
                    "fallback": fallback,
                    "errors": state["plan_errors"],
                    "tasks": [
                        {
                            "id": t.id,
                            "task_type": t.task_type.value,
                            "depends_on": list(t.depends_on),
                            "instruction": t.instruction,
                        }
                        for t in graph.tasks
                    ],
                },
            ),
        )

    def _context(self, request: str) -> ContextSize:
        return ContextSize(
            system_tokens=estimate_tokens(prompts.WORKER_SYSTEM),
            shared_context_tokens=estimate_tokens(prompts.worker_request(request, "", "", "", {})),
        )

    async def _schedule(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        started = time.perf_counter()
        graph, strategy = state["plan"], state["strategy"]
        ctx = self._context(state["request"])
        provider = self.estimator.provider()
        profiles = build_profiles(
            graph, ctx, self.estimator.output_token_estimates(), self.estimator.success_estimates()
        )
        gp = graph_profile(
            graph,
            {
                t: provider.expected_latency(p.input_tokens, p.output_tokens)
                for t, p in profiles.items()
            },
        )
        common = {"total_work_s": gp.total_work, "span_s": gp.span, "parallelism": gp.parallelism}

        if strategy is Strategy.ADAPTIVE:
            sel = select_strategy(
                graph,
                profiles,
                provider,
                objective=self.options.objective,
                samples=self.options.selector_samples,
                seed=self.options.seed,
                per_call_context_tokens=ctx.per_call,
                retry=self.options.retry,
            )
            cfg = sel.chosen.config
            decision = StrategyDecision(
                strategy=strategy,
                config_name=cfg.name,
                workers=cfg.workers,
                fuse=cfg.fuse,
                mode=cfg.mode,
                policy=cfg.policy,
                predicted_latency_s=sel.chosen.mean_latency,
                predicted_p95_s=sel.chosen.p95_latency,
                selection_time_s=sel.selection_time_s,
                candidates=[
                    CandidateSummary(
                        name=p.config.name,
                        workers=p.config.workers,
                        fuse=p.config.fuse,
                        mean_latency_s=p.mean_latency,
                        p95_latency_s=p.p95_latency,
                        cost_usd=p.mean_cost_usd,
                        calls=p.mean_calls,
                        success_rate=p.success_rate,
                        score=p.score,
                    )
                    for p in sel.candidates
                ],
                **common,
            )
        else:
            cfg = BASELINES[strategy]
            decision = StrategyDecision(
                strategy=strategy,
                config_name=cfg.name,
                workers=cfg.workers,
                fuse=cfg.fuse,
                mode=cfg.mode,
                policy=cfg.policy,
                **common,
            )

        prepared = prepare(graph, profiles, provider, decision.to_config(), ctx.per_call)
        await emit(
            _sink(config),
            AgentEvent(
                type=EventType.DECISION,
                run_id=state["run_id"],
                data=decision.model_dump(mode="json") | {"groups": prepared.groups},
            ),
        )
        return {
            "decision": decision,
            "prepared": prepared,
            "timings": _add(state, "selection", started),
        }

    async def _execute(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        started = time.perf_counter()
        plan = state["prepared"]
        report = await self.executor.run(
            state["request"],
            plan,
            state["decision"].to_config(),
            on_event=_sink(config),
            run_id=state["run_id"],
        )
        self._learn(plan.graph, report)
        return {"report": report, "timings": _add(state, "execution", started)}

    def _learn(self, graph: TaskGraph, report: ExecutionReport) -> None:
        learn_from_report(self.estimator, graph, report, self.options.time_scale)

    async def _synthesize(self, state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        report, graph = state["report"], state["prepared"].graph
        sinks = [t for t in graph.topological_order if not graph.successor_map[t]]
        done = {t: r.output for t, r in report.records.items() if r.status is TaskStatus.COMPLETED}
        missing = [t for t, r in report.records.items() if r.status is not TaskStatus.COMPLETED]
        usage = list(state["usage"])

        if len(sinks) == 1 and sinks[0] in done:
            answer = done[sinks[0]]  # the plan already ends in a synthesis task: no extra call
        elif not done:
            answer = "The request could not be completed: every task failed."
        else:
            results = {t: done[t] for t in sinks if t in done} or done
            messages = [
                SystemMessage(prompts.SYNTHESIS_SYSTEM),
                HumanMessage(prompts.synthesis_request(state["request"], results, missing)),
            ]
            call_started = time.perf_counter()
            reply = await self.llm.ainvoke(messages)
            usage.append(usage_from(reply, "synthesis", time.perf_counter() - call_started))
            answer = reply.text
        return {"answer": answer, "usage": usage, "timings": _add(state, "synthesis", started)}

    # ------------------------------------------------------------------ result assembly
    def _result(self, state: AgentState, wall_clock_s: float) -> AgentResult:
        report, usage, timings = state["report"], state["usage"], state["timings"]
        tokens_in = report.input_tokens + sum(u.input_tokens for u in usage)
        tokens_out = report.output_tokens + sum(u.output_tokens for u in usage)
        metrics = RunMetrics(
            wall_clock_s=wall_clock_s,
            planning_s=timings.get("planning", 0.0),
            selection_s=timings.get("selection", 0.0),
            execution_s=report.makespan_s,
            synthesis_s=timings.get("synthesis", 0.0),
            model_calls=report.model_calls + len(usage),
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            cost_usd=self.options.provider_prior.cost_usd(tokens_in, tokens_out),
            success=report.success and not state.get("plan_fallback", False),
            plan_attempts=state["plan_attempts"],
            plan_fallback=state.get("plan_fallback", False),
            tasks_planned=len(state["plan"]),
            tasks_executed=len(state["prepared"].graph),
            peak_concurrency=report.peak_concurrency,
        )
        return AgentResult(
            run_id=state["run_id"],
            request=state["request"],
            answer=state["answer"],
            plan=state["plan"],
            executed_plan=state["prepared"].graph,
            groups=state["prepared"].groups,
            decision=state["decision"],
            report=report,
            metrics=metrics,
            plan_errors=state["plan_errors"],
        )


def learn_from_report(
    estimator: CostEstimator, graph: TaskGraph, report: ExecutionReport, time_scale: float = 1.0
) -> None:
    """Feed every executed call of a run into the estimator (latency rescaled to model seconds)."""
    scale = time_scale or 1.0
    for tid, rec in report.records.items():
        if rec.status is TaskStatus.CANCELLED or not rec.call_latencies_s:
            continue
        task = graph.task(tid)
        estimator.observe(
            task.task_type,
            latency_s=rec.call_latencies_s[-1] / scale,
            input_tokens=rec.input_tokens,
            output_tokens=rec.output_tokens,
            succeeded=rec.status is TaskStatus.COMPLETED,
            # A fused call's output covers several tasks: keep it out of per-type averages.
            update_type_stats=not task.fused_from,
        )


def _sink(config: RunnableConfig) -> EventSink | None:
    sink: EventSink | None = config.get("configurable", {}).get("on_event")
    return sink


def _add(state: AgentState, key: str, started: float) -> dict[str, float]:
    timings = dict(state.get("timings", {}))
    timings[key] = timings.get(key, 0.0) + time.perf_counter() - started
    return timings


def _short_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "; ".join(str(e.get("msg", e)) for e in exc.errors())
    return str(exc)
