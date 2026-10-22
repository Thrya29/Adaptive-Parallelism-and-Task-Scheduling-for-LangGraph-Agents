"""Request decomposition into a task DAG.

Two planners share one interface:

* :class:`LLMPlanner` - the LLM returns a :class:`PlanSpec` through structured output (tool
  calling). Used with a real model.
* :class:`HeuristicPlanner` - a deterministic rule-based decomposition used with the mock LLM. It
  still makes one mock LLM call so planning latency and tokens are accounted for like a real run.

Validation and the repair loop live in the LangGraph workflow: an invalid plan's exact error
(unknown dependency, cycle, size limit...) is fed back to the planner once.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Protocol, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from adaptive_scheduler.agent import prompts
from adaptive_scheduler.graph.model import Task, TaskGraph, TaskType


class TaskSpec(BaseModel):
    """One task of the plan."""

    id: str = Field(description="Short unique snake_case identifier")
    instruction: str = Field(description="Self-contained instruction for one LLM call")
    task_type: TaskType = Field(
        description="retrieval | reasoning | tool_call | generation | synthesis"
    )
    depends_on: list[str] = Field(
        default_factory=list, description="ids of tasks whose output this task needs"
    )


class PlanSpec(BaseModel):
    """A plan: tasks and the dependencies between them (must form a DAG)."""

    tasks: list[TaskSpec] = Field(min_length=1)

    def to_graph(self) -> TaskGraph:
        """Validate into a :class:`TaskGraph`; raises ``pydantic.ValidationError`` if invalid."""
        return TaskGraph(
            tasks=tuple(
                Task(
                    id=t.id,
                    instruction=t.instruction,
                    task_type=t.task_type,
                    depends_on=tuple(t.depends_on),
                )
                for t in self.tasks
            )
        )


@dataclass(frozen=True, slots=True)
class CallUsage:
    """Accounting for one LLM call made outside the task executor (planning, synthesis)."""

    purpose: str
    latency_s: float
    input_tokens: int
    output_tokens: int


class PlanningError(RuntimeError):
    def __init__(self, message: str, usage: CallUsage | None = None) -> None:
        super().__init__(message)
        self.usage = usage


class Planner(Protocol):
    async def propose(self, request: str, feedback: str | None) -> tuple[PlanSpec, CallUsage]: ...


def usage_from(message: Any, purpose: str, latency_s: float) -> CallUsage:
    meta = getattr(message, "usage_metadata", None) or {}
    return CallUsage(
        purpose, latency_s, int(meta.get("input_tokens", 0)), int(meta.get("output_tokens", 0))
    )


class LLMPlanner:
    def __init__(self, llm: BaseChatModel, max_tasks: int) -> None:
        self._structured = llm.with_structured_output(PlanSpec, include_raw=True)
        self._max_tasks = max_tasks

    async def propose(self, request: str, feedback: str | None) -> tuple[PlanSpec, CallUsage]:
        messages = [
            SystemMessage(prompts.PLANNER_SYSTEM.format(max_tasks=self._max_tasks)),
            HumanMessage(prompts.planner_request(request, feedback)),
        ]
        started = time.perf_counter()
        out = cast("dict[str, Any]", await self._structured.ainvoke(messages))
        usage = usage_from(out["raw"], "planning", time.perf_counter() - started)
        parsed = out.get("parsed")
        if not isinstance(parsed, PlanSpec):
            raise PlanningError(
                f"planner output could not be parsed: {out.get('parsing_error')}", usage
            )
        return parsed, usage


# ---------------------------------------------------------------------------- heuristic planner
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.*\S)\s*$")
_STAGE_SPLIT = re.compile(
    r"\s*(?:;|,?\s*\b(?:then|after that|afterwards|finally|next)\b,?)\s*", re.I
)
_ITEM_SPLIT = re.compile(r"\s*(?:,\s*(?:and\s+)?|\band\b)\s*", re.I)
_MAX_ITEMS = 8


def _expand_items(stage: str) -> list[str]:
    """``"Research the EV markets of India, China and Germany"`` -> one instruction per item,
    re-using the shared prefix ("Research the EV markets of ...")."""
    parts = [p for p in _ITEM_SPLIT.split(stage) if p]
    if len(parts) < 2:
        return [stage]
    words = parts[0].split()
    if len(words) < 2:
        return parts[:_MAX_ITEMS]
    prefix = " ".join(words[:-1])
    return [parts[0], *(f"{prefix} {p}" for p in parts[1:])][:_MAX_ITEMS]


def heuristic_plan(request: str) -> PlanSpec:
    """Deterministic decomposition: bullets are parallel tasks; "then"/";" separate sequential
    stages; comma/"and" lists inside a stage fan out. A synthesis task joins multi-task plans."""
    text = request.strip()
    bullets = [m.group(1) for line in text.splitlines() if (m := _BULLET.match(line))]
    stages: list[list[str]]
    if len(bullets) >= 2:
        stages = [bullets[:_MAX_ITEMS]]
    else:
        raw = [s.strip(" .,") for s in _STAGE_SPLIT.split(" ".join(text.split()))]
        stages = [_expand_items(s) for s in raw if s]
    if not stages:
        stages = [[text or "Answer the request."]]

    if len(stages) == 1 and len(stages[0]) == 1:
        return PlanSpec(
            tasks=[TaskSpec(id="answer", instruction=stages[0][0], task_type=TaskType.GENERATION)]
        )

    tasks: list[TaskSpec] = []
    previous: list[str] = []
    for si, items in enumerate(stages):
        task_type = TaskType.RETRIEVAL if si == 0 else TaskType.REASONING
        ids = [f"s{si + 1}_{j + 1}" for j in range(len(items))]
        tasks.extend(
            TaskSpec(id=i, instruction=item, task_type=task_type, depends_on=list(previous))
            for i, item in zip(ids, items, strict=True)
        )
        previous = ids
    tasks.append(
        TaskSpec(
            id="final",
            instruction="Combine the results into the final answer to the request.",
            task_type=TaskType.SYNTHESIS,
            depends_on=previous,
        )
    )
    return PlanSpec(tasks=tasks)


class HeuristicPlanner:
    def __init__(self, llm: BaseChatModel, max_tasks: int) -> None:
        self._llm = llm
        self._max_tasks = max_tasks

    async def propose(self, request: str, feedback: str | None) -> tuple[PlanSpec, CallUsage]:
        plan = heuristic_plan(request)
        messages = [
            SystemMessage(prompts.PLANNER_SYSTEM.format(max_tasks=self._max_tasks)),
            HumanMessage(prompts.planner_request(request, feedback)),
        ]
        started = time.perf_counter()
        # Plan JSON is roughly 60 output tokens per task.
        reply: AIMessage = await self._llm.ainvoke(
            messages, expected_output_tokens=60 * len(plan.tasks)
        )
        return plan, usage_from(reply, "planning", time.perf_counter() - started)
