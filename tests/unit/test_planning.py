from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from adaptive_scheduler.agent import LLMPlanner, PlanSpec, TaskSpec, heuristic_plan
from adaptive_scheduler.agent.planning import PlanningError
from adaptive_scheduler.graph import TaskType


def test_single_request_is_single_task() -> None:
    plan = heuristic_plan("Explain quicksort")
    assert [t.id for t in plan.tasks] == ["answer"]
    assert plan.tasks[0].task_type is TaskType.GENERATION


def test_lists_fan_out_and_stages_chain() -> None:
    plan = heuristic_plan(
        "Research the EV markets of India, China and Germany, then write a summary"
    )
    g = plan.to_graph()
    assert g.task("s1_2").instruction == "Research the EV markets of China"
    assert g.task("s2_1").depends_on == ("s1_1", "s1_2", "s1_3")
    assert g.task("final").task_type is TaskType.SYNTHESIS
    assert (g.depth, g.max_width) == (3, 3)


def test_bullets_are_parallel() -> None:
    plan = heuristic_plan("Check:\n- the weather\n- the traffic\n- the news")
    g = plan.to_graph()
    assert g.max_width == 3
    assert g.task("final").depends_on == ("s1_1", "s1_2", "s1_3")


def test_fan_out_is_capped() -> None:
    plan = heuristic_plan("Look up " + ", ".join(f"item{i}" for i in range(20)))
    assert len(plan.tasks) == 8 + 1


def test_invalid_plan_spec_raises() -> None:
    spec = PlanSpec(
        tasks=[
            TaskSpec(id="a", instruction="x", task_type=TaskType.REASONING, depends_on=["b"]),
            TaskSpec(id="b", instruction="y", task_type=TaskType.REASONING, depends_on=["a"]),
        ]
    )
    with pytest.raises(ValidationError, match="cycle"):
        spec.to_graph()


class _StructuredStub:
    """Duck-typed chat model whose structured output returns a fixed payload."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def with_structured_output(self, schema: Any, include_raw: bool = False) -> RunnableLambda:  # type: ignore[type-arg]
        assert include_raw
        return RunnableLambda(lambda _messages: self.payload)


async def test_llm_planner_returns_plan_and_usage() -> None:
    spec = PlanSpec(tasks=[TaskSpec(id="a", instruction="x", task_type=TaskType.REASONING)])
    raw = AIMessage(
        "", usage_metadata={"input_tokens": 50, "output_tokens": 20, "total_tokens": 70}
    )
    planner = LLMPlanner(_StructuredStub({"raw": raw, "parsed": spec}), max_tasks=8)  # type: ignore[arg-type]
    plan, usage = await planner.propose("do x", None)
    assert plan == spec
    assert (usage.input_tokens, usage.output_tokens, usage.purpose) == (50, 20, "planning")


async def test_llm_planner_reports_parse_failure_with_usage() -> None:
    raw = AIMessage("", usage_metadata={"input_tokens": 5, "output_tokens": 1, "total_tokens": 6})
    planner = LLMPlanner(
        _StructuredStub({"raw": raw, "parsed": None, "parsing_error": "bad json"}),  # type: ignore[arg-type]
        max_tasks=8,
    )
    with pytest.raises(PlanningError, match="bad json") as info:
        await planner.propose("do x", None)
    assert info.value.usage is not None
    assert info.value.usage.input_tokens == 5
