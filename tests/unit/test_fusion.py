import math

import pytest

from adaptive_scheduler.graph import TaskGraph, TaskType, profile
from adaptive_scheduler.scheduling import (
    FusionParams,
    fuse_chains,
    maximal_chains,
    optimal_chain_segmentation,
)


def test_maximal_chains() -> None:
    # a -> b -> c is a chain; c fans out to d, e which join at f; f -> g is a chain.
    g = TaskGraph.from_dependencies(
        {"a": [], "b": ["a"], "c": ["b"], "d": ["c"], "e": ["c"], "f": ["d", "e"], "g": ["f"]}
    )
    assert maximal_chains(g) == [("a", "b", "c"), ("d",), ("e",), ("f", "g")]


def test_overhead_dominated_chain_is_fused() -> None:
    params = FusionParams(call_overhead=1.0, max_tasks_per_call=4)
    plan = optimal_chain_segmentation([1.5, 1.5, 1.5], params)
    assert plan.segments == ((0, 3),)
    assert plan.objective == pytest.approx(1.0 + 3 * 0.5)


def test_no_overhead_means_no_fusion() -> None:
    plan = optimal_chain_segmentation([2.0, 2.0], FusionParams(call_overhead=0.0))
    assert plan.segments == ((0, 1), (1, 2))


def test_penalty_limits_segment_length() -> None:
    # Saving per extra task is 1.0; quadratic penalty 0.4*(L-1)^2 makes L=2 optimal blocks.
    params = FusionParams(call_overhead=1.0, max_tasks_per_call=10, penalty_per_extra_task=0.4)
    plan = optimal_chain_segmentation([2.0] * 4, params)
    assert plan.segments == ((0, 2), (2, 4))


def test_work_limit_respected() -> None:
    params = FusionParams(call_overhead=1.0, max_work_per_call=2.5)
    plan = optimal_chain_segmentation([3.0, 3.0, 3.0], params)  # work 2 each
    assert all(b - a == 1 for a, b in plan.segments)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"call_overhead": -1.0},
        {"call_overhead": math.inf},
        {"call_overhead": 1.0, "max_tasks_per_call": 0},
        {"call_overhead": 1.0, "max_work_per_call": 0.0},
        {"call_overhead": 1.0, "penalty_per_extra_task": -0.1},
    ],
)
def test_params_validation(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        FusionParams(**kwargs)  # type: ignore[arg-type]


def test_fuse_chains_builds_valid_graph() -> None:
    g = TaskGraph.from_dependencies({"a": [], "b": ["a"], "c": ["b"], "x": [], "z": ["c", "x"]})
    d = {"a": 1.2, "b": 1.2, "c": 1.2, "x": 3.0, "z": 1.0}
    res = fuse_chains(g, d, FusionParams(call_overhead=1.0))
    assert res.groups["fused:a..c"] == ("a", "b", "c")
    assert res.calls_saved == 2
    fused = res.graph.task("fused:a..c")
    assert fused.fused_from == ("a", "b", "c")
    assert fused.task_type is TaskType.REASONING
    assert "Step 2/3 (b)" in fused.instruction
    assert res.graph.task("z").depends_on == ("fused:a..c", "x")
    assert res.durations["fused:a..c"] == pytest.approx(1.6)
    assert res.durations["x"] == 3.0

    before, after = profile(g, d), profile(res.graph, res.durations)
    assert after.total_work < before.total_work
    assert after.span <= before.span


def test_mixed_types_become_composite() -> None:
    g = TaskGraph.from_dependencies({"a": [], "b": ["a"]})
    g = TaskGraph(
        tasks=(g.task("a").model_copy(update={"task_type": TaskType.RETRIEVAL}), g.task("b"))
    )
    res = fuse_chains(g, {"a": 2.0, "b": 2.0}, FusionParams(call_overhead=1.0))
    assert res.graph.tasks[0].task_type is TaskType.COMPOSITE
