import pytest
from pydantic import ValidationError

from adaptive_scheduler.graph import (
    CycleError,
    PlanLimitError,
    Task,
    TaskGraph,
    critical_path,
    earliest_finish,
    profile,
    slack,
    upward_rank,
)
from adaptive_scheduler.graph import algorithms as alg

# a -> b -> d, a -> c -> d   (diamond) with b on the critical path
DIAMOND = {"a": [], "b": ["a"], "c": ["a"], "d": ["b", "c"]}
DIAMOND_D = {"a": 1.0, "b": 5.0, "c": 2.0, "d": 1.0}


# ---------------------------------------------------------------- model validation
def test_valid_graph_exposes_structure() -> None:
    g = TaskGraph.from_dependencies(DIAMOND)
    assert len(g) == 4
    assert g.num_edges == 4
    assert g.topological_order == ("a", "b", "c", "d")
    assert g.successor_map["a"] == ("b", "c")
    assert g.levels == {"a": 0, "b": 1, "c": 1, "d": 2}
    assert (g.depth, g.max_width) == (3, 2)


@pytest.mark.parametrize(
    ("deps", "message"),
    [
        ({"a": [], "a2": ["zzz"]}, "unknown task 'zzz'"),
        ({"a": ["a"]}, "depends on itself"),
        ({"a": ["c"], "b": ["a"], "c": ["b"]}, "dependency cycle: a -> b -> c -> a"),
    ],
)
def test_invalid_graphs_are_rejected_with_precise_errors(
    deps: dict[str, list[str]], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        TaskGraph.from_dependencies(deps)


def test_duplicate_ids_and_dependencies_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate task ids"):
        TaskGraph(tasks=(Task(id="a"), Task(id="a")))
    with pytest.raises(ValidationError, match="duplicate dependencies"):
        Task(id="b", depends_on=("a", "a"))


@pytest.mark.parametrize("bad_id", ["", "has space", "-leading", "x" * 300])
def test_task_id_pattern(bad_id: str) -> None:
    with pytest.raises(ValidationError):
        Task(id=bad_id)


def test_empty_graph_rejected() -> None:
    with pytest.raises(ValidationError):
        TaskGraph(tasks=())


def test_graph_is_immutable() -> None:
    g = TaskGraph.from_dependencies(DIAMOND)
    with pytest.raises(ValidationError):
        g.tasks = ()  # type: ignore[misc]


def test_plan_limits() -> None:
    g = TaskGraph.from_dependencies(DIAMOND)
    g.check_limits(max_tasks=4, max_depth=3, max_fan_in=2)
    with pytest.raises(PlanLimitError, match="tasks"):
        g.check_limits(max_tasks=3, max_depth=3, max_fan_in=2)
    with pytest.raises(PlanLimitError, match="depth"):
        g.check_limits(max_tasks=4, max_depth=2, max_fan_in=2)
    with pytest.raises(PlanLimitError, match="dependencies"):
        g.check_limits(max_tasks=4, max_depth=3, max_fan_in=1)


# ---------------------------------------------------------------- raw algorithms
def test_topological_order_raises_cycle_error() -> None:
    with pytest.raises(CycleError) as info:
        alg.topological_order({"a": ["b"], "b": ["a"], "c": []})
    assert set(info.value.cycle) == {"a", "b"}


def test_find_cycle_none_for_dag() -> None:
    assert alg.find_cycle(DIAMOND) is None


def test_transitive_reduction_removes_implied_edge() -> None:
    g = TaskGraph.from_dependencies({"a": [], "b": ["a"], "c": ["a", "b"]})
    assert g.transitive_reduction().dependency_map == {"a": (), "b": ("a",), "c": ("b",)}


def test_transitive_reduction_keeps_diamond() -> None:
    g = TaskGraph.from_dependencies(DIAMOND)
    assert g.transitive_reduction().dependency_map == g.dependency_map


# ---------------------------------------------------------------- analysis
def test_critical_path_and_ranks() -> None:
    g = TaskGraph.from_dependencies(DIAMOND)
    assert earliest_finish(g, DIAMOND_D) == {"a": 1.0, "b": 6.0, "c": 3.0, "d": 7.0}
    assert upward_rank(g, DIAMOND_D) == {"a": 7.0, "b": 6.0, "c": 3.0, "d": 1.0}
    cp = critical_path(g, DIAMOND_D)
    assert cp.length == 7.0
    assert cp.tasks == ("a", "b", "d")
    assert slack(g, DIAMOND_D) == {"a": 0.0, "b": 0.0, "c": 3.0, "d": 0.0}


def test_profile_and_bounds() -> None:
    p = profile(TaskGraph.from_dependencies(DIAMOND), DIAMOND_D)
    assert (p.total_work, p.span) == (9.0, 7.0)
    assert p.parallelism == pytest.approx(9 / 7)
    assert p.lower_bound(1) == 9.0
    assert p.lower_bound(2) == 7.0
    assert p.graham_bound(2) == pytest.approx(9 / 2 + 7 / 2)
    assert p.edge_density == pytest.approx(4 / 6)
    with pytest.raises(ValueError, match="workers"):
        p.lower_bound(0)


def test_durations_are_validated() -> None:
    g = TaskGraph.from_dependencies(DIAMOND)
    with pytest.raises(ValueError, match="missing"):
        critical_path(g, {"a": 1.0})
    with pytest.raises(ValueError, match="non-negative"):
        critical_path(g, {**DIAMOND_D, "a": -1.0})
    with pytest.raises(ValueError, match="non-negative"):
        critical_path(g, {**DIAMOND_D, "a": float("nan")})
