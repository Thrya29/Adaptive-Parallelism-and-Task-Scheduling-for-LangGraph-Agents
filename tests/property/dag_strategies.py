"""Hypothesis strategies for random task DAGs with durations."""

from hypothesis import strategies as st

from adaptive_scheduler.graph import TaskGraph

durations_st = st.floats(
    min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False, allow_subnormal=False
)


@st.composite
def dags(draw: st.DrawFn, max_tasks: int = 24, max_fan_in: int = 4) -> TaskGraph:
    """Random DAG. Edges only go from lower to higher index (so it is acyclic), and tasks are
    inserted in a random order so tie-breaking by insertion order is exercised too."""
    n = draw(st.integers(min_value=1, max_value=max_tasks))
    deps: dict[int, list[int]] = {}
    for j in range(n):
        deps[j] = (
            draw(st.lists(st.integers(0, j - 1), unique=True, max_size=min(j, max_fan_in)))
            if j
            else []
        )
    order = draw(st.permutations(range(n)))
    return TaskGraph.from_dependencies({f"t{j}": [f"t{i}" for i in deps[j]] for j in order})


@st.composite
def dags_with_durations(draw: st.DrawFn, max_tasks: int = 24) -> tuple[TaskGraph, dict[str, float]]:
    graph = draw(dags(max_tasks=max_tasks))
    return graph, {tid: draw(durations_st) for tid in graph.ids}
