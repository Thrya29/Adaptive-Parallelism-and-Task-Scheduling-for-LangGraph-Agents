"""Invariants that must hold for every DAG, every duration vector and every worker count."""

import itertools
import math

from dag_strategies import dags, dags_with_durations, durations_st
from hypothesis import given, settings
from hypothesis import strategies as st

from adaptive_scheduler.graph import TaskGraph, profile
from adaptive_scheduler.graph import algorithms as alg
from adaptive_scheduler.scheduling import (
    FusionParams,
    PriorityPolicy,
    fuse_chains,
    level_parallel_schedule,
    list_schedule,
    maximal_chains,
    optimal_chain_segmentation,
    sequential_schedule,
    validate_schedule,
)
from adaptive_scheduler.scheduling.fusion import segment_feasible, segment_objective

TOL = 1e-6
settings.register_profile("default", max_examples=300, deadline=None)
settings.load_profile("default")

workers_st = st.integers(min_value=1, max_value=8)
policy_st = st.sampled_from(list(PriorityPolicy))


def _reachable(preds: dict[str, tuple[str, ...]]) -> set[tuple[str, str]]:
    succ = alg.successors_of(preds)
    pairs = set()
    for src in preds:
        stack, seen = list(succ[src]), set()
        while stack:
            v = stack.pop()
            if v not in seen:
                seen.add(v)
                stack.extend(succ[v])
        pairs |= {(src, v) for v in seen}
    return pairs


# ---------------------------------------------------------------- graph algorithms
@given(dags())
def test_topological_order_respects_every_edge(g: TaskGraph) -> None:
    pos = g.topological_index
    assert sorted(pos) == sorted(g.ids)
    for tid, deps in g.dependency_map.items():
        assert all(pos[d] < pos[tid] for d in deps)


@given(dags(max_tasks=16))
def test_transitive_reduction_preserves_reachability_and_is_minimal(g: TaskGraph) -> None:
    reduced = g.transitive_reduction().dependency_map
    assert _reachable(reduced) == _reachable(g.dependency_map)
    for node, deps in reduced.items():  # removing any remaining edge must lose reachability
        for dep in deps:
            pruned = {**reduced, node: tuple(x for x in deps if x != dep)}
            assert (dep, node) not in _reachable(pruned)


# ---------------------------------------------------------------- schedulers
@given(dags_with_durations(), workers_st, policy_st)
def test_list_schedule_is_valid_and_within_theoretical_bounds(
    gd: tuple[TaskGraph, dict[str, float]], k: int, policy: PriorityPolicy
) -> None:
    g, d = gd
    s = list_schedule(g, d, k, policy)
    validate_schedule(s, g, d)
    p = profile(g, d)
    assert s.makespan >= p.lower_bound(k) - TOL
    assert s.makespan <= p.graham_bound(k) + TOL  # Graham (1969) list-scheduling bound
    assert s.makespan <= p.total_work + TOL


@given(dags_with_durations())
def test_enough_workers_achieves_the_critical_path(gd: tuple[TaskGraph, dict[str, float]]) -> None:
    g, d = gd
    s = list_schedule(g, d, len(g))
    assert math.isclose(s.makespan, profile(g, d).span, rel_tol=1e-9, abs_tol=TOL)


@given(dags_with_durations())
def test_sequential_equals_total_work(gd: tuple[TaskGraph, dict[str, float]]) -> None:
    g, d = gd
    s = sequential_schedule(g, d)
    validate_schedule(s, g, d)
    assert math.isclose(s.makespan, profile(g, d).total_work, rel_tol=1e-9, abs_tol=TOL)


@given(dags_with_durations(), st.none() | workers_st)
def test_level_parallel_is_valid_and_never_beats_critical_path(
    gd: tuple[TaskGraph, dict[str, float]], k: int | None
) -> None:
    g, d = gd
    s = level_parallel_schedule(g, d, k)
    validate_schedule(s, g, d)
    assert s.makespan >= profile(g, d).span - TOL


@given(dags_with_durations(), workers_st, policy_st)
def test_schedulers_are_deterministic(
    gd: tuple[TaskGraph, dict[str, float]], k: int, policy: PriorityPolicy
) -> None:
    g, d = gd
    assert list_schedule(g, d, k, policy) == list_schedule(g, d, k, policy)


# ---------------------------------------------------------------- fusion
fusion_params_st = st.builds(
    FusionParams,
    call_overhead=st.floats(0.0, 5.0),
    max_tasks_per_call=st.integers(1, 5),
    max_work_per_call=st.sampled_from([math.inf, 2.0, 10.0, 50.0]),
    penalty_per_extra_task=st.floats(0.0, 3.0),
)


def _compositions(n: int):  # type: ignore[no-untyped-def]
    for cuts in itertools.product([False, True], repeat=n - 1):
        bounds = [0, *(i + 1 for i, c in enumerate(cuts) if c), n]
        yield list(itertools.pairwise(bounds))


@given(st.lists(durations_st, min_size=1, max_size=9), fusion_params_st)
def test_chain_dp_matches_brute_force(durations: list[float], params: FusionParams) -> None:
    plan = optimal_chain_segmentation(durations, params)
    best = min(
        sum(segment_objective(durations[a:b], params) for a, b in segs)
        for segs in _compositions(len(durations))
        if all(segment_feasible(durations[a:b], params) for a, b in segs)
    )
    assert math.isclose(plan.objective, best, rel_tol=1e-9, abs_tol=1e-9)
    assert all(segment_feasible(durations[a:b], params) for a, b in plan.segments)
    assert [i for seg in plan.segments for i in range(*seg)] == list(range(len(durations)))


@given(dags_with_durations(), fusion_params_st, workers_st)
def test_fusion_never_increases_work_or_span(
    gd: tuple[TaskGraph, dict[str, float]], params: FusionParams, k: int
) -> None:
    g, d = gd
    res = fuse_chains(g, d, params)
    members = sorted(t for group in res.groups.values() for t in group)
    assert members == sorted(g.ids)  # fused groups partition the original tasks
    assert len(res.graph) == len(g) - res.calls_saved
    before, after = profile(g, d), profile(res.graph, res.durations)
    assert after.total_work <= before.total_work + TOL
    assert after.span <= before.span + TOL
    validate_schedule(list_schedule(res.graph, res.durations, k), res.graph, res.durations)


@given(dags())
def test_maximal_chains_partition_tasks(g: TaskGraph) -> None:
    chains = maximal_chains(g)
    assert sorted(t for c in chains for t in c) == sorted(g.ids)
    for chain in chains:
        for u, v in itertools.pairwise(chain):
            assert g.successor_map[u] == (v,)
            assert g.dependency_map[v] == (u,)
