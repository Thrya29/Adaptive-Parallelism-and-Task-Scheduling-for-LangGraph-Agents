"""The stochastic engine must agree with the deterministic schedulers when randomness is off."""

import math

import numpy as np
from dag_strategies import dags
from hypothesis import given
from hypothesis import strategies as st

from adaptive_scheduler.cost import ContextSize, ProviderProfile, build_profiles, expected_durations
from adaptive_scheduler.graph import TaskGraph
from adaptive_scheduler.scheduling import PriorityPolicy, level_parallel_schedule, list_schedule
from adaptive_scheduler.simulation import ExecutionConfig, Mode, deterministic, prepare, simulate

PROVIDER = deterministic(ProviderProfile())


def _certain_profiles(g: TaskGraph):  # type: ignore[no-untyped-def]
    from dataclasses import replace  # noqa: PLC0415

    return {t: replace(p, success_prob=1.0) for t, p in build_profiles(g, ContextSize()).items()}


@given(dags(max_tasks=20), st.integers(1, 6), st.sampled_from(list(PriorityPolicy)))
def test_engine_matches_list_schedule(g: TaskGraph, k: int, policy: PriorityPolicy) -> None:
    prof = _certain_profiles(g)
    cfg = ExecutionConfig("x", workers=k, policy=policy)
    res = simulate(prepare(g, prof, PROVIDER, cfg), PROVIDER, cfg, np.random.default_rng(0))
    expected = list_schedule(g, expected_durations(prof, PROVIDER), k, policy).makespan
    assert math.isclose(res.makespan, expected, rel_tol=1e-9)
    assert res.success
    assert res.model_calls == len(g)
    assert res.rate_limited_calls == 0


@given(dags(max_tasks=20))
def test_engine_matches_level_parallel_schedule(g: TaskGraph) -> None:
    prof = _certain_profiles(g)
    cfg = ExecutionConfig("static", workers=None, mode=Mode.LEVEL_SYNC)
    res = simulate(prepare(g, prof, PROVIDER, cfg), PROVIDER, cfg, np.random.default_rng(0))
    expected = level_parallel_schedule(g, expected_durations(prof, PROVIDER)).makespan
    assert math.isclose(res.makespan, expected, rel_tol=1e-9)


@given(dags(max_tasks=16), st.integers(0, 10_000), st.none() | st.integers(1, 6))
def test_every_task_is_accounted_for(g: TaskGraph, seed: int, k: int | None) -> None:
    prof = build_profiles(g, ContextSize())
    prov = ProviderProfile(max_concurrency=3)
    cfg = ExecutionConfig("x", workers=k)
    res = simulate(prepare(g, prof, prov, cfg), prov, cfg, np.random.default_rng(seed))
    assert res.completed + res.failed + res.cancelled == len(g)
    assert res.model_calls >= res.completed + res.failed
    assert res.makespan > 0
