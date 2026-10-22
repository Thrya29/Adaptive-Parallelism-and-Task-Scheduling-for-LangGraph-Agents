from dataclasses import replace

import numpy as np
import pytest

from adaptive_scheduler.cost import ContextSize, ProviderProfile, TaskProfile, build_profiles
from adaptive_scheduler.graph import TaskGraph, TaskType
from adaptive_scheduler.simulation import (
    ExecutionConfig,
    Mode,
    Objective,
    RetryPolicy,
    deterministic,
    predict,
    prepare,
    select_strategy,
    simulate,
)

CTX = ContextSize()


def _profiles(graph: TaskGraph, success: float = 1.0) -> dict[str, TaskProfile]:
    return {t: replace(p, success_prob=success) for t, p in build_profiles(graph, CTX).items()}


def _run(graph, profiles, provider, config, seed=0):  # type: ignore[no-untyped-def]
    plan = prepare(graph, profiles, provider, config, CTX.per_call)
    return simulate(plan, provider, config, np.random.default_rng(seed))


def fan_out(n: int) -> TaskGraph:
    deps: dict[str, list[str]] = {f"w{i}": [] for i in range(n)}
    deps["merge"] = list(deps)
    return TaskGraph.from_dependencies(deps)


def chain(n: int) -> TaskGraph:
    return TaskGraph.from_dependencies({f"s{i}": [f"s{i - 1}"] if i else [] for i in range(n)})


def test_runs_are_reproducible_per_seed() -> None:
    g, prov = fan_out(6), ProviderProfile()
    cfg = ExecutionConfig("k4", workers=4)
    assert _run(g, _profiles(g, 0.9), prov, cfg, 7) == _run(g, _profiles(g, 0.9), prov, cfg, 7)
    assert _run(g, _profiles(g, 0.9), prov, cfg, 7) != _run(g, _profiles(g, 0.9), prov, cfg, 8)


def test_token_accounting_matches_profiles() -> None:
    g = fan_out(3)
    prof = _profiles(g)
    res = _run(g, prof, deterministic(ProviderProfile()), ExecutionConfig("seq", workers=1))
    assert res.success
    assert res.completed == 4
    assert res.model_calls == 4
    assert res.input_tokens == pytest.approx(sum(p.input_tokens for p in prof.values()))
    assert res.output_tokens == pytest.approx(sum(p.output_tokens for p in prof.values()))
    assert res.cost_usd == pytest.approx(
        ProviderProfile().cost_usd(res.input_tokens, res.output_tokens)
    )


def test_failure_cancels_dependents_and_counts_retries() -> None:
    g = TaskGraph.from_dependencies({"a": [], "b": ["a"], "x": []})
    prof = _profiles(g)
    prof["a"] = replace(prof["a"], success_prob=1e-9)
    prov = deterministic(ProviderProfile())
    res = _run(g, prof, prov, ExecutionConfig("k2", workers=2))
    assert (res.completed, res.failed, res.cancelled) == (1, 1, 1)
    assert not res.success
    assert res.model_calls == 1 + 3  # x once, a with 2 retries


def test_provider_cap_causes_rate_limiting_only_under_blind_fan_out() -> None:
    g = fan_out(12)
    prov = replace(ProviderProfile(), max_concurrency=4)
    prof = _profiles(g)
    blind = _run(g, prof, prov, ExecutionConfig("unbounded", workers=None))
    capped = _run(g, prof, prov, ExecutionConfig("k4", workers=4))
    assert blind.rate_limited_calls > 0
    assert capped.rate_limited_calls == 0


def test_level_sync_waits_at_barriers() -> None:
    g = TaskGraph.from_dependencies({"a": [], "b": [], "c": ["a"], "e": ["c"], "d": ["b"]})
    prof = _profiles(g)
    prof["b"] = replace(prof["b"], output_tokens=2000)
    prov = deterministic(ProviderProfile())
    static = _run(g, prof, prov, ExecutionConfig("static", mode=Mode.LEVEL_SYNC))
    greedy = _run(g, prof, prov, ExecutionConfig("greedy", workers=None))
    assert static.makespan > greedy.makespan


def test_fusion_reduces_calls_and_tokens_on_chains() -> None:
    g = chain(4)
    prof = _profiles(g)
    prov = deterministic(ProviderProfile())
    plain = _run(g, prof, prov, ExecutionConfig("plain", workers=1))
    fused = _run(g, prof, prov, ExecutionConfig("fused", workers=1, fuse=True))
    assert fused.model_calls < plain.model_calls
    assert fused.input_tokens < plain.input_tokens
    assert fused.makespan < plain.makespan
    assert fused.output_tokens == pytest.approx(plain.output_tokens)


def test_retry_backoff_is_capped_and_jittered() -> None:
    rp = RetryPolicy(backoff_base_s=1.0, backoff_cap_s=4.0)
    rng = np.random.default_rng(0)
    delays = [rp.backoff(10, rng) for _ in range(100)]
    assert all(2.0 <= d < 6.0 for d in delays)
    assert len(set(delays)) > 1


def test_invalid_worker_config() -> None:
    with pytest.raises(ValueError, match="workers"):
        ExecutionConfig("bad", workers=0)


# ---------------------------------------------------------------- selector
def test_selector_runs_chains_sequentially_and_fuses_them() -> None:
    g = chain(6)
    sel = select_strategy(
        g, _profiles(g, 0.99), ProviderProfile(), samples=32, per_call_context_tokens=CTX.per_call
    )
    assert sel.chosen.config.workers == 1  # no parallelism to exploit
    assert sel.chosen.config.fuse  # fewer calls, fewer tokens, lower latency


def test_selector_parallelizes_wide_graphs_within_provider_limits() -> None:
    g = fan_out(16)
    prov = replace(ProviderProfile(), max_concurrency=8)
    sel = select_strategy(
        g, _profiles(g, 0.99), prov, samples=32, per_call_context_tokens=CTX.per_call
    )
    k = sel.chosen.config.workers
    assert k is not None
    assert 4 <= k <= 8
    seq = next(p for p in sel.candidates if p.config.workers == 1 and not p.config.fuse)
    assert sel.chosen.mean_latency < seq.mean_latency / 2.5


def test_selector_respects_success_constraint() -> None:
    # Fusion packs many steps into one call; with a harsh quality decay it must not be chosen.
    g = chain(4)
    prov = replace(ProviderProfile(), fusion_quality_decay=0.5)
    sel = select_strategy(
        g, _profiles(g, 0.99), prov, samples=64, per_call_context_tokens=CTX.per_call
    )
    assert not sel.chosen.config.fuse


def test_predict_reports_all_metrics() -> None:
    g = fan_out(4)
    pred = predict(
        g,
        _profiles(g),
        ProviderProfile(),
        ExecutionConfig("k2", workers=2),
        samples=16,
        seed=0,
        objective=Objective(),
    )
    assert pred.p95_latency >= pred.mean_latency * 0.9
    assert pred.mean_calls >= 5
    assert pred.mean_cost_usd > 0
    assert 0 <= pred.success_rate <= 1
    with pytest.raises(ValueError, match="samples"):
        select_strategy(g, _profiles(g), ProviderProfile(), samples=0)


def test_candidate_workers_never_exceed_graph_width() -> None:
    g = TaskGraph.from_dependencies({"a": [], "b": [], "c": ["a", "b"]})
    sel = select_strategy(g, _profiles(g), ProviderProfile(), samples=4)
    assert {p.config.workers for p in sel.candidates} == {1, 2}
    assert sel.selection_time_s >= 0
    _ = TaskType  # re-exported for readers of this module
