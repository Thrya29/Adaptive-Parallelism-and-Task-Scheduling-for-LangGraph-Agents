"""The live-validation harness, exercised offline on the mock LLM."""

from pathlib import Path

import pytest

from adaptive_scheduler.agent import HeuristicPlanner
from adaptive_scheduler.bench.live import LiveConfig, run_live
from adaptive_scheduler.bench.live_replay import load_live, replay, replay_summary, warm_estimator
from adaptive_scheduler.bench.live_report import analyse_live, write_live
from adaptive_scheduler.cost import ProviderProfile
from adaptive_scheduler.cost.provider import MODEL_PRICES_USD_PER_MTOK
from adaptive_scheduler.llm import MockChatModel
from adaptive_scheduler.simulation import RetryPolicy

SCALE = 0.002
REQUESTS = (
    "Research A, B and C, then compare them",
    "Draft a plan, then estimate its cost",
)


async def _run(cfg: LiveConfig):  # type: ignore[no-untyped-def]
    llm = MockChatModel(time_scale=SCALE)
    prior = ProviderProfile()
    return prior, await run_live(
        cfg,
        llm,
        HeuristicPlanner(llm, 16),
        prior,
        retry=RetryPolicy(backoff_base_s=0.001),
        time_scale=SCALE,
    )


async def test_live_harness_end_to_end(tmp_path: Path) -> None:
    cfg = LiveConfig(requests=REQUESTS, reps=2, selector_samples=4, budget_usd=10)
    prior, result = await _run(cfg)
    assert len(result.plans) == 2
    assert len(result.runs) == 2 * 4 * 2  # requests x strategies x reps
    assert {r["calibrated"] for r in result.runs} == {False, True}
    assert all(r["latency_s"] > 0 and r["predicted_mean_s"] > 0 for r in result.runs)
    assert result.calls
    assert result.spent_usd > 0
    assert result.calibration["observations"] > 0

    analysis = analyse_live(result, prior)
    assert {s["strategy"] for s in analysis["strategies"]} == {
        "adaptive",
        "sequential",
        "static_parallel",
        "eager_parallel",
    }
    assert analysis["prediction"]["calibrated_model"]["n"] == 2 * 4  # rep 1 only
    assert analysis["ranking"]["per_request"]
    write_live(result, analysis, tmp_path, "mock")
    for name in ("runs.csv", "calls.csv", "plans.json", "analysis.json", "LIVE_RESULTS.md"):
        assert (tmp_path / name).exists(), name
    assert (
        "## Does the cost model predict real latency?" in (tmp_path / "LIVE_RESULTS.md").read_text()
    )


async def test_budget_guard_stops_and_keeps_partial_results() -> None:
    cfg = LiveConfig(requests=REQUESTS, reps=2, selector_samples=4, budget_usd=1e-9)
    _, result = await _run(cfg)
    assert result.stopped_early is not None
    assert "budget" in result.stopped_early
    assert result.plans  # planning happened and was kept
    assert result.runs == []


def test_live_config_validation() -> None:
    with pytest.raises(ValueError, match="reps"):
        LiveConfig(reps=0)
    with pytest.raises(ValueError, match="budget"):
        LiveConfig(budget_usd=0)


def test_model_prices() -> None:
    p = ProviderProfile.for_model("claude-haiku-5-5", max_concurrency=4)
    assert (p.usd_per_mtok_in, p.usd_per_mtok_out) == MODEL_PRICES_USD_PER_MTOK["claude-haiku-5-5"]
    assert p.max_concurrency == 4
    assert p.cost_usd(1_000_000, 1_000_000) == pytest.approx(0.60)
    unknown = ProviderProfile.for_model("some-other-model")
    assert unknown.usd_per_mtok_in == ProviderProfile().usd_per_mtok_in


async def test_warm_start_and_replay(tmp_path: Path) -> None:
    cfg = LiveConfig(requests=REQUESTS, reps=2, selector_samples=4, budget_usd=10)
    prior, first = await _run(cfg)
    write_live(first, analyse_live(first, prior), tmp_path, "mock")

    loaded = load_live(tmp_path)
    assert len(loaded.runs) == len(first.runs)
    assert isinstance(loaded.runs[0]["calibrated"], bool)

    rows = replay(loaded, prior, samples=4)
    assert len(rows) == len(loaded.runs)
    assert replay_summary(loaded, prior, rows)["calibrated_model"]["n"] > 0

    est = warm_estimator(loaded, prior)
    assert est.observations == len(loaded.calls)
    llm = MockChatModel(time_scale=SCALE)
    second = await run_live(
        LiveConfig(requests=REQUESTS[:1], reps=1, selector_samples=4),
        llm,
        HeuristicPlanner(llm, 16),
        prior,
        retry=RetryPolicy(backoff_base_s=0.001),
        time_scale=SCALE,
        estimator=est,
    )
    assert second.runs
    assert all(r["calibrated"] for r in second.runs)  # warm-started: no uncalibrated runs
