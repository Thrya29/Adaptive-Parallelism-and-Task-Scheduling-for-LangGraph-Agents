import numpy as np
import pytest

from adaptive_scheduler.cost import (
    ContextSize,
    CostEstimator,
    ProviderProfile,
    TaskProfile,
    build_profiles,
    estimate_tokens,
    expected_durations,
)
from adaptive_scheduler.graph import Task, TaskGraph, TaskType


def test_latency_formula_and_price() -> None:
    p = ProviderProfile(
        ttft_s=0.5, prefill_tps=1000, decode_tps=50, tail_prob=0.1, tail_multiplier=3
    )
    assert p.base_latency(1000, 100) == pytest.approx(0.5 + 1 + 2)
    assert p.expected_latency(1000, 100) == pytest.approx(3.5 * 1.2)
    assert p.cost_usd(1_000_000, 200_000) == pytest.approx(1.0 + 1.0)


def test_sampled_latency_is_unbiased_and_contention_slows_calls() -> None:
    p = ProviderProfile()
    rng = np.random.default_rng(0)
    samples = [p.sample_latency(rng, 2000, 300) for _ in range(40_000)]
    assert np.mean(samples) == pytest.approx(p.expected_latency(2000, 300), rel=0.02)
    rng = np.random.default_rng(1)
    loaded = [p.sample_latency(rng, 2000, 300, active_calls=11) for _ in range(40_000)]
    assert np.mean(loaded) == pytest.approx(np.mean(samples) * (1 + 10 * p.contention), rel=0.03)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"decode_tps": 0},
        {"ttft_s": -1},
        {"tail_prob": 1.5},
        {"max_concurrency": 0},
        {"fusion_quality_decay": 0},
        {"noise_sigma": float("nan")},
    ],
)
def test_provider_validation(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        ProviderProfile(**kwargs)  # type: ignore[arg-type]


def test_profiles_carry_dependency_outputs_into_prompts() -> None:
    g = TaskGraph(
        tasks=(
            Task(id="a", instruction="x" * 40, task_type=TaskType.RETRIEVAL),
            Task(id="b", instruction="x" * 40, task_type=TaskType.RETRIEVAL),
            Task(id="c", instruction="", task_type=TaskType.SYNTHESIS, depends_on=("a", "b")),
        )
    )
    ctx = ContextSize(system_tokens=100, shared_context_tokens=50)
    prof = build_profiles(g, ctx)
    assert prof["a"].input_tokens == 150 + 10
    assert prof["c"].input_tokens == 150 + 1 + 2 * 300  # both retrieval outputs flow into c
    assert prof["c"].output_tokens == 500
    d = expected_durations(prof, ProviderProfile())
    assert d["c"] > d["a"]


def test_estimate_tokens() -> None:
    assert estimate_tokens("") == 1
    assert estimate_tokens("abcd" * 10) == 10


def test_task_profile_validation() -> None:
    with pytest.raises(ValueError, match="success_prob"):
        TaskProfile(TaskType.REASONING, 1, 1, 0.0)
    with pytest.raises(ValueError, match="token"):
        TaskProfile(TaskType.REASONING, -1, 1, 0.5)


def test_estimator_recovers_mean_latency() -> None:
    """After calibration, predicted mean latency matches the true provider's mean latency."""
    true = ProviderProfile(ttft_s=1.2, decode_tps=40.0)
    est = CostEstimator(prior=ProviderProfile())
    assert est.provider() == est.prior  # no data yet -> prior
    rng = np.random.default_rng(0)
    for _ in range(4000):
        out = float(rng.uniform(50, 800))
        lat = true.sample_latency(rng, 1500, out)
        est.observe(
            TaskType.REASONING, latency_s=lat, input_tokens=1500, output_tokens=out, succeeded=True
        )
    fitted = est.provider()
    for out in (50, 300, 800):
        assert fitted.expected_latency(1500, out) == pytest.approx(
            true.expected_latency(1500, out), rel=0.05
        )
    assert fitted.tail_prob < 0.01  # the assumed tail is replaced by the measured residual spread
    assert 0.2 < fitted.noise_sigma < 0.7


def test_poor_prior_fades_with_evidence() -> None:
    """A prior 2.5x too slow must not keep weight forever (regression test for the live run)."""
    true = ProviderProfile(ttft_s=0.7, decode_tps=200.0, noise_sigma=0.1, tail_prob=0.0)
    est = CostEstimator(prior=ProviderProfile(decode_tps=80.0))
    rng = np.random.default_rng(1)
    for _ in range(500):
        out = float(rng.uniform(500, 4000))
        est.observe(
            TaskType.REASONING,
            latency_s=true.sample_latency(rng, 1000, out),
            input_tokens=1000,
            output_tokens=out,
            succeeded=True,
        )
    assert est.provider().expected_latency(1000, 2000) == pytest.approx(
        true.expected_latency(1000, 2000), rel=0.05
    )


def test_estimator_tracks_tokens_and_success() -> None:
    est = CostEstimator(alpha=0.5)
    rng = np.random.default_rng(2)
    for _ in range(200):
        out = float(rng.lognormal(np.log(2000), 0.6))
        est.observe(
            TaskType.TOOL_CALL, latency_s=1.0, input_tokens=10, output_tokens=out, succeeded=False
        )
    # Decayed mean over a ~50-call window of a lognormal with mean 2000 * exp(0.18) ~ 2394.
    assert est.output_token_estimates()[TaskType.TOOL_CALL] == pytest.approx(2394, rel=0.2)
    assert est.provider().output_tokens_sigma == pytest.approx(0.6, rel=0.2)
    assert est.success_estimates()[TaskType.TOOL_CALL] < 0.01
    assert est.success_estimates()[TaskType.RETRIEVAL] == 0.98  # untouched prior
    assert est.observations == 200


def test_single_long_output_cannot_swing_estimate() -> None:
    est = CostEstimator()
    est.observe(
        TaskType.RETRIEVAL, latency_s=60, input_tokens=10, output_tokens=6000, succeeded=True
    )
    # (3 * 300 + 6000) / (3 + 1) = 1725: moves, but not all the way to one extreme sample.
    assert est.output_token_estimates()[TaskType.RETRIEVAL] == pytest.approx(1725)


def test_estimator_validation() -> None:
    with pytest.raises(ValueError, match="alpha"):
        CostEstimator(alpha=0)
    with pytest.raises(ValueError, match="decay"):
        CostEstimator(decay=0)
    with pytest.raises(ValueError, match="prior"):
        CostEstimator(prior_strength=-1)
