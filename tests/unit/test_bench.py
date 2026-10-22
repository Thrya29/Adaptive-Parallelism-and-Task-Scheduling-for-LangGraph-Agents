import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from adaptive_scheduler.bench import BenchConfig, run_benchmark
from adaptive_scheduler.bench.conditions import CONDITIONS, blind_variant
from adaptive_scheduler.bench.stats import (
    bootstrap_ci,
    holm,
    paired_latencies,
    paired_ratio_ci,
    wilcoxon_p,
)
from adaptive_scheduler.bench.workloads import CONTEXT, FAMILIES, generate, stable_seed, suite
from adaptive_scheduler.cost import ProviderProfile, expected_durations
from adaptive_scheduler.graph import TaskType, profile

TINY = BenchConfig(
    families=("chain", "fan_out"),
    instances=1,
    conditions=("baseline",),
    eval_runs=3,
    oracle_runs=3,
    selector_samples=4,
)


# ---------------------------------------------------------------------------- workloads
@pytest.mark.parametrize("family", [*FAMILIES, "realistic"])
def test_every_family_generates_valid_workloads(family: str) -> None:
    for i in range(3):
        w = generate(family, i)
        assert len(w.graph) >= 1
        assert set(w.profiles) == set(w.graph.ids)
        assert generate(family, i) == w  # deterministic


def test_family_shapes_span_the_parallelism_range() -> None:
    def ws(family: str) -> float:
        w = generate(family, 1)
        return profile(w.graph, expected_durations(w.profiles, ProviderProfile())).parallelism

    assert ws("chain") == pytest.approx(1.0)
    assert ws("fan_out") > 3
    assert 1 < ws("fork_join") < ws("fan_out")


def test_task_types_follow_graph_position() -> None:
    g = generate("fan_out", 0).graph
    assert g.task("m0").task_type is TaskType.RETRIEVAL
    assert g.task("reduce").task_type is TaskType.SYNTHESIS


def test_unknown_family_and_stable_seed() -> None:
    with pytest.raises(ValueError, match="unknown"):
        generate("nope", 0)
    assert stable_seed("a", 1) == stable_seed("a", 1) != stable_seed("a", 2)
    assert len(suite(("chain",), instances=2)) == 2


# ---------------------------------------------------------------------------- conditions
def test_blind_variant_drops_dependencies_and_adds_aggregator() -> None:
    w = generate("chain", 0)
    bv = blind_variant(w.graph, w.profiles, CONTEXT.per_call)
    assert bv.blind_tasks == len(w.graph) - 1
    assert all(not bv.graph.task(t).depends_on for t in w.graph.ids)
    assert set(bv.graph.task("decision_agent").depends_on) == set(w.graph.ids)
    for t in w.graph.ids[1:]:  # upstream outputs are no longer in the prompt
        assert bv.profiles[t].input_tokens < w.profiles[t].input_tokens


def test_failure_multiplier_scales_failure_probability() -> None:
    w = generate("chain", 0)
    adjusted = CONDITIONS["unreliable"].adjust(w.profiles)
    for t, p in w.profiles.items():
        assert 1 - adjusted[t].success_prob == pytest.approx(4 * (1 - p.success_prob))
    assert CONDITIONS["baseline"].adjust(w.profiles) is w.profiles
    assert CONDITIONS["misestimated"].beliefs != CONDITIONS["misestimated"].true_provider


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="unknown conditions"):
        BenchConfig(conditions=("nope",))
    with pytest.raises(ValueError, match="blind_quality"):
        BenchConfig(blind_quality=0)
    with pytest.raises(ValueError, match="runs"):
        BenchConfig(eval_runs=1)


# ---------------------------------------------------------------------------- statistics
def test_holm_matches_hand_computation() -> None:
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5]) == [0.5]
    assert holm([0.9, 0.8]) == pytest.approx([1.0, 1.0])


def test_bootstrap_and_paired_ratio() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(10, 1, 500)
    lo, hi = bootstrap_ci(x)
    assert lo < 10 < hi
    assert np.isnan(bootstrap_ci([])[0])
    point, lo, hi = paired_ratio_ci(2 * x, x)
    assert point == pytest.approx(2.0)
    assert lo == pytest.approx(2.0)
    assert hi == pytest.approx(2.0)
    with pytest.raises(ValueError, match="paired"):
        paired_ratio_ci([1.0], [1.0, 2.0])


def test_wilcoxon() -> None:
    assert wilcoxon_p([1, 2, 3], [1, 2, 3]) == 1.0
    a = np.arange(30.0)
    assert wilcoxon_p(a + 1, a) < 0.001


def test_paired_latencies_align_on_seed() -> None:
    rows = [
        {"workload": "w", "condition": "c", "seed_index": i, "strategy": s, "latency_s": v,
         "family": "f"}
        for i in range(3)
        for s, v in (("a", 1.0 + i), ("b", 10.0 + i))
    ]  # fmt: skip
    a, b = paired_latencies(rows, "a", "b")
    assert a == [1.0, 2.0, 3.0]
    assert b == [10.0, 11.0, 12.0]


# ---------------------------------------------------------------------------- end to end
def test_run_benchmark_writes_report(tmp_path: Path) -> None:
    analysis = run_benchmark(TINY, tmp_path, plots=True)
    for name in ("runs.csv.gz", "selections.csv", "blind_sensitivity.csv", "analysis.json",
                 "RESULTS.md", "speedup_by_family.png", "p95_by_condition.png"):  # fmt: skip
        assert (tmp_path / name).exists(), name
    with gzip.open(tmp_path / "runs.csv.gz", "rt") as fh:
        rows = fh.read().splitlines()
    strategies = {"sequential", "static_parallel", "eager_parallel", "blind_parallel",
                  "adaptive", "adaptive_fifo", "adaptive_nofusion", "fixed_k_cap"}  # fmt: skip
    assert len(rows) == 1 + 2 * len(strategies) * TINY.eval_runs
    assert json.loads((tmp_path / "analysis.json").read_text())["h1"]["cells"]
    assert 0 <= analysis["selection"]["exact_rate"] <= 1
    report = (tmp_path / "RESULTS.md").read_text()
    assert "## H1" in report
    assert "## Selector accuracy" in report


def test_cli_bench_quick(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from adaptive_scheduler.cli import main  # noqa: PLC0415

    assert main(["bench", "--profile", "quick", "--no-plots", "--out", str(tmp_path)]) == 0
    assert "Selector picked" in capsys.readouterr().out
    assert (tmp_path / "RESULTS.md").exists()
