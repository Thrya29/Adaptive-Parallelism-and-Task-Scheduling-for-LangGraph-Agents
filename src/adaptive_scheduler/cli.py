"""Command-line entry point (``apsched``)."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence

from adaptive_scheduler import __version__
from adaptive_scheduler.config import ConfigurationError, LLMProvider, Settings


def _cmd_info(settings: Settings) -> int:
    try:
        provider = settings.resolve_provider()
    except ConfigurationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"adaptive-scheduler {__version__}")
    print(f"requested provider : {settings.llm_provider.value}")
    print(f"resolved provider  : {provider.value}")
    if provider is LLMProvider.ANTHROPIC:
        print(f"model              : {settings.anthropic_model}")
    else:
        print("model              : deterministic mock LLM (no API key needed)")
        print(f"mock time scale    : {settings.mock_time_scale} wall-clock s per simulated s")
    print(f"provider max conc. : {settings.max_concurrency}")
    return 0


def _cmd_run(settings: Settings, args: argparse.Namespace) -> int:
    # Imported lazily so `apsched info` stays fast.
    from adaptive_scheduler.agent import AdaptiveAgent, Strategy  # noqa: PLC0415
    from adaptive_scheduler.report import render  # noqa: PLC0415

    try:
        agent = AdaptiveAgent.from_settings(settings)
    except ConfigurationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    is_mock = settings.resolve_provider() is LLMProvider.MOCK
    if is_mock and not args.json:
        print("[mock LLM: no API key configured - responses are simulated]\n")
    result = asyncio.run(agent.arun(args.request, strategy=Strategy(args.strategy)))
    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        print(render(result, max_answer_chars=300 if is_mock else None))
    return 0 if result.metrics.success else 1


def _cmd_bench(args: argparse.Namespace) -> int:
    from dataclasses import replace  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    from adaptive_scheduler.bench import PROFILES, run_benchmark  # noqa: PLC0415

    cfg = replace(PROFILES[args.profile], seed=args.seed)

    def progress(done: int, total: int) -> None:
        if done == total or done % max(1, total // 10) == 0:
            print(f"  {done}/{total} units", file=sys.stderr)

    print(
        f"Running '{args.profile}' benchmark ({args.jobs} process(es)) -> {args.out}",
        file=sys.stderr,
    )
    analysis = run_benchmark(
        cfg, Path(args.out), jobs=args.jobs, plots=not args.no_plots, progress=progress
    )
    h1, sel = analysis["h1"], analysis["selection"]
    print(f"H1: adaptive within 5% of the best fixed strategy in {h1['within_5pct_rate']:.0%}")
    print(f"Selector picked the hindsight-best configuration in {sel['exact_rate']:.0%} of units")
    print(f"Report: {Path(args.out) / 'RESULTS.md'}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn  # noqa: PLC0415

    uvicorn.run(
        "adaptive_scheduler.api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        log_config=None,  # the app configures structured logging itself
    )
    return 0


def _cmd_live(settings: Settings, args: argparse.Namespace) -> int:
    from dataclasses import replace  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    from adaptive_scheduler.agent.planning import HeuristicPlanner, LLMPlanner  # noqa: PLC0415
    from adaptive_scheduler.bench.live import LiveConfig, run_live  # noqa: PLC0415
    from adaptive_scheduler.bench.live_report import analyse_live, write_live  # noqa: PLC0415
    from adaptive_scheduler.cost.provider import ProviderProfile  # noqa: PLC0415
    from adaptive_scheduler.llm import MockChatModel, create_chat_model  # noqa: PLC0415

    try:
        provider = settings.resolve_provider()
    except ConfigurationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    cfg = LiveConfig(reps=args.reps, budget_usd=args.budget)
    if args.requests:
        cfg = replace(cfg, requests=cfg.requests[: args.requests])
    llm = create_chat_model(settings)
    if provider is LLMProvider.ANTHROPIC:
        model, scale = settings.anthropic_model, 1.0
        prior = ProviderProfile.for_model(model, max_concurrency=settings.max_concurrency)
        planner = LLMPlanner(llm, cfg.max_plan_tasks)
    else:
        model, scale = "mock", settings.mock_time_scale
        llm = (
            llm.model_copy(update={"time_scale": scale}) if isinstance(llm, MockChatModel) else llm
        )
        prior = ProviderProfile(max_concurrency=settings.max_concurrency)
        planner = HeuristicPlanner(llm, cfg.max_plan_tasks)  # type: ignore[assignment]
    print(
        f"Live validation on {model}: {len(cfg.requests)} requests x {len(cfg.strategies)} "
        f"strategies x {cfg.reps} reps, budget ${cfg.budget_usd:.2f}",
        file=sys.stderr,
    )
    estimator = None
    if args.warm_start:
        from adaptive_scheduler.bench.live_replay import load_live, warm_estimator  # noqa: PLC0415

        estimator = warm_estimator(load_live(Path(args.warm_start)), prior)
        print(
            f"Warm-started estimator from {estimator.observations} recorded calls", file=sys.stderr
        )
    result = asyncio.run(
        run_live(
            cfg,
            llm,
            planner,
            prior,
            time_scale=scale,
            estimator=estimator,
            progress=lambda m: print("  " + m, file=sys.stderr),
        )
    )
    analysis = analyse_live(result, prior)
    write_live(result, analysis, Path(args.out), model)
    print(f"Spent ~${result.spent_usd:.4f}; report: {Path(args.out) / 'LIVE_RESULTS.md'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="apsched", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--provider",
        choices=[p.value for p in LLMProvider],
        help="override APSCHED_LLM_PROVIDER for this invocation",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("info", help="show resolved configuration")

    run = sub.add_parser("run", help="plan, schedule and execute a request")
    run.add_argument("request", help="the request to solve")
    run.add_argument(
        "--strategy",
        default="adaptive",
        choices=["adaptive", "sequential", "static_parallel", "eager_parallel"],
        help="execution strategy (default: adaptive)",
    )
    run.add_argument("--json", action="store_true", help="print the full result as JSON")

    bench = sub.add_parser("bench", help="run the benchmark suite (simulation; no API key needed)")
    bench.add_argument("--profile", choices=["quick", "full"], default="quick")
    bench.add_argument("--out", default="docs/results", help="output directory")
    bench.add_argument("--jobs", type=int, default=1, help="worker processes")
    bench.add_argument("--seed", type=int, default=0)
    bench.add_argument("--no-plots", action="store_true", help="skip matplotlib figures")

    live = sub.add_parser("live-bench", help="validate the scheduler against the configured LLM")
    live.add_argument("--reps", type=int, default=3)
    live.add_argument(
        "--budget", type=float, default=2.0, help="stop once spend reaches this (USD)"
    )
    live.add_argument("--requests", type=int, default=0, help="use only the first N requests")
    live.add_argument("--out", default="docs/results/live")
    live.add_argument(
        "--warm-start", help="calibrate first on the calls recorded in this directory"
    )

    serve = sub.add_parser("serve", help="run the HTTP service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings()
    if args.provider:
        settings = settings.model_copy(update={"llm_provider": LLMProvider(args.provider)})

    if args.command == "info":
        return _cmd_info(settings)
    if args.command == "run":
        return _cmd_run(settings, args)
    if args.command == "bench":
        return _cmd_bench(args)
    if args.command == "live-bench":
        return _cmd_live(settings, args)
    if args.command == "serve":  # pragma: no cover - starts a blocking server
        return _cmd_serve(args)
    return 1  # pragma: no cover - argparse enforces a valid command


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
