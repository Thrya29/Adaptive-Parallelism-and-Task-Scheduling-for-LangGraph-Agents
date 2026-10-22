"""Command-line entry point (``apsched``)."""

from __future__ import annotations

import argparse
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
    print(f"max concurrency    : {settings.max_concurrency}")
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings()
    if args.provider:
        settings = settings.model_copy(update={"llm_provider": LLMProvider(args.provider)})

    if args.command == "info":
        return _cmd_info(settings)
    return 1  # pragma: no cover - argparse enforces a valid command


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
