"""Record a real run from a running service, for the dashboard's replay mode and the docs media.

Usage (with the service running, e.g. ``apsched serve --port 8080``)::

    python scripts/record_demo.py --url http://localhost:8080 --out docs/media/demo_run.json

It streams one adaptive run (every lifecycle event, with server timestamps) and then runs the same
request under all four strategies to record their execution times for the comparison chart.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import httpx

REQUEST = (
    "Research the EV markets of India, China and Germany, then compare their growth and risks, "
    "then write a short investment summary"
)
STRATEGIES = ("sequential", "static_parallel", "eager_parallel", "adaptive")


def stream(
    client: httpx.Client, request: str, strategy: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    events: list[dict[str, Any]] = []
    result: dict[str, Any] = {}
    with client.stream(
        "POST", "/v1/runs/stream", json={"request": request, "strategy": strategy}
    ) as r:
        r.raise_for_status()
        name = None
        for line in r.iter_lines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: ") and name:
                data = json.loads(line[6:])
                if name == "result":
                    result = data
                elif name == "error":
                    raise RuntimeError(data)
                else:
                    events.append(data)
    if events:
        events.append({"type": "result", "ts": events[-1]["ts"], "data": result})
    return events, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--out", default="docs/media/demo_run.json")
    parser.add_argument("--request", default=REQUEST)
    args = parser.parse_args()

    with httpx.Client(base_url=args.url, timeout=600) as client:
        info = client.get("/v1/info").json()
        events, result = stream(client, args.request, "adaptive")
        print(
            f"recorded adaptive run: {len(events)} events, {result['metrics']['wall_clock_s']:.1f}s"
        )
        compare = []
        for strategy in STRATEGIES:
            _, r = stream(client, args.request, strategy)
            compare.append({"strategy": strategy, "latency": r["metrics"]["execution_s"]})
            print(f"  {strategy:16s} {r['metrics']['execution_s']:6.1f}s")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"info": info, "request": args.request, "strategy": "adaptive",
               "events": events, "compare": compare}  # fmt: skip
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
