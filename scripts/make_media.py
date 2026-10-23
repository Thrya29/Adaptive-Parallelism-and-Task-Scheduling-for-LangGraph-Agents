"""Build the documentation media from the real system.

Produces, in ``docs/media/``:

* ``screenshots/*.png``   - real dashboard, CLI, API docs and metrics views (README / user manual)
* ``demo.mp4`` and ``demo.gif`` - a ~20 s clip of the dashboard replaying a real run (README)
* ``overview.mp4``        - a ~80 s captioned walkthrough at 1920x1080 for sharing (LinkedIn etc.)

Every visual is captured from the running software: the dashboard is the real page replaying a
recorded run (``docs/media/demo_run.json``, made by ``scripts/record_demo.py``), code is rendered
from the source files, and terminal views are the actual output of the commands shown.

Usage::

    pip install -e ".[media]"
    python scripts/make_media.py                       # starts a temporary mock service itself
    python scripts/make_media.py --service-url http://localhost:8080   # or capture a running one
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from media_lib import (
    ACCENT,
    BG,
    FAIL,
    INK,
    INK2,
    INK3,
    OK,
    ORANGE,
    PANEL,
    ROOT,
    YELLOW,
    H,
    Video,
    W,
    canvas,
    caption,
    code_render,
    fit,
    font,
    gif,
    page_url,
    place,
    replay_page,
    screenshot,
    terminal_render,
    window,
)  # fmt: skip

MEDIA = ROOT / "docs" / "media"
SHOTS = MEDIA / "screenshots"
BUILD = ROOT / "build" / "media"
REPO_URL = "github.com/Thrya29/Adaptive-Parallelism-and-Task-Scheduling-for-LangGraph-Agents"
EV_REQUEST = (
    "Research the EV markets of India, China and Germany, then compare their growth and risks, "
    "then write a short investment summary"
)
APSCHED = str(Path(sys.executable).with_name("apsched.exe" if os.name == "nt" else "apsched"))


# ---------------------------------------------------------------------------- live captures
def run_cmd(args: list[str], env: dict[str, str] | None = None) -> str:
    out = subprocess.run(args, capture_output=True, text=True, env={**os.environ, **(env or {})},
                         cwd=ROOT, timeout=900, check=False)  # fmt: skip
    return out.stdout + out.stderr


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@contextlib.contextmanager
def service(url: str | None) -> Iterator[str]:
    """Use a running service, or start a temporary mock one and populate it with a few runs."""
    if url:
        yield url
        return
    port = free_port()
    env = {**os.environ, "APSCHED_LLM_PROVIDER": "mock", "APSCHED_MOCK_TIME_SCALE": "0.02",
           "APSCHED_LOG_FORMAT": "text"}  # fmt: skip
    proc = subprocess.Popen([APSCHED, "serve", "--port", str(port)], env=env, cwd=ROOT,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # fmt: skip
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(60):
            with contextlib.suppress(httpx.HTTPError):
                if httpx.get(f"{base}/readyz").status_code == 200:
                    break
            time.sleep(0.5)
        with httpx.Client(base_url=base, timeout=120) as c:
            for strategy in (
                "adaptive",
                "sequential",
                "static_parallel",
                "eager_parallel",
                "adaptive",
            ):
                c.post("/v1/runs", json={"request": EV_REQUEST, "strategy": strategy})
        yield base
    finally:
        proc.terminate()
        proc.wait(timeout=30)


def terminal_texts(base: str) -> dict[str, str]:
    mock = {"APSCHED_LLM_PROVIDER": "mock", "APSCHED_MOCK_TIME_SCALE": "0.05"}
    info = "$ apsched info\n" + run_cmd([APSCHED, "info"], mock)
    run = run_cmd([APSCHED, "run", EV_REQUEST], mock)
    run_lines = run.splitlines()
    cut = next(
        (i for i, line in enumerate(run_lines) if line.startswith("Answer:")), len(run_lines)
    )
    run = f'$ apsched run "{EV_REQUEST[:62]}..."\n' + "\n".join(run_lines[:cut]).rstrip()
    metrics = httpx.get(f"{base}/metrics").text
    keep = ("apsched_runs_total", "apsched_runs_in_flight", "apsched_tasks_total",
            "apsched_model_calls_total", "apsched_tokens_total", "apsched_decisions_total")  # fmt: skip
    metric_lines = [line for line in metrics.splitlines() if line.startswith(keep)]
    metrics_txt = "$ curl -s localhost:8080/metrics | grep apsched_\n" + "\n".join(metric_lines)
    start = (
        "# first time\n"
        "$ git clone https://" + REPO_URL + ".git\n"
        "$ cd Adaptive-Parallelism-and-Task-Scheduling-for-LangGraph-Agents\n"
        "$ python -m venv .venv && .venv\\Scripts\\activate\n"
        '$ pip install -e ".[api,observability,anthropic]"\n'
        "\n# every day\n"
        "$ apsched serve            # dashboard at http://localhost:8080\n"
        '$ apsched run "your request"\n'
    )
    return {"info": info, "run": run, "metrics": metrics_txt, "start": start}


# ---------------------------------------------------------------------------- dashboard frames
class Dashboard:
    def __init__(self, demo: dict) -> None:
        self.demo = demo
        self.page = replay_page(demo, BUILD / "replay.html")
        self.t_end = demo["events"][-1]["ts"] - demo["events"][0]["ts"]

    def at(self, t: float, size: tuple[int, int] = (W, H)) -> Image.Image:
        """The dashboard frozen at ``t`` seconds into the run (cached on disk, not in memory)."""
        t = round(t, 2)
        out = BUILD / "frames" / f"dash_{t:07.2f}_{size[0]}x{size[1]}.png"
        if out.exists():
            return Image.open(out).convert("RGB")
        return screenshot(page_url(self.page, theme="dark", at=t), out, size)

    def event_time(self, kind: str, nth: int = 0) -> float:
        t0 = self.demo["events"][0]["ts"]
        return [e["ts"] - t0 for e in self.demo["events"] if e["type"] == kind][nth]


def run_frames(
    dash: Dashboard, duration: float, with_captions: bool = True
) -> Iterator[tuple[Image.Image, float]]:
    """The recorded run, time-compressed into ``duration`` seconds, with phase captions."""
    plan_t, decide_t = dash.event_time("plan_ready"), dash.event_time("decision")
    first_done = dash.event_time("task_finished")
    last_start = dash.event_time("task_started", -1)
    end = dash.t_end
    decision = next(e for e in dash.demo["events"] if e["type"] == "decision")["data"]
    n_tasks = len(
        next(e for e in dash.demo["events"] if e["type"] == "plan_ready")["data"]["tasks"]
    )
    fused = " — the last two dependent steps merged into one call" if decision["fuse"] else ""
    phases = [
        (0.0, plan_t, f"The LLM plans the request: {n_tasks} tasks with dependencies", None),
        (plan_t, decide_t + 0.01, "The scheduler simulates every candidate configuration",
         f"and picks {decision['config_name']}: up to {decision['workers']} tasks at once{fused}"),
        (decide_t + 0.01, first_done, "Independent tasks run at the same time",
         "three research calls in parallel on real Claude Haiku"),
        (first_done, last_start + 0.5, "Dependent steps wait for their inputs",
         "a task starts the moment everything it needs has finished"),
        (last_start + 0.5, end, "Each call is tracked live", "latency, tokens and status, task by task"),
    ]  # fmt: skip
    total_t = end
    samples = max(12, int(duration * 3))
    # Hold on the plan and the decision a little longer: they are the key moments.
    holds = {round(plan_t + 0.02, 2): 1.6, round(decide_t + 0.05, 2): 2.6}
    times = sorted({round(total_t * i / (samples - 1), 2) for i in range(samples)} | set(holds))
    per = duration / len(times)
    for t in times:
        img = dash.at(t)
        if with_captions:
            title, sub = next(((a, b) for lo, hi, a, b in phases if lo <= t < hi), phases[-1][2:])
            img = caption(img, title, sub)
        yield img, holds.get(t, per)
    final = dash.at(end + 1)
    m = next(e for e in dash.demo["events"] if e["type"] == "run_finished")["data"]["metrics"]
    if with_captions:
        final = caption(final, f"Done: {m['wall_clock_s']:.0f} s, {m['model_calls']} model calls, all tasks succeeded",
                        "the full answer streams back to the user")  # fmt: skip
    yield final, 2.6


def _last_panel(img: Image.Image) -> Image.Image:
    """Crop the last panel on the page (the strategy comparison) by scanning for its edges."""
    gray = img.convert("L")
    w, h = gray.size
    col = w // 2
    rows = [gray.getpixel((col, y)) for y in range(h)]
    bg = rows[-1]
    end = max(y for y in range(h) if rows[y] > bg + 4)
    begin = end
    while begin > 0 and rows[begin - 1] > bg + 4:
        begin -= 1
    return img.crop((0, max(0, begin - 12), w, min(h, end + 12)))


def compare_frame(dash: Dashboard) -> Image.Image:
    # Re-render at 1.6x device scale so the comparison panel is large and crisp in the video.
    hi = BUILD / "frames" / "compare_hidpi.png"
    if hi.exists():
        tall = Image.open(hi).convert("RGB")
    else:
        url = page_url(dash.page, theme="dark", at=round(dash.t_end + 1, 2))
        tall = screenshot(url, hi, size=(1150, 1900), scale=1.6)
    crop = _last_panel(tall)
    img = canvas()
    title_font = font("bold", 46)
    d = ImageDraw.Draw(img)
    d.text(
        (W // 2, 150),
        "Same request, four execution strategies",
        font=title_font,
        fill=INK,
        anchor="mm",
    )
    d.text(
        (W // 2, 210),
        "measured on real Claude Haiku calls",
        font=font("sans", 30),
        fill=INK2,
        anchor="mm",
    )
    place(img, fit(crop, (1800, 600)), ((W - min(1800, crop.width)) // 2, 300))
    comp = {c["strategy"]: c["latency"] for c in dash.demo["compare"]}
    best_parallel = min(v for k, v in comp.items() if k != "sequential")
    worst_parallel = max(v for k, v in comp.items() if k != "sequential")
    lo, hi = comp["sequential"] / worst_parallel, comp["sequential"] / best_parallel
    return caption(img, f"Running independent tasks in parallel: {lo:.1f}–{hi:.1f}x faster than sequential",
                   "parallel strategies are close here; what matters is not waiting needlessly")  # fmt: skip


# ---------------------------------------------------------------------------- slides
CAPTION_TOP = H - 150  # content must stay above the caption band


def above_caption(img: Image.Image, top: int = 40) -> Image.Image:
    """Place ``img`` centred in the space above the caption band."""
    img = fit(img, (W - 200, CAPTION_TOP - top - 30))
    y = top + (CAPTION_TOP - top - img.height) // 2
    return place(canvas(), img, ((W - img.width) // 2, y))


def title_slide() -> Image.Image:
    img = canvas()
    d = ImageDraw.Draw(img)
    d.text(
        (W // 2, 380),
        "Adaptive Parallelism for LLM Agents",
        font=font("bold", 84),
        fill=INK,
        anchor="mm",
    )
    d.text((W // 2, 480), "An algorithm that decides which tasks run in parallel — and which should wait",
           font=font("sans", 40), fill=INK2, anchor="mm")  # fmt: skip
    tags = ["LangGraph", "Task DAGs", "Critical path", "Monte Carlo prediction", "Real Claude runs"]
    x = W // 2 - sum(d.textlength(t, font=font("sans", 30)) + 60 for t in tags) // 2
    for t in tags:
        w_ = d.textlength(t, font=font("sans", 30)) + 40
        d.rounded_rectangle([x, 600, x + w_, 656], 28, outline=ACCENT, width=2)
        d.text((x + w_ / 2, 628), t, font=font("sans", 30), fill=ACCENT, anchor="mm")
        x += w_ + 20
    return img


def lane(d: ImageDraw.ImageDraw, y: int, label: str, verdict: str, color: tuple[int, int, int],
         boxes: list[tuple[float, float, int, str, tuple[int, int, int]]], scale: float) -> None:  # fmt: skip
    d.text((120, y - 70), label, font=font("bold", 40), fill=INK)
    d.text((120, y - 22), verdict, font=font("sans", 28), fill=color)
    x0 = 120
    for start, dur, row, name, c in boxes:
        x1, x2 = x0 + start * scale, x0 + (start + dur) * scale - 6
        y1 = y + 30 + row * 40
        d.rounded_rectangle([x1, y1, x2, y1 + 33], 7, fill=c)
        d.text((x1 + 12, y1 + 16), name, font=font("sans", 22), fill=(255, 255, 255), anchor="lm")


def problem_slide() -> list[tuple[Image.Image, float]]:
    img = canvas()
    d = ImageDraw.Draw(img)
    d.text((W // 2, 70), "One request, five tasks: research A, B, C → compare → summarize",
           font=font("bold", 44), fill=INK, anchor="mm")  # fmt: skip
    s = 16.0
    seq = [(0, 20, 0, "research A", ORANGE), (20, 18, 0, "research B", ORANGE), (38, 19, 0, "research C", ORANGE),
           (57, 14, 0, "compare", ORANGE), (71, 9, 0, "summary", ORANGE)]  # fmt: skip
    lane(d, 215, "Everything in sequence", "correct, but slow: every task waits for the previous one",
         ORANGE, seq, s)  # fmt: skip
    blind = [(0, 20, 0, "research A", YELLOW), (0, 18, 1, "research B", YELLOW), (0, 19, 2, "research C", YELLOW),
             (0, 14, 3, "compare ×", FAIL), (0, 9, 4, "summary ×", FAIL)]  # fmt: skip
    lane(d, 400, "Everything in parallel", "fast, but compare and summary run before their inputs exist → wrong answers",
         FAIL, blind, s)  # fmt: skip
    adaptive = [(0, 20, 0, "research A", ACCENT), (0, 18, 1, "research B", ACCENT), (0, 19, 2, "research C", ACCENT),
                (20, 14, 0, "compare", OK), (34, 9, 0, "summary", OK)]  # fmt: skip
    lane(d, 725, "Adaptive scheduling", "parallel where the dependency graph allows, waiting only where it must",
         OK, adaptive, s)  # fmt: skip
    return [
        (
            caption(
                img,
                "Which tasks should run in parallel?",
                "too little is slow, too much breaks dependencies",
            ),
            7.0,
        )
    ]


def pipeline_slide() -> list[tuple[Image.Image, float]]:
    steps = [
        ("Plan", "LLM splits the request\ninto a task graph (DAG)"),
        ("Analyze", "critical path, total work,\navailable parallelism"),
        ("Predict", "simulate each configuration\n64x: latency, P95, cost"),
        ("Schedule", "priority queue: longest\nremaining path first"),
        ("Execute", "LangGraph workflow, async\nworkers, retries, streaming"),
    ]
    frames = []
    for k in range(1, len(steps) + 1):
        img = canvas()
        d = ImageDraw.Draw(img)
        d.text((W // 2, 150), "How it works", font=font("bold", 60), fill=INK, anchor="mm")
        bw, gap = 300, 50
        x = (W - (len(steps) * bw + (len(steps) - 1) * gap)) // 2
        for i, (name, desc) in enumerate(steps):
            on = i < k
            d.rounded_rectangle([x, 380, x + bw, 640], 18, fill=PANEL if on else BG,
                                outline=ACCENT if on else (60, 60, 58), width=3)  # fmt: skip
            d.text(
                (x + bw // 2, 430),
                f"{i + 1}",
                font=font("bold", 40),
                fill=ACCENT if on else INK3,
                anchor="mm",
            )
            d.text(
                (x + bw // 2, 490),
                name,
                font=font("bold", 40),
                fill=INK if on else INK3,
                anchor="mm",
            )
            for j, line in enumerate(desc.split("\n")):
                d.text(
                    (x + bw // 2, 555 + j * 34),
                    line,
                    font=font("sans", 25),
                    fill=INK2 if on else INK3,
                    anchor="mm",
                )
            if i < len(steps) - 1:
                d.text((x + bw + gap // 2, 510), "→", font=font("bold", 44), fill=INK3, anchor="mm")
            x += bw + gap
        frames.append((caption(img, "Plan → analyze → predict → schedule → execute",
                               "every run learns: observed latency and tokens recalibrate the model"), 1.4))  # fmt: skip
    frames[-1] = (frames[-1][0], 3.0)
    return frames


def code_slides() -> list[tuple[Image.Image, float]]:
    src = ROOT / "src" / "adaptive_scheduler"
    items = [
        (src / "agent" / "workflow.py", "builder = StateGraph(AgentState)", "return builder.compile", 14,
         "agent/workflow.py", "The LangGraph workflow", "plan, repair invalid plans, schedule, execute, synthesize"),
        (src / "graph" / "analysis.py", "def upward_rank(", "@dataclass", 16,
         "graph/analysis.py", "Critical-path analysis", "upward rank = longest remaining path from each task"),
        (src / "scheduling" / "ready_queue.py", "    def pop(self) -> str:", "    def fail(", 22,
         "scheduling/ready_queue.py", "Dependency-aware priority queue",
         "the same queue drives the simulator and the live executor"),
        (src / "simulation" / "selector.py", "    best_success = max(", "    return Selection(", 18,
         "simulation/selector.py", "Choosing the configuration",
         "success constraint, then lowest score, then the simplest near-tie"),
    ]  # fmt: skip
    frames = []
    for path, start, end, n, title, cap, sub in items:
        code = code_render(path, start, end, max_lines=n, size=32)
        img = above_caption(window(code, title))
        frames.append((caption(img, cap, sub), 3.4))
    return frames


def results_slides() -> list[tuple[Image.Image, float]]:
    res = ROOT / "docs" / "results"
    frames = []
    for name, cap, sub in (
        ("speedup_by_family.png", "136,000 simulated runs across 7 workload families",
         "within 5% of the best fixed strategy in every condition tested"),
        ("p95_by_condition.png", "Under a tight provider rate limit: 14% lower P95 than eager fan-out",
         "it bounds concurrency instead of flooding the API"),
    ):  # fmt: skip
        fig = Image.open(res / name).convert("RGB")
        img = above_caption(fig)
        frames.append((caption(img, cap, sub), 4.2))
    return frames


def numbers_slide() -> list[tuple[Image.Image, float]]:
    img = canvas()
    d = ImageDraw.Draw(img)
    d.text(
        (W // 2, 140),
        "Validated on real Claude Haiku 5.5 calls",
        font=font("bold", 56),
        fill=INK,
        anchor="mm",
    )
    stats = [("1.66x", "faster than sequential\n(95% CI 1.53–1.80)"), ("100%", "task success\n160 executions"),
             ("−2.4%", "latency forecast bias\non fresh data"), ("$0.70", "total cost of the\nlive study")]  # fmt: skip
    cw = W // len(stats)
    for i, (big, small) in enumerate(stats):
        cx = cw * i + cw // 2
        d.text((cx, 470), big, font=font("bold", 120), fill=ACCENT, anchor="mm")
        for j, line in enumerate(small.split("\n")):
            d.text((cx, 590 + j * 44), line, font=font("sans", 34), fill=INK2, anchor="mm")
    return [
        (
            caption(
                img,
                "Measured, not assumed",
                "the live run also exposed a calibration flaw, which was fixed and re-validated",
            ),
            5.0,
        )
    ]


def production_slide(swagger: Image.Image, metrics_txt: str) -> list[tuple[Image.Image, float]]:
    img = canvas()
    left = fit(window(swagger.crop((0, 0, 1400, 1000)), "localhost:8080/docs"), (980, 820))
    right = fit(window(terminal_render(metrics_txt, size=20), "Prometheus metrics"), (860, 820))
    place(img, left, (40, 60))
    place(img, right, (W - right.width - 40, 60))
    return [
        (
            caption(
                img,
                "Production-ready service",
                "FastAPI + live streaming, metrics, tracing, Docker and Kubernetes",
            ),
            4.5,
        )
    ]


def start_slide(start_txt: str) -> list[tuple[Image.Image, float]]:
    img = canvas()
    d = ImageDraw.Draw(img)
    d.text((W // 2, 120), "Try it", font=font("bold", 64), fill=INK, anchor="mm")
    term = fit(window(terminal_render(start_txt, size=30), "terminal"), (1500, 640))
    place(img, term, ((W - term.width) // 2, 200))
    d.text((W // 2, 900), REPO_URL, font=font("bold", 36), fill=ACCENT, anchor="mm")
    d.text((W // 2, 960), "works offline with a built-in mock LLM, or with an Anthropic API key",
           font=font("sans", 30), fill=INK2, anchor="mm")  # fmt: skip
    return [(img, 5.5)]


# ---------------------------------------------------------------------------- outputs
def write_screenshots(dash: Dashboard, texts: dict[str, str], swagger: Image.Image) -> None:
    SHOTS.mkdir(parents=True, exist_ok=True)
    running = dash.at(dash.event_time("task_finished") - 4)
    running.save(SHOTS / "dashboard_running.png", optimize=True)
    done = dash.at(dash.t_end + 1, size=(W, 1500))
    done.save(SHOTS / "dashboard_complete.png", optimize=True)
    done.crop((1100, 120, 1920, 550)).save(SHOTS / "dashboard_decision.png", optimize=True)
    done.crop((0, 555, 1920, 790)).save(SHOTS / "dashboard_timeline.png", optimize=True)
    done.crop((0, 1100, 1920, 1370)).save(SHOTS / "dashboard_compare.png", optimize=True)
    window(terminal_render(texts["run"], size=20), "apsched run").save(
        SHOTS / "cli_run.png", optimize=True
    )
    window(terminal_render(texts["info"], size=22), "apsched info").save(
        SHOTS / "cli_info.png", optimize=True
    )
    window(terminal_render(texts["metrics"], size=20), "metrics").save(
        SHOTS / "metrics.png", optimize=True
    )
    swagger.crop((0, 0, 1600, 1000)).save(SHOTS / "api_docs.png", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--service-url", help="capture /docs and /metrics from this running service"
    )
    parser.add_argument("--skip-videos", action="store_true")
    args = parser.parse_args()

    demo = json.loads((MEDIA / "demo_run.json").read_text(encoding="utf-8"))
    dash = Dashboard(demo)
    with service(args.service_url) as base:
        texts = terminal_texts(base)
        swagger = screenshot(
            f"{base}/docs", BUILD / "swagger.png", size=(1600, 1000), budget_ms=4000
        )
    write_screenshots(dash, texts, swagger)
    print("screenshots written")
    if args.skip_videos:
        return

    # README clip: the real run, then the strategy comparison.
    short = MEDIA / "demo.mp4"
    v = Video(short)
    v.scene(run_frames(dash, duration=13.0), fade=0)
    v.scene([(compare_frame(dash), 4.0)])
    v.close()
    gif(short, MEDIA / "demo.gif", width=960, fps=8)
    print(f"{short.name}: {v.frames / v.fps:.1f}s")

    # Full captioned walkthrough.
    full = MEDIA / "overview.mp4"
    v = Video(full)
    v.scene([(title_slide(), 3.6)], fade=0)
    v.scene(problem_slide())
    v.scene(pipeline_slide())
    v.scene(run_frames(dash, duration=15.0))
    v.scene([(compare_frame(dash), 4.5)])
    v.scene(code_slides())
    v.scene(results_slides())
    v.scene(numbers_slide())
    v.scene(production_slide(swagger, texts["metrics"]))
    v.scene(start_slide(texts["start"]))
    v.close()
    print(f"{full.name}: {v.frames / v.fps:.1f}s")


if __name__ == "__main__":
    main()
