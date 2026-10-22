"""HTTP service tests (FastAPI TestClient, mock LLM, no network)."""

import json
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from adaptive_scheduler.agent import AdaptiveAgent, AgentOptions, HeuristicPlanner
from adaptive_scheduler.api import create_app
from adaptive_scheduler.config import Settings
from adaptive_scheduler.llm import MockChatModel
from adaptive_scheduler.simulation import RetryPolicy

REQUEST = "Research the EV markets of India, China and Germany, then compare them"


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("ANTHROPIC_API_KEY", "APSCHED_API_KEY", "APSCHED_LLM_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setitem(Settings.model_config, "env_file", None)


def make_agent() -> AdaptiveAgent:
    llm = MockChatModel(time_scale=0.001)
    opts = AgentOptions(
        time_scale=0.001, selector_samples=4, retry=RetryPolicy(backoff_base_s=0.001)
    )
    return AdaptiveAgent(llm, HeuristicPlanner(llm, 32), options=opts)


def client(agent: Any = None, **settings: Any) -> TestClient:
    app = create_app(Settings(log_format="text", **settings), agent=agent or make_agent())
    return TestClient(app)


@pytest.fixture
def api() -> Iterator[TestClient]:
    with client() as c:
        yield c


# ---------------------------------------------------------------------------- ops endpoints
def test_health_ready_info(api: TestClient) -> None:
    assert api.get("/healthz").json() == {"status": "ok"}
    assert api.get("/readyz").json() == {"ready": True}
    info = api.get("/v1/info").json()
    assert info["llm_provider"] == "mock"
    assert "adaptive" in info["strategies"]


def test_run_returns_full_result(api: TestClient) -> None:
    r = api.post("/v1/runs", json={"request": REQUEST}, headers={"x-request-id": "req-1"})
    assert r.status_code == 200
    assert r.headers["x-request-id"] == "req-1"
    body = r.json()
    assert body["answer"]
    assert body["metrics"]["success"]
    assert body["decision"]["strategy"] == "adaptive"
    assert len(body["report"]["records"]) == body["metrics"]["tasks_executed"]
    assert body["run_id"]


def test_baseline_strategy_selectable(api: TestClient) -> None:
    r = api.post("/v1/runs", json={"request": REQUEST, "strategy": "sequential"})
    assert r.json()["metrics"]["peak_concurrency"] == 1


@pytest.mark.parametrize(
    ("payload", "status"),
    [
        ({"request": ""}, 422),
        ({"request": "   "}, 422),
        ({"request": "x", "strategy": "nope"}, 422),
        ({}, 422),
    ],
)
def test_input_validation(api: TestClient, payload: dict[str, Any], status: int) -> None:
    assert api.post("/v1/runs", json=payload).status_code == status


def test_request_size_limit() -> None:
    with client(max_request_chars=10) as c:
        r = c.post("/v1/runs", json={"request": "x" * 11})
        assert r.status_code == 413


def test_api_key_auth() -> None:
    with client(api_key="s3cret") as c:
        assert c.post("/v1/runs", json={"request": REQUEST}).status_code == 401
        bad = c.post("/v1/runs", json={"request": REQUEST}, headers={"x-api-key": "wrong"})
        assert bad.status_code == 401
        ok = c.post("/v1/runs", json={"request": REQUEST}, headers={"x-api-key": "s3cret"})
        assert ok.status_code == 200
        assert c.get("/healthz").status_code == 200  # ops endpoints stay open for probes


def test_admission_control_rejects_when_full() -> None:
    with client(max_concurrent_runs=1) as c:
        service = c.app.state.service  # type: ignore[attr-defined]
        service.slots._value = 0  # simulate one run in flight
        r = c.post("/v1/runs", json={"request": REQUEST})
        assert r.status_code == 503
        assert r.headers["retry-after"] == "5"
        service.slots._value = 1


def test_run_timeout() -> None:
    with client(run_timeout_s=1e-6) as c:
        assert c.post("/v1/runs", json={"request": REQUEST}).status_code == 504


def test_internal_errors_do_not_leak() -> None:
    class Broken:
        async def arun(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("database password is hunter2")

    with client(agent=Broken()) as c:
        r = c.post("/v1/runs", json={"request": REQUEST})
        assert r.status_code == 500
        assert "hunter2" not in r.text
        assert r.json()["request_id"] == r.headers["x-request-id"]


# ---------------------------------------------------------------------------- streaming
def _sse(text: str) -> list[tuple[str, dict[str, Any]]]:
    events = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "event" in fields:
            events.append((fields["event"], json.loads(fields["data"])))
    return events


def test_stream_emits_lifecycle_then_result(api: TestClient) -> None:
    with api.stream("POST", "/v1/runs/stream", json={"request": REQUEST}) as r:
        assert r.status_code == 200
        events = _sse(r.read().decode())
    names = [e for e, _ in events]
    assert names[0] == "run_started"
    assert names[1:3] == ["plan_ready", "decision"]
    assert names[-2:] == ["run_finished", "result"]
    started = [d["data"]["task_id"] for e, d in events if e == "task_started"]
    finished = [d["data"]["task_id"] for e, d in events if e == "task_finished"]
    assert sorted(started) == sorted(finished)
    result = events[-1][1]
    assert result["answer"]
    assert len({d["run_id"] for e, d in events if e != "result"}) == 1


def test_stream_reports_errors_as_events() -> None:
    class Broken:
        async def arun(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("secret internals")

    with (
        client(agent=Broken()) as c,
        c.stream("POST", "/v1/runs/stream", json={"request": "x"}) as r,
    ):
        events = _sse(r.read().decode())
    assert events == [("error", {"error": "run failed", "request_id": None})]
    assert c.app.state.service.slots._value == 8  # type: ignore[attr-defined]  # slot released


# ---------------------------------------------------------------------------- metrics
def test_metrics_reflect_runs(api: TestClient) -> None:
    api.post("/v1/runs", json={"request": REQUEST})
    text = api.get("/metrics").text
    assert 'apsched_runs_total{outcome="success",strategy="adaptive"} 1.0' in text
    assert "apsched_tasks_total" in text
    assert "apsched_run_duration_seconds_bucket" in text
    assert "apsched_runs_in_flight 0.0" in text
    assert 'apsched_tokens_total{direction="input"}' in text
