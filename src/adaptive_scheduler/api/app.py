"""HTTP service.

Endpoints
---------
``GET  /healthz``         liveness (process is up)
``GET  /readyz``          readiness (agent constructed, provider resolved)
``GET  /v1/info``         version, LLM backend, limits
``POST /v1/runs``         run a request; return plan, decision, timeline and metrics
``POST /v1/runs/stream``  same, streamed as Server-Sent Events (one event per lifecycle step)
``GET  /metrics``         Prometheus exposition

Production concerns handled here: optional API-key auth (constant-time comparison), admission
control (at most ``max_concurrent_runs`` runs; excess requests get ``503`` + ``Retry-After``
instead of queueing unboundedly), a per-run timeout (``504``), request-size limits (``413``), a
request id on every response and log line, cancellation of abandoned streams, and error
responses that never leak internals.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from importlib import resources
from typing import Annotated, Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from adaptive_scheduler import __version__
from adaptive_scheduler.agent.workflow import AdaptiveAgent, AgentResult
from adaptive_scheduler.config import LLMProvider, Settings
from adaptive_scheduler.events import AgentEvent, EventSink, fan_out
from adaptive_scheduler.observability import (
    MetricsCollector,
    SpanRecorder,
    configure_logging,
    request_id_var,
    setup_tracing,
)
from adaptive_scheduler.simulation.strategies import Strategy

logger = logging.getLogger("adaptive_scheduler.api")


class RunRequest(BaseModel):
    request: str = Field(min_length=1, description="The task to solve.")
    strategy: Strategy = Strategy.ADAPTIVE


class ErrorBody(BaseModel):
    error: str
    request_id: str | None = None


class Service:
    """Per-application state, stored on ``app.state`` (no module globals, so apps are isolated)."""

    def __init__(self, settings: Settings, agent: AdaptiveAgent | None) -> None:
        self.settings = settings
        self.agent = agent
        self.metrics = MetricsCollector()
        self.spans = SpanRecorder()
        self.slots = asyncio.Semaphore(settings.max_concurrent_runs)
        self.ready = False

    def sink(
        self, extra: Callable[[AgentEvent], Awaitable[None] | None] | None = None
    ) -> EventSink:
        return fan_out([self.metrics, self.spans, extra])


# ---------------------------------------------------------------------------- dependencies
def get_service(request: Request) -> Service:
    service: Service = request.app.state.service
    return service


ServiceDep = Annotated[Service, Depends(get_service)]


def require_api_key(request: Request, service: ServiceDep) -> None:
    expected = service.settings.api_key
    if expected is None:
        return
    given = request.headers.get("x-api-key", "")
    if not hmac.compare_digest(given.encode(), expected.get_secret_value().encode()):
        raise HTTPException(status_code=401, detail="invalid or missing API key")


def validated_body(body: RunRequest, service: ServiceDep) -> RunRequest:
    limit = service.settings.max_request_chars
    if len(body.request) > limit:
        raise HTTPException(status_code=413, detail=f"request exceeds {limit} characters")
    if not body.request.strip():
        raise HTTPException(status_code=422, detail="request must not be blank")
    return body


def ready_agent(service: ServiceDep) -> AdaptiveAgent:
    if service.agent is None or not service.ready:
        raise HTTPException(status_code=503, detail="service not ready")
    if service.slots.locked():
        raise HTTPException(
            status_code=503, detail="server at capacity; retry later", headers={"Retry-After": "5"}
        )
    return service.agent


BodyDep = Annotated[RunRequest, Depends(validated_body)]
AgentDep = Annotated[AdaptiveAgent, Depends(ready_agent)]
ERRORS: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorBody} for code in (401, 413, 503, 504)
}

# ---------------------------------------------------------------------------- operational routes
ops = APIRouter(tags=["ops"])


@ops.get("/", response_class=HTMLResponse, include_in_schema=False)
@ops.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard() -> HTMLResponse:
    """Interactive dashboard: run a request and watch planning, scheduling and execution live."""
    page = resources.files("adaptive_scheduler.api").joinpath("static/dashboard.html")
    return HTMLResponse(page.read_text(encoding="utf-8"))


@ops.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@ops.get("/readyz")
async def readyz(service: ServiceDep) -> JSONResponse:
    ok = service.ready and service.agent is not None
    return JSONResponse({"ready": ok}, status_code=200 if ok else 503)


@ops.get("/v1/info")
async def info(service: ServiceDep) -> dict[str, Any]:
    s = service.settings
    provider = s.resolve_provider()
    return {
        "version": __version__,
        "llm_provider": provider.value,
        "model": s.anthropic_model if provider is LLMProvider.ANTHROPIC else "mock",
        "mock_time_scale": s.mock_time_scale if provider is LLMProvider.MOCK else 1.0,
        "max_concurrent_runs": s.max_concurrent_runs,
        "max_request_chars": s.max_request_chars,
        "run_timeout_s": s.run_timeout_s,
        "strategies": [st.value for st in Strategy],
    }


@ops.get("/metrics")
async def metrics(service: ServiceDep) -> Response:
    return Response(generate_latest(service.metrics.registry), media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------------------- run routes
runs = APIRouter(prefix="/v1", tags=["runs"], dependencies=[Depends(require_api_key)])


@runs.post("/runs", responses=ERRORS)
async def create_run(body: BodyDep, agent: AgentDep, service: ServiceDep) -> dict[str, Any]:
    async with service.slots:
        try:
            result = await asyncio.wait_for(
                agent.arun(body.request, strategy=body.strategy, on_event=service.sink()),
                timeout=service.settings.run_timeout_s,
            )
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail="run timed out") from exc
    return result.model_dump(mode="json")


@runs.post("/runs/stream", responses=ERRORS)
async def stream_run(body: BodyDep, agent: AgentDep, service: ServiceDep) -> EventSourceResponse:
    queue: asyncio.Queue[AgentEvent | AgentResult | BaseException] = asyncio.Queue()
    # Take the slot now (ready_agent just saw it free and nothing awaited since), so two streams
    # cannot both pass admission and then wait for the same slot.
    await service.slots.acquire()

    async def run() -> None:
        try:
            result = await asyncio.wait_for(
                agent.arun(body.request, strategy=body.strategy, on_event=service.sink(queue.put)),
                timeout=service.settings.run_timeout_s,
            )
            await queue.put(result)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # forwarded to the client as an error event
            logger.exception("streamed run failed")
            await queue.put(exc)
        finally:
            service.slots.release()

    task = asyncio.create_task(run())

    async def events() -> AsyncIterator[dict[str, str]]:
        try:
            while True:
                item = await queue.get()
                if isinstance(item, AgentEvent):
                    yield {"event": item.type.value, "data": item.model_dump_json()}
                elif isinstance(item, AgentResult):
                    yield {"event": "result", "data": item.model_dump_json()}
                    return
                else:
                    message = "run timed out" if isinstance(item, TimeoutError) else "run failed"
                    yield {"event": "error", "data": ErrorBody(error=message).model_dump_json()}
                    return
        finally:
            if not task.done():  # client went away: stop spending tokens
                task.cancel()

    return EventSourceResponse(events())


# ---------------------------------------------------------------------------- application
async def _request_context(request: Request, call_next: Any) -> Response:
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex
    token = request_id_var.set(rid)
    started = time.perf_counter()
    try:
        response: Response = await call_next(request)
    except Exception:
        logger.exception("unhandled error")
        body = ErrorBody(error="internal error", request_id=rid).model_dump()
        response = JSONResponse(body, status_code=500)
    finally:
        request_id_var.reset(token)
    response.headers["x-request-id"] = rid
    logger.info(
        "request",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": round(1000 * (time.perf_counter() - started), 1),
            "request_id": rid,
        },
    )
    return response


def create_app(settings: Settings | None = None, agent: AdaptiveAgent | None = None) -> FastAPI:
    """Build the application. ``agent`` can be injected (tests); otherwise it is created from
    ``settings`` at startup, which fails fast on invalid configuration."""
    settings = settings or Settings()
    service = Service(settings, agent)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings.log_level, settings.log_format)
        setup_tracing(settings.service_name, console=settings.otel_console)
        if service.agent is None:
            service.agent = AdaptiveAgent.from_settings(settings)
        service.ready = True
        logger.info("service ready", extra={"provider": settings.resolve_provider().value})
        yield
        service.ready = False

    app = FastAPI(
        title="Adaptive Scheduler",
        version=__version__,
        description="Cost-aware adaptive parallelism for LangGraph agents.",
        lifespan=lifespan,
    )
    app.state.service = service
    app.middleware("http")(_request_context)
    app.include_router(ops)
    app.include_router(runs)
    return app
