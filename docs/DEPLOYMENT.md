# Deployment

The service is a stateless HTTP API (FastAPI + Uvicorn) packaged as a non-root container. It runs
on the deterministic mock LLM unless `ANTHROPIC_API_KEY` is provided.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/healthz` | Liveness: the process is up |
| GET | `/readyz` | Readiness: agent constructed, provider resolved |
| GET | `/v1/info` | Version, LLM backend, limits |
| POST | `/v1/runs` | Run a request; returns plan, decision (with every candidate's prediction), per-task timeline, metrics and answer |
| POST | `/v1/runs/stream` | Same, as Server-Sent Events: `run_started`, `plan_ready`, `decision`, `task_started`, `task_finished`, `run_finished`, `result` (or `error`) |
| GET | `/metrics` | Prometheus exposition |
| GET | `/docs` | Interactive OpenAPI documentation |

```bash
curl -X POST localhost:8080/v1/runs -H 'content-type: application/json' \
  -d '{"request": "Compare PostgreSQL, MySQL and SQLite, then recommend one", "strategy": "adaptive"}'

curl -N -X POST localhost:8080/v1/runs/stream -H 'content-type: application/json' \
  -d '{"request": "Research A, B and C, then compare them"}'
```

## Run it

```bash
# Local process
pip install -e ".[api,observability,anthropic]"
apsched serve --host 0.0.0.0 --port 8080

# Container
docker build -t adaptive-scheduler .
docker run -p 8080:8080 --env-file .env adaptive-scheduler

# API + Prometheus
docker compose up --build        # API :8080, Prometheus :9090
```

### Google Cloud Run

The image listens on `$PORT` and is stateless, so it deploys as-is:

```bash
gcloud run deploy adaptive-scheduler --source . --region us-central1 \
  --set-secrets ANTHROPIC_API_KEY=anthropic-key:latest,APSCHED_API_KEY=service-key:latest \
  --concurrency 8 --timeout 300
```

Keep Cloud Run's `--concurrency` equal to `APSCHED_MAX_CONCURRENT_RUNS` so the platform's load
balancing and the service's admission control agree.

### Kubernetes

`deploy/k8s/deployment.yaml` defines a Deployment and Service with readiness/liveness probes,
resource requests/limits, a non-root read-only root filesystem, dropped capabilities, Prometheus
scrape annotations and secrets injected from a `Secret`.

```bash
kubectl create secret generic adaptive-scheduler --from-literal=ANTHROPIC_API_KEY=...
kubectl apply -f deploy/k8s/deployment.yaml
```

## Configuration

All settings are environment variables (or a `.env` file).

| Variable | Default | Meaning |
|---|---|---|
| `APSCHED_LLM_PROVIDER` | `auto` | `auto` (Anthropic if a key is set, else mock), `mock`, `anthropic` |
| `ANTHROPIC_API_KEY` | - | Enables real Claude calls |
| `APSCHED_ANTHROPIC_MODEL` | `claude-haiku-5-5` | Model id |
| `APSCHED_API_KEY` | - | If set, `/v1/*` requires the `X-API-Key` header |
| `APSCHED_MAX_CONCURRENT_RUNS` | `8` | Admission limit; excess requests get `503` + `Retry-After` |
| `APSCHED_RUN_TIMEOUT_S` | `300` | Per-run deadline (`504` when exceeded) |
| `APSCHED_MAX_REQUEST_CHARS` | `8000` | Request size limit (`413` when exceeded) |
| `APSCHED_MAX_CONCURRENCY` | `8` | Provider concurrency limit the scheduler plans for |
| `APSCHED_SELECTOR_SAMPLES` | `64` | Monte Carlo samples per candidate configuration |
| `APSCHED_MAX_PLAN_TASKS` | `32` | Plan size guardrail |
| `APSCHED_MAX_RETRIES` | `2` | Retries per task |
| `APSCHED_TASK_TIMEOUT_S` | `120` | Per-attempt timeout |
| `APSCHED_MOCK_TIME_SCALE` | `0.1` | Mock LLM wall-clock seconds per simulated second |
| `APSCHED_LOG_LEVEL` / `APSCHED_LOG_FORMAT` | `INFO` / `json` | Logging |
| `APSCHED_OTEL_CONSOLE` | `false` | Print spans to stdout |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | - | Export spans via OTLP (requires `opentelemetry-exporter-otlp`) |

## Observability

* **Metrics** (`/metrics`): `apsched_runs_total{strategy,outcome}`, `apsched_run_duration_seconds`,
  `apsched_runs_in_flight`, `apsched_tasks_total{task_type,status}`,
  `apsched_task_duration_seconds`, `apsched_model_calls_total`, `apsched_tokens_total{direction}`,
  `apsched_cost_usd_total`, `apsched_decisions_total{config}`, `apsched_plan_parallelism`.
  Labels have bounded cardinality (no run or task ids).
* **Traces**: one `agent.run` span per run with an `agent.task` child per executed task, so a trace
  viewer renders the real schedule as a Gantt chart; plan and decision details are attributes.
* **Logs**: one JSON object per line with `request_id`; the same id is returned in the
  `x-request-id` response header (and honoured if the client sends one).

Metrics, traces and the SSE stream are all derived from one typed event stream, so they cannot
disagree.

## Operational notes

* **Statelessness and learning.** The cost estimator learns from observed calls in memory, per
  replica. Replicas converge independently; a restart returns to the priors. Persisting estimator
  state is a natural extension.
* **Checkpointing.** `agent.checkpoint.memory_checkpointer()` is in-process. For resumable runs
  across replicas use a shared LangGraph saver (e.g. Postgres) built with
  `serde=agent.checkpoint.serializer()`, which restricts deserialization to this agent's types.
* **Cancellation.** If a streaming client disconnects, its run is cancelled so no further tokens
  are spent.
* **Graceful shutdown.** Uvicorn is PID 1 (`exec`) and drains in-flight requests for up to 30 s on
  SIGTERM; the Kubernetes manifest allows 60 s.
