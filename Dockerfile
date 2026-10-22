# syntax=docker/dockerfile:1

# ---- build: install the package into a self-contained virtualenv -------------------------------
FROM python:3.12-slim AS build
# Generous network settings: package downloads are the flakiest part of any image build.
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
# The BuildKit cache mount keeps downloaded wheels between builds without baking them into a layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv /opt/venv \
 && /opt/venv/bin/pip install ".[api,observability,anthropic]"

# ---- runtime: slim image, non-root, no build tooling ---------------------------------------------
FROM python:3.12-slim AS runtime
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    APSCHED_LOG_FORMAT=json
RUN useradd --create-home --uid 10001 app
COPY --from=build /opt/venv /opt/venv
USER 10001
WORKDIR /home/app
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/healthz' % os.environ.get('PORT', '8080'), timeout=2)"

# exec so uvicorn is PID 1 and receives SIGTERM for graceful shutdown.
CMD ["sh", "-c", "exec uvicorn adaptive_scheduler.api.app:create_app --factory --host 0.0.0.0 --port ${PORT} --no-access-log --timeout-graceful-shutdown 30"]
