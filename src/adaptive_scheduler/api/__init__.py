"""FastAPI service exposing the adaptive agent over HTTP (JSON and Server-Sent Events)."""

from adaptive_scheduler.api.app import RunRequest, create_app

__all__ = ["RunRequest", "create_app"]
