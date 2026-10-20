"""FastAPI middleware for request correlation, latency instrumentation, and IP rate limiting."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from autovoice.config import get_settings
from autovoice.logging_conf import get_logger

log = get_logger(__name__)

# In-memory sliding window rate limiter: IP -> list of timestamps
_RATE_LIMIT_BUCKET: dict[str, list[float]] = {}


class RequestIDAndTimingMiddleware(BaseHTTPMiddleware):
    """Assigns or propagates X-Request-ID and measures endpoint latency."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process incoming request, inject request_id, and measure latency."""
        req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = req_id

        start_time = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            log.error(
                "http.request.failed",
                request_id=req_id,
                path=request.url.path,
                method=request.method,
                duration_ms=round(duration_ms, 2),
                error=str(exc),
            )
            raise exc

        duration_ms = (time.perf_counter() - start_time) * 1000
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"

        # Structured request logging
        log.info(
            "http.request.completed",
            request_id=req_id,
            path=request.url.path,
            method=request.method,
            status=response.status_code,
            duration_ms=round(duration_ms, 2),
        )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Lightweight in-memory sliding window rate limiter per client IP."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Enforce request quota per IP per rolling 60-second window."""
        settings = get_settings()

        # Skip rate limiting for health/ready probes or tests
        if request.url.path in ("/health", "/ready", "/docs", "/openapi.json"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"

        # Allow localhost / test client to bypass high limits
        if client_ip in ("127.0.0.1", "testclient") and settings.environment in ("test", "dev"):
            return await call_next(request)

        now = time.time()
        window_start = now - 60.0

        # Retrieve and prune timestamps
        timestamps = _RATE_LIMIT_BUCKET.setdefault(client_ip, [])
        timestamps = [ts for ts in timestamps if ts > window_start]
        _RATE_LIMIT_BUCKET[client_ip] = timestamps

        if len(timestamps) >= settings.rate_limit_per_minute:
            log.warning("rate_limit.exceeded", client_ip=client_ip, count=len(timestamps))
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded. Max requests per minute reached.",
                    "limit_per_minute": settings.rate_limit_per_minute,
                },
                headers={"Retry-After": "60"},
            )

        timestamps.append(now)
        return await call_next(request)
