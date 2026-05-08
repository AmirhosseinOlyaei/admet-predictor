"""API middleware: rate limiting and request timing."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from threading import Lock

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# Rate limiting configuration
_MAX_REQUESTS_PER_MINUTE = 100
_WINDOW_SECONDS = 60

# Per-IP request tracking: {ip: [timestamp, ...]}
_request_counts: dict[str, list[float]] = defaultdict(list)
_lock = Lock()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting: max 100 requests per minute per IP."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        client_ip = self._get_client_ip(request)
        now = time.time()

        with _lock:
            timestamps = _request_counts[client_ip]
            # Remove timestamps older than the window
            cutoff = now - _WINDOW_SECONDS
            _request_counts[client_ip] = [t for t in timestamps if t > cutoff]
            count = len(_request_counts[client_ip])

            if count >= _MAX_REQUESTS_PER_MINUTE:
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": (
                            f"Rate limit exceeded: max {_MAX_REQUESTS_PER_MINUTE} "
                            f"requests per minute."
                        )
                    },
                )

            _request_counts[client_ip].append(now)

        return await call_next(request)

    @staticmethod
    def _get_client_ip(request: Request) -> str:
        """Extract client IP, respecting X-Forwarded-For header."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        if request.client:
            return request.client.host
        return "unknown"


class TimingMiddleware(BaseHTTPMiddleware):
    """Log request timing and add X-Process-Time header."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - t0) * 1000.0
        response.headers["X-Process-Time-Ms"] = f"{elapsed:.2f}"
        logger.info(
            "%s %s → %d (%.1f ms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )
        return response
