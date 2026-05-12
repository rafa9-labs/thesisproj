"""Security middleware for FastAPI — rate limiting, headers, input validation."""

from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Optional

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

IS_DESKTOP = os.environ.get("FX_APP_MODE", "") == "desktop"

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Cache-Control": "no-store" if not IS_DESKTOP else "private",
}


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        for header, value in SECURITY_HEADERS.items():
            response.headers[header] = value
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app: FastAPI,
        general_limit: int = 600,
        general_window: int = 60,
        activation_limit: int = 5,
        activation_window: int = 300,
    ):
        super().__init__(app)
        self.general_limit = general_limit
        self.general_window = general_window
        self.activation_limit = activation_limit
        self.activation_window = activation_window
        self._general_requests: dict[str, list[float]] = defaultdict(list)
        self._activation_requests: dict[str, list[float]] = defaultdict(list)

    def _cleanup(self, bucket: dict[str, list[float]], window: int) -> None:
        cutoff = time.time() - window
        for ip in list(bucket.keys()):
            bucket[ip] = [t for t in bucket[ip] if t > cutoff]
            if not bucket[ip]:
                del bucket[ip]

    async def dispatch(self, request: Request, call_next):
        if IS_DESKTOP:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        is_activation = "/license/activate" in (request.url.path or "")
        if is_activation:
            self._cleanup(self._activation_requests, self.activation_window)
            bucket = self._activation_requests[client_ip]
            if len(bucket) >= self.activation_limit:
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Too many activation attempts. Try again later."},
                )
            bucket.append(now)
        else:
            self._cleanup(self._general_requests, self.general_window)
            bucket = self._general_requests[client_ip]
            if len(bucket) >= self.general_limit:
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded. Try again later."},
                )
            bucket.append(now)

        return await call_next(request)


def install_security_middleware(app: FastAPI) -> None:
    app.add_middleware(SecurityHeadersMiddleware)
    if not IS_DESKTOP:
        app.add_middleware(RateLimitMiddleware)