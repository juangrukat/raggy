"""
HTTP transport security middleware for the MCP server.

Two concerns are addressed here:

1. **Origin validation** — DNS rebinding attacks can let a malicious page
   reach a localhost server from a victim's browser. Reject requests whose
   `Origin` is not in the allowlist when the server is bound to a loopback
   address.

2. **Optional Bearer token auth** — when ``MCP_HTTP_AUTH_TOKEN`` is set,
   require ``Authorization: Bearer <token>`` on every MCP request. This is
   not a substitute for proper auth, but it prevents any local process from
   trivially impersonating a connected agent.

Both checks are no-ops on requests that don't apply (no Origin header,
no token configured) so behavior under stdio is unchanged.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)


def _allowed_origins() -> set[str]:
    raw = os.getenv("MCP_HTTP_ALLOWED_ORIGINS", "")
    base = {
        "http://localhost",
        "http://127.0.0.1",
        "https://localhost",
        "https://127.0.0.1",
    }
    extra = {o.strip() for o in raw.split(",") if o.strip()}
    return base | extra


class OriginValidationMiddleware(BaseHTTPMiddleware):
    """Reject cross-origin requests on a loopback HTTP MCP server."""

    def __init__(self, app, allowed_origins: Iterable[str] | None = None):
        super().__init__(app)
        self._allowed = set(allowed_origins) if allowed_origins else _allowed_origins()

    async def dispatch(self, request: Request, call_next) -> Response:
        origin = request.headers.get("origin")
        if origin:
            # An origin like http://example.com:1234 → check the scheme://host portion
            # and accept if any allowed origin is a prefix.
            if not any(origin == o or origin.startswith(o + ":") for o in self._allowed):
                logger.warning("Rejected request with disallowed Origin: %s", origin)
                return JSONResponse(
                    status_code=403,
                    content={"error": "origin_not_allowed", "origin": origin},
                )
        return await call_next(request)


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Require ``Authorization: Bearer <token>`` when an auth token is configured."""

    def __init__(self, app, expected_token: str | None = None):
        super().__init__(app)
        self._token = expected_token or os.getenv("MCP_HTTP_AUTH_TOKEN") or None

    async def dispatch(self, request: Request, call_next) -> Response:
        if not self._token:
            return await call_next(request)
        # Allow OpenAPI/health probes without auth if you ever add them; for now require all.
        header = request.headers.get("authorization", "")
        if not header.lower().startswith("bearer "):
            return JSONResponse(status_code=401, content={"error": "missing_bearer_token"})
        provided = header.split(" ", 1)[1].strip()
        if provided != self._token:
            return JSONResponse(status_code=401, content={"error": "invalid_token"})
        return await call_next(request)


def build_middleware_stack(auth_token: str | None = None):
    """Return the ordered middleware list for fastmcp's `http_app(middleware=...)`."""
    from starlette.middleware import Middleware

    stack = [Middleware(OriginValidationMiddleware)]
    if auth_token or os.getenv("MCP_HTTP_AUTH_TOKEN"):
        stack.append(Middleware(BearerAuthMiddleware, expected_token=auth_token))
    return stack
