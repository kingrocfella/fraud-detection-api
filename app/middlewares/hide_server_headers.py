from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable


class HideServerHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to hide server headers."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Dispatch the middleware."""
        response = await call_next(request)

        # Remove headers
        for header in ["server", "x-powered-by", "date"]:
            response.headers.pop(header, None)

        return response
