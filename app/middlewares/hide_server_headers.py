from fastapi import HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable

from app.config import logger


class HideServerHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to hide server headers."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Dispatch the middleware."""
        try:
            response = await call_next(request)

            for header in ["server", "x-powered-by", "date"]:
                if header in response.headers:
                    del response.headers[header]

            return response
        except Exception as e:
            logger.error(
                "Error in HideServerHeadersMiddleware: %s", str(e), exc_info=True
            )
            raise HTTPException(status_code=500, detail="Internal server error") from e
