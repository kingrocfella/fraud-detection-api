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
            logger.debug("Response headers before removal: %s", dict(response.headers))

            for header in ["server", "x-powered-by", "date"]:
                if header in response.headers:
                    logger.debug("Removing header: %s", header)
                    del response.headers[header]
            
            logger.debug("Response headers after removal: %s", dict(response.headers))

            return response
        except Exception as e:
            logger.error(
                "Error in HideServerHeadersMiddleware: %s", str(e), exc_info=True
            )
            raise HTTPException(status_code=500, detail="Internal server error") from e
