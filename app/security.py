"""Access-control helpers: constant-time key checks and a Redis fixed-window
rate limiter shared across API replicas."""

import hashlib
import hmac
import time

import redis as redis_lib
from fastapi import HTTPException, status

from app.config import (
    IS_PRODUCTION,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
    logger,
)
from app.database.redis import get_redis_url

_client: "redis_lib.Redis | None" = None

# The access-control key travels in a header, never a query parameter: a URL is
# recorded by every proxy, access log and browser history between the caller and
# this process, and no amount of log sanitising here reaches those.
SECURITY_KEY_HEADER = "X-API-Key"


def _get_client() -> "redis_lib.Redis":
    global _client
    if _client is None:
        _client = redis_lib.Redis.from_url(
            get_redis_url(),
            socket_connect_timeout=2,
            socket_timeout=2,
        )
    return _client


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def verify_any_security_key(provided: str, configured: tuple[str, ...]) -> None:
    """Accept the request if the key matches any of the configured keys.

    Used by the job-status route, which serves results for both job types and
    so must admit either caller. Every candidate is compared even after a match
    so the comparison cost does not depend on which key was supplied.
    """
    usable = tuple(value for value in configured if value)
    if not usable:
        logger.error("No access-control key is configured; refusing request")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service temporarily unavailable",
        )
    matched = False
    for candidate in usable:
        if hmac.compare_digest(provided or "", candidate):
            matched = True
    if not matched:
        logger.warning("Rejected request with an invalid security key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid security key"
        )


def verify_security_key(provided: str, configured: str) -> None:
    """Constant-time comparison that fails closed on an unset configured key.

    Never logs the provided value. An empty configured key means access control
    is misconfigured, so it is rejected rather than matching an empty input.
    """
    if not configured:
        logger.error("Access-control key is not configured; refusing request")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service temporarily unavailable",
        )
    if not hmac.compare_digest(provided or "", configured):
        logger.warning("Rejected request with an invalid security key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid security key"
        )


def enforce_rate_limit(action: str, identity: str) -> None:
    """Enforce a Redis-backed fixed-window quota. Fails closed in production."""
    window = RATE_LIMIT_WINDOW_SECONDS
    bucket = int(time.time()) // window
    key = f"rate:v1:{action}:{_fingerprint(identity)}:{bucket}"
    try:
        pipeline = _get_client().pipeline()
        pipeline.incr(key)
        pipeline.expire(key, window + 1)
        count, _ = pipeline.execute()
    except Exception as exc:  # pragma: no cover - backend availability
        logger.error("Rate limiter backend error: %s", exc)
        if IS_PRODUCTION:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service temporarily unavailable",
            ) from exc
        return
    if int(count) > RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests",
            headers={"Retry-After": str(window)},
        )
