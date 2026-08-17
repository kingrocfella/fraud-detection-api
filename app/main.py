from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

# Import queues module to register Dramatiq actors
import app.queues.job_queue  # type: ignore  # noqa: F401
from app.config import (
    ENABLE_HSTS,
    MAX_REQUEST_BODY_BYTES,
    REQUEST_TIMEOUT_SECONDS,
    logger,
    validate_production_config,
)
from app.middlewares import (
    HideServerHeadersMiddleware,
    LoggingMiddleware,
    RequestProtectionMiddleware,
)
from app.routes import (
    detect_fraud_router,
    finetune_model_router,
    health_router,
    jobs_router,
)

# Fail fast if production is started with unset/placeholder access-control keys.
validate_production_config()

app = FastAPI(title="Nigerian Transactions Fraud Detection API", version="1.0.0")

# Add middlewares. RequestProtectionMiddleware is added last so it is the
# outermost layer: it caps the request body, enforces the request deadline, and
# stamps security headers on every response (including its own 413/504 errors).
app.add_middleware(LoggingMiddleware)
app.add_middleware(HideServerHeadersMiddleware)
app.add_middleware(
    RequestProtectionMiddleware,
    max_body_bytes=MAX_REQUEST_BODY_BYTES,
    timeout_seconds=REQUEST_TIMEOUT_SECONDS,
    enable_hsts=ENABLE_HSTS,
)

# Include routes
app.include_router(detect_fraud_router)
app.include_router(finetune_model_router)
app.include_router(health_router)
app.include_router(jobs_router)


@app.exception_handler(404)
def not_found_handler(request: Request, _exc: HTTPException) -> JSONResponse:
    """Return a small JSON 404 for unmatched paths."""
    logger.warning("404 Not Found: %s %s", request.method, request.url.path)
    return JSONResponse(status_code=404, content={"detail": "Not found"})


@app.exception_handler(Exception)
def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle all unhandled exceptions without leaking internals to the client."""
    logger.error(
        "Unhandled exception: %s %s - %s",
        request.method,
        request.url.path,
        str(exc),
        exc_info=True,
    )
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
