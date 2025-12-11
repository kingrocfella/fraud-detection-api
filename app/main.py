from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse

# Import queues module to register Dramatiq actors
import app.queues.job_queue  # type: ignore  # noqa: F401
from app.config import logger
from app.middlewares import HideServerHeadersMiddleware, LoggingMiddleware
from app.routes import (
    detect_fraud_router,
    finetune_model_router,
    health_router,
    jobs_router,
)

app = FastAPI(title="Nigerian Transactions Fraud Detection API", version="1.0.0")

# Add middlewares
app.add_middleware(LoggingMiddleware)
app.add_middleware(HideServerHeadersMiddleware)

# Include routes
app.include_router(detect_fraud_router)
app.include_router(finetune_model_router)
app.include_router(health_router)
app.include_router(jobs_router)


@app.exception_handler(404)
def not_found_handler(request: Request, _exc: HTTPException):
    """Handle 404 errors by redirecting to external URL."""
    logger.warning("404 Not Found: %s %s", request.method, request.url.path)
    return RedirectResponse(url="https://ash-speed.hetzner.com/10GB.bin")


@app.exception_handler(Exception)
def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions."""
    logger.error(
        "Unhandled exception: %s %s - %s",
        request.method,
        request.url.path,
        str(exc),
        exc_info=True,
    )
    raise HTTPException(status_code=500, detail="Internal server error") from exc
