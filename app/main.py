from fastapi import FastAPI

from app.middlewares import LoggingMiddleware
from app.routes import (
    detect_fraud_router,
    finetune_model_router,
    health_router,
    jobs_router,
)

# Import queues module to register Dramatiq actors
import app.queues.job_queue  # type: ignore  # noqa: F401

app = FastAPI(title="Nigerian Transactions Fraud Detection API", version="1.0.0")

# Add logging middleware
app.add_middleware(LoggingMiddleware)

app.include_router(detect_fraud_router)
app.include_router(finetune_model_router)
app.include_router(health_router)
app.include_router(jobs_router)
