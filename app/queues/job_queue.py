"""Dramatiq queue utilities for background job processing."""

import os
from typing import Any, Dict

import dramatiq
import redis

from app.config import logger
from app.database.redis import get_redis_broker, get_redis_url, get_result_backend
from app.workers import (
    process_fraud_detection_job_sync,
    process_model_training_job_sync,
)

redis_broker = get_redis_broker()
result_backend = get_result_backend()

# Redis client for job type tracking
_redis_client: redis.Redis | None = None


def _get_redis_client() -> redis.Redis:
    """Get Redis client for job type tracking."""
    global _redis_client  # pylint: disable=global-statement
    if _redis_client is None:
        _redis_client = redis.from_url(get_redis_url())
    return _redis_client


# Job type constants
JOB_TYPE_FRAUD_DETECTION = "fraud_detection"
JOB_TYPE_MODEL_TRAINING = "model_training"
JOB_TYPE_KEY_PREFIX = "job:type:"
JOB_TYPE_TTL = 86400 * int(os.getenv("JOB_TYPE_TTL_DAYS", "7"))

# Set the broker for dramatiq
dramatiq.set_broker(redis_broker)


def _store_job_type(message_id: str, job_type: str) -> None:
    """Store job type in Redis for later lookup."""
    client = _get_redis_client()
    key = f"{JOB_TYPE_KEY_PREFIX}{message_id}"
    client.setex(key, JOB_TYPE_TTL, job_type)
    logger.debug("Stored job type '%s' for message_id: %s", job_type, message_id)


# =============================================================================
# Fraud Detection Processing
# =============================================================================


@dramatiq.actor(store_results=True, max_retries=3, time_limit=600000)
def process_fraud_detection_job(job_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process a fraud detection job using Dramatiq."""
    try:
        logger.info("Starting fraud detection job processing")
        result = process_fraud_detection_job_sync(job_data)
        logger.info("Fraud detection job completed successfully")
        return result
    except Exception as e:
        logger.error("Error processing fraud detection job: %s", e, exc_info=True)
        raise


def enqueue_fraud_detection_job(job_data: Dict[str, Any]) -> str:
    """Enqueue a fraud detection processing job."""
    message = process_fraud_detection_job.send(job_data)
    _store_job_type(message.message_id, JOB_TYPE_FRAUD_DETECTION)
    logger.info("Enqueued fraud detection job with message ID: %s", message.message_id)
    return message.message_id


# =============================================================================
# Model Training Processing
# =============================================================================


@dramatiq.actor(store_results=True, max_retries=1, time_limit=3600000)
def process_model_training_job(job_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process a model training job using Dramatiq."""
    try:
        logger.info("Starting model training job processing")
        result = process_model_training_job_sync(job_data)
        logger.info("Model training job completed successfully")
        return result
    except Exception as e:
        logger.error("Error processing model training job: %s", e, exc_info=True)
        raise


def enqueue_model_training_job(job_data: Dict[str, Any]) -> str:
    """Enqueue a model training processing job."""
    message = process_model_training_job.send(job_data)
    _store_job_type(message.message_id, JOB_TYPE_MODEL_TRAINING)
    logger.info("Enqueued model training job with message ID: %s", message.message_id)
    return message.message_id
