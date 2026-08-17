from fastapi import APIRouter, Header, HTTPException, Request, status

from app.config import FINETUNE_MODEL_SECURITY_KEY, logger
from app.queues import enqueue_model_training_job
from app.schemas import JobQueuedResponse
from app.security import SECURITY_KEY_HEADER, enforce_rate_limit, verify_security_key

router = APIRouter()


# POST, not GET: this queues a fine-tuning run, which is the most expensive
# thing this service can be asked to do. A GET invites a crawler, a prefetcher
# or a retried browser navigation to start one.
@router.post(
    "/finetune-model",
    response_model=JobQueuedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def finetune_model(
    http_request: Request,
    key: str = Header(default="", alias=SECURITY_KEY_HEADER),
) -> JobQueuedResponse:
    """Queue a model fine-tuning job.

    The access-control key is an `X-API-Key` header, not a query parameter.

    Returns a job ID that can be used to check the status via GET /job/{message_id}.
    """
    client_host = http_request.client.host if http_request.client else "unknown"
    enforce_rate_limit("finetune-model", client_host)
    verify_security_key(key, FINETUNE_MODEL_SECURITY_KEY)

    try:
        logger.info("Queueing model fine-tuning job")

        # Enqueue the job
        job_data = {}
        job_id = enqueue_model_training_job(job_data)

        logger.info("Model training job enqueued with message ID: %s", job_id)

        return JobQueuedResponse(
            message_id=job_id,
            status="queued",
            message="Job has been queued for processing. Use GET /job/{message_id} to check status.",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Error trying to queue model training job: %s", str(e), exc_info=True
        )
        raise HTTPException(
            status_code=500, detail="Failed to queue model training job"
        ) from e
