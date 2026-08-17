from fastapi import APIRouter, Header, HTTPException, Request, status

from app.config import DETECT_FRAUD_SECURITY_KEY, logger
from app.queues import enqueue_fraud_detection_job
from app.schemas import FraudDetectionRequest, JobQueuedResponse
from app.security import SECURITY_KEY_HEADER, enforce_rate_limit, verify_security_key

router = APIRouter()


@router.post(
    "/detect-fraud",
    response_model=JobQueuedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def detect_fraud(
    payload: FraudDetectionRequest,
    http_request: Request,
    key: str = Header(default="", alias=SECURITY_KEY_HEADER),
) -> JobQueuedResponse:
    """Queue a fraud detection job.

    The access-control key is an `X-API-Key` header, not a query parameter: a
    key in a URL is copied into every proxy and access log on the path.

    Returns a job ID that can be used to check the status via GET /job/{message_id}.
    """
    client_host = http_request.client.host if http_request.client else "unknown"
    enforce_rate_limit("detect-fraud", client_host)
    verify_security_key(key, DETECT_FRAUD_SECURITY_KEY)

    try:
        logger.info("Queueing fraud detection job")

        # Enqueue the job
        job_data = {"request": payload.model_dump()}
        job_id = enqueue_fraud_detection_job(job_data)

        logger.info("Fraud detection job enqueued with message ID: %s", job_id)

        return JobQueuedResponse(
            message_id=job_id,
            status="queued",
            message="Job has been queued for processing. Use GET /job/{message_id} to check status.",
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(
            "Error trying to queue fraud detection job: %s", str(e), exc_info=True
        )
        raise HTTPException(
            status_code=500, detail="Failed to queue fraud detection job"
        ) from e
