from fastapi import APIRouter, HTTPException, status

from app.config import logger
from app.schemas import FraudDetectionRequest, JobQueuedResponse
from app.queues import enqueue_fraud_detection_job

router = APIRouter()


@router.post("/detect-fraud", response_model=JobQueuedResponse, status_code=status.HTTP_202_ACCEPTED)
def detect_fraud(request: FraudDetectionRequest) -> JobQueuedResponse:
    """Queue a fraud detection job.

    Returns a job ID that can be used to check the status via GET /job/{message_id}.
    """
    try:
        logger.info("Queueing fraud detection job")

        # Enqueue the job
        job_data = {"request": request.model_dump()}
        job_id = enqueue_fraud_detection_job(job_data)

        logger.info("Fraud detection job enqueued with message ID: %s", job_id)

        return JobQueuedResponse(
            message_id=job_id,
            status="queued",
            message="Job has been queued for processing. Use GET /job/{message_id} to check status.",
        )

    except Exception as e:
        logger.error(f"Error trying to queue fraud detection job: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to queue fraud detection job: {str(e)}"
        )

