from fastapi import APIRouter, HTTPException, status

from app.config import logger
from app.schemas import JobQueuedResponse
from app.queues import enqueue_model_training_job

router = APIRouter()


@router.get("/finetune-model", response_model=JobQueuedResponse, status_code=status.HTTP_202_ACCEPTED)
def finetune_model() -> JobQueuedResponse:
    """Queue a model fine-tuning job.

    Returns a job ID that can be used to check the status via GET /job/{message_id}.
    """
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

    except Exception as e:
        logger.error(f"Error trying to queue model training job: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to queue model training job: {str(e)}"
        )

