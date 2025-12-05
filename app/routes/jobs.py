"""Unified job status routes."""

from fastapi import APIRouter, HTTPException, status

from app.config import logger
from app.queues import JOB_TYPE_FRAUD_DETECTION, JOB_TYPE_MODEL_TRAINING, get_job_status
from app.schemas import (
    FraudDetectionResult,
    JobStatusFailed,
    JobStatusPending,
    ModelTrainingResult,
)

router = APIRouter()


@router.get(
    "/job/{message_id}",
    status_code=status.HTTP_200_OK,
    response_model=FraudDetectionResult
    | ModelTrainingResult
    | JobStatusPending
    | JobStatusFailed,
)
def get_job_status_endpoint(
    message_id: str,
) -> FraudDetectionResult | ModelTrainingResult | JobStatusPending | JobStatusFailed:
    """Get the status of any background job.

    Returns the job status and result if completed.
    """
    try:
        status_info = get_job_status(message_id)
        job_type = status_info.get("job_type")

        # If job is finished, return appropriate result type
        if status_info["status"] == "finished" and status_info.get("result"):
            result = status_info["result"]

            # Return type based on job type
            if job_type == JOB_TYPE_FRAUD_DETECTION:
                return FraudDetectionResult(
                    response=result.get("response", ""),
                )
            elif job_type == JOB_TYPE_MODEL_TRAINING:
                return ModelTrainingResult(
                    status=result.get("status", ""),
                    message=result.get("message", ""),
                )
            else:
                raise ValueError(f"Unknown job type: {job_type}")

        # If job is pending
        if status_info["status"] == "pending":
            return JobStatusPending(
                message_id=message_id,
                status="pending",
                message=status_info.get("message", "Job is being processed"),
            )

        # If job failed or unknown status
        job_status = status_info["status"]
        if job_status not in ("failed", "unknown"):
            job_status = "unknown"
        return JobStatusFailed(
            message_id=message_id,
            status=job_status,  # type: ignore[arg-type]
            error=status_info.get("error", "Unknown error"),
        )

    except Exception as exc:
        logger.error("Failed to get job status: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve job status.",
        ) from exc
