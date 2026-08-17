"""Unified job status routes."""

from fastapi import APIRouter, Header, HTTPException, Request, status

from app.config import DETECT_FRAUD_SECURITY_KEY, FINETUNE_MODEL_SECURITY_KEY, logger
from app.queues import JOB_TYPE_FRAUD_DETECTION, JOB_TYPE_MODEL_TRAINING, get_job_status
from app.schemas import (
    FraudDetectionResult,
    JobStatusFailed,
    JobStatusPending,
    ModelTrainingResult,
)
from app.security import SECURITY_KEY_HEADER, enforce_rate_limit, verify_any_security_key

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
    http_request: Request,
    key: str = Header(default="", alias=SECURITY_KEY_HEADER),
) -> FraudDetectionResult | ModelTrainingResult | JobStatusPending | JobStatusFailed:
    """Get the status of any background job.

    This route serves the *results* — a fraud verdict on a real transaction, or
    a training outcome — so it is gated and rate limited like the routes that
    create the jobs. It accepts either configured key, because it answers for
    both job types. Without the gate a message ID is a bearer capability that
    anyone reaching this service could probe for, without bound.

    Returns the job status and result if completed.
    """
    client_host = http_request.client.host if http_request.client else "unknown"
    enforce_rate_limit("job-status", client_host)
    verify_any_security_key(
        key, (DETECT_FRAUD_SECURITY_KEY, FINETUNE_MODEL_SECURITY_KEY)
    )

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
