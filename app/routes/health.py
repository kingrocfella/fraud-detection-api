import logging

from fastapi import APIRouter

from app.schemas import DefaultResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health", response_model=DefaultResponse)
def health_check() -> DefaultResponse:
    """Check the health of the API"""
    logger.info("Health check endpoint called")
    return DefaultResponse(response="Fraud Detection API is healthy")
