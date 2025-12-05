from .detect_fraud import router as detect_fraud_router
from .finetune_model import router as finetune_model_router
from .health import router as health_router

__all__ = [
    "detect_fraud_router",
    "health_router",
    "finetune_model_router",
]
