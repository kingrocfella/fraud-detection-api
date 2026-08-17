import os

TRAIN_BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", "2"))
TRAIN_EPOCHS = int(os.getenv("TRAIN_EPOCHS", "1"))
TRAIN_MAX_STEPS = int(os.getenv("TRAIN_MAX_STEPS", "1000"))
DATA_TRAIN_END = os.getenv("DATA_TRAIN_END", None)
MODEL_NAME = os.getenv("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
LOW_CPU_MEM_USAGE = os.getenv("LOW_CPU_MEM_USAGE", "True").lower() in ("true",)
DETECT_FRAUD_SECURITY_KEY = os.getenv("DETECT_FRAUD_SECURITY_KEY", "")
FINETUNE_MODEL_SECURITY_KEY = os.getenv("FINETUNE_MODEL_SECURITY_KEY", "")
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "256"))

ENVIRONMENT = os.getenv("ENVIRONMENT", "development").strip().lower()
IS_PRODUCTION = ENVIRONMENT in {"prod", "production"}

# --- Request protection ---------------------------------------------------
MAX_REQUEST_BODY_BYTES = int(os.getenv("MAX_REQUEST_BODY_BYTES", str(1024 * 1024)))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))
ENABLE_HSTS = IS_PRODUCTION

# --- Rate limiting (per-client fixed window on the expensive endpoints) ----
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "30"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))


def validate_production_config() -> None:
    """Fail fast in production if access-control keys are unset or placeholders.

    Called at startup. In development the placeholder keys are allowed so the
    stack runs locally; the per-request check still fails closed on an empty
    configured key regardless of environment.
    """
    if not IS_PRODUCTION:
        return
    for name, value in (
        ("DETECT_FRAUD_SECURITY_KEY", DETECT_FRAUD_SECURITY_KEY),
        ("FINETUNE_MODEL_SECURITY_KEY", FINETUNE_MODEL_SECURITY_KEY),
    ):
        if len(value) < 16 or value.lower().startswith("change-me"):
            raise RuntimeError(
                f"{name} must be set to a real non-placeholder value "
                "(>=16 chars) when ENVIRONMENT=production"
            )
