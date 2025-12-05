import os

TRAIN_BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", "2"))
TRAIN_EPOCHS = int(os.getenv("TRAIN_EPOCHS", "1"))
DATA_TRAIN_END = os.getenv("DATA_TRAIN_END", None)
MODEL_NAME = os.getenv("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
LOW_CPU_MEM_USAGE = os.getenv("LOW_CPU_MEM_USAGE", "True").lower() in ("true",)
TRAINING_SECURITY_KEY = os.getenv("TRAINING_SECURITY_KEY", "")
