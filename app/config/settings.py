import os
from typing import cast

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
TRAIN_BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", "2"))
TRAIN_EPOCHS = int(os.getenv("TRAIN_EPOCHS", "1"))
DATA_TRAIN_END = os.getenv("DATA_TRAIN_END", None)
MODEL_NAME = os.getenv("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
LOW_CPU_MEM_USAGE = cast(bool, os.getenv("LOW_CPU_MEM_USAGE", True))

