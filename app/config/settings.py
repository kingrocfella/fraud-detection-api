import os

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
TRAIN_BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", "2"))
TRAIN_EPOCHS = int(os.getenv("TRAIN_EPOCHS", "1"))
DATA_TRAIN_END = os.getenv("DATA_TRAIN_END", None)
