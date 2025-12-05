"""Worker functions for processing fraud detection jobs."""

from typing import Any, Dict

from transformers import AutoModelForCausalLM, AutoTokenizer

from app.config import logger
from app.utils import generate_prompt

# Lazy-loaded model and tokenizer instances
_MODEL = None
_TOKENIZER = None


def _get_model_and_tokenizer():
    """Get or create model and tokenizer instances."""
    global _MODEL, _TOKENIZER  # pylint: disable=global-statement

    if _MODEL is None or _TOKENIZER is None:
        logger.info("Loading model and tokenizer for fraud detection...")
        _MODEL = AutoModelForCausalLM.from_pretrained("/app/models/merged")
        _TOKENIZER = AutoTokenizer.from_pretrained("/app/models/merged")
        logger.info("Model and tokenizer loaded successfully")

    return _MODEL, _TOKENIZER


def process_fraud_detection_job_sync(job_data: Dict[str, Any]) -> Dict[str, Any]:
    """Synchronous implementation of fraud detection job processing."""
    logger.info("Processing fraud detection job")

    try:
        # Get model and tokenizer
        model, tokenizer = _get_model_and_tokenizer()

        # Generate prompt from request
        prompt_data = generate_prompt(job_data["request"])
        prompt = prompt_data["instruction"]
        logger.info("Prompt generated successfully")

        # Tokenize inputs
        logger.info("Tokenizing inputs...")
        inputs = tokenizer(prompt, return_tensors="pt")
        logger.info("Inputs tokenized successfully")

        # Generate output
        logger.info("Generating output...")
        outputs = model.generate(**inputs, max_new_tokens=100)
        logger.info("Model generated output successfully")

        # Decode response
        logger.info("Decoding response...")
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info("Response decoded successfully")

        return {"response": response}

    except Exception as e:
        logger.error("Error processing fraud detection job: %s", e, exc_info=True)
        raise
