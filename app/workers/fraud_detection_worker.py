"""Worker functions for processing fraud detection jobs."""

from typing import Any, Dict

from transformers import AutoModelForCausalLM, AutoTokenizer

from app.config import logger
from app.utils import generate_prompt

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
        prompt_data = generate_prompt(
            job_data["request"],
            state_reasoning="State the primary reason for your decision.",
        )
        instruction = prompt_data["instruction"].strip()
        prompt = f"{instruction}\n\nAnswer:"
        logger.info("Prompt generated successfully")

        # Tokenize inputs
        logger.info("Tokenizing inputs...")
        inputs = tokenizer(prompt, return_tensors="pt")
        logger.info("Inputs tokenized successfully")

        logger.info("Generating output...")
        outputs = model.generate(
            **inputs,
            max_new_tokens=1000,
            do_sample=True,
            temperature=0.5,
            top_p=0.9,
            top_k=50,
            num_beams=1,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        logger.info("Model generated output successfully")

        # Decode response
        logger.info("Decoding response...")
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info("Response decoded successfully: %s", decoded)

        # Extract the part after "Answer:" and strip
        if "Answer:" in decoded:
            response = decoded.split("Answer:", 1)[1].strip()
        else:
            response = decoded.strip()

        logger.info("Response extracted successfully: %s", response)

        return {"response": response}

    except Exception as e:
        logger.error("Error processing fraud detection job: %s", e, exc_info=True)
        raise
