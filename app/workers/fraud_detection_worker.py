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
        prompt_data = generate_prompt(job_data["request"])
        instruction = prompt_data["instruction"].strip()
        prompt = f"{instruction}\n\nAnswer:"
        logger.info("Prompt generated successfully")

        # Tokenize inputs
        logger.info("Tokenizing inputs...")
        inputs = tokenizer(prompt, return_tensors="pt")
        logger.info("Inputs tokenized successfully")

        # Generate output (short answer with light sampling to allow "guessing")
        logger.info("Generating output...")
        outputs = model.generate(
            **inputs,
            max_new_tokens=12,  # still short
            do_sample=True,
            temperature=0.85,
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
        # Extract text after the explicit Answer: cue, then take a short phrase
        answer_text = (
            decoded.split("Answer:", 1)[1].strip()
            if "Answer:" in decoded
            else decoded.strip()
        )
        tokens = answer_text.split()
        response = " ".join(tokens[:3]) if tokens else ""
        logger.info("Response decoded successfully: %s", response)

        return {"response": response}

    except Exception as e:
        logger.error("Error processing fraud detection job: %s", e, exc_info=True)
        raise
