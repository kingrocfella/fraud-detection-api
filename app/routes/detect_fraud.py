from fastapi import APIRouter, HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.config import logger
from app.schemas import DefaultResponse, FraudDetectionRequest
from app.utils import generate_prompt

router = APIRouter()


@router.post("/detect-fraud", response_model=DefaultResponse)
def detect_fraud(request: FraudDetectionRequest) -> DefaultResponse:
    """Detect fraud"""

    try:
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained("/app/models/merged")
        logger.info("Model loaded successfully")
        tokenizer = AutoTokenizer.from_pretrained("/app/models/merged")
        logger.info("Tokenizer loaded successfully")

        # Generate prompt from request
        prompt_data = generate_prompt(request)
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

        return DefaultResponse(response=response)

    except Exception as e:
        logger.error(f"Error trying to detect fraud: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to detect fraud: {str(e)}")
