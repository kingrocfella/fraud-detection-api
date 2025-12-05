import logging

import torch
from fastapi import APIRouter, HTTPException
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from app.config import TRAIN_BATCH_SIZE, TRAIN_EPOCHS
from app.schemas import DefaultResponse
from app.utils import generate_prompts_from_dataset

router = APIRouter()
logger = logging.getLogger(__name__)

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@router.get("/finetune-model", response_model=DefaultResponse)
def finetune_model() -> DefaultResponse:
    """Finetune the model"""
    try:
        logger.info("Starting fine-tuning process...")

        # Load tokenizer
        logger.info(f"Loading tokenizer for {MODEL_NAME}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer loaded successfully")

        # Load base model on CPU
        logger.info(f"Loading model {MODEL_NAME}...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        logger.info("Model loaded successfully")

        # Configure LoRA
        logger.info("Configuring LoRA...")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        logger.info(
            f"LoRA configured. Trainable params: {model.print_trainable_parameters()}"
        )

        # Load and prepare dataset
        logger.info("Loading dataset...")
        dataset = generate_prompts_from_dataset()
        logger.info(f"Dataset loaded with {len(dataset)} samples")

        # Tokenize dataset
        logger.info("Tokenizing dataset...")

        def tokenize_func(batch):
            # Combine instruction and output for causal LM training
            # Format: <instruction>\n\nAnswer: <output>
            texts = [
                f"{inst}\n\nAnswer: {out}"
                for inst, out in zip(batch["instruction"], batch["output"])
            ]

            # Tokenize the combined text
            tokenized = tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors=None,  # Return lists, not tensors
            )

            # For causal LM, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].copy()

            return tokenized

        tokenized_dataset = dataset.map(tokenize_func, batched=True)
        logger.info("Dataset tokenized successfully")

        # Configure trainer
        logger.info("Configuring trainer...")
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=tokenized_dataset,
            args=SFTConfig(
                output_dir="/app/models",
                per_device_train_batch_size=TRAIN_BATCH_SIZE,
                gradient_accumulation_steps=4,
                learning_rate=2e-4,
                num_train_epochs=TRAIN_EPOCHS,
                logging_steps=1,
                save_strategy="epoch",
                bf16=False,
                fp16=False,
                use_cpu=True,
            ),
        )
        logger.info("Trainer configured successfully")

        # Train
        logger.info("Starting training...")
        trainer.train()
        logger.info("Training completed successfully!")

        # Merge LoRA weights into base model
        logger.info("Merging LoRA weights into base model...")
        model = PeftModel.from_pretrained(model, "/app/models/checkpoint-2")
        logger.info("LoRA weights merged successfully")

        # Save the merged model and tokenizer
        logger.info("Saving merged model and tokenizer...")
        model.save_pretrained("/app/models/merged")
        tokenizer.save_pretrained("/app/models/merged")
        logger.info("Model and tokenizer saved successfully")

        return DefaultResponse(response="Model finetuned successfully")

    except Exception as e:
        logger.error(f"Error during fine-tuning: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Fine-tuning failed: {str(e)}")
