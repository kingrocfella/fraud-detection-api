"""Worker functions for processing model training jobs."""

from typing import Any, Dict

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from app.config import (
    LOW_CPU_MEM_USAGE,
    MODEL_NAME,
    TRAIN_EPOCHS,
    TRAIN_MAX_STEPS,
    MAX_SEQ_LENGTH,
    logger,
)
from app.utils import generate_prompts_from_dataset


def process_model_training_job_sync(_job_data: Dict[str, Any]) -> Dict[str, Any]:
    """Optimized synchronous implementation of model training job processing for CPU-only, low-RAM systems."""
    logger.info("Starting optimized model training job on CPU")

    try:
        logger.info("Loading tokenizer for %s...", MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer loaded successfully")

        logger.info("Loading model %s...", MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map={"": "cpu"},
            low_cpu_mem_usage=LOW_CPU_MEM_USAGE,
            torch_dtype=torch.float32,
        )
        logger.info("Model loaded successfully")

        logger.info("Configuring LoRA...")
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.05,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        logger.info("LoRA configured. Trainable params:")
        model.print_trainable_parameters()

        # Load and generate prompts from dataset
        logger.info("Loading dataset...")
        dataset = generate_prompts_from_dataset()
        logger.info("Dataset loaded with %d samples", len(dataset))

        logger.info("Tokenizing dataset (single process)...")
        def tokenize_func(batch):
            # Combine instruction and output for causal LM training
            texts = [
                f"{inst}\n\nAnswer: {out}"
                for inst, out in zip(batch["instruction"], batch["output"])
            ]

            tokenized = tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=MAX_SEQ_LENGTH,
                return_tensors=None,
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        tokenized_dataset = dataset.map(
            tokenize_func,
            batched=True,
            remove_columns=[
                "instruction",
                "output",
            ],
        )
        logger.info("Dataset tokenized successfully")

        logger.info("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled successfully")

        logger.info("Configuring trainer...")

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=tokenized_dataset,
            args=SFTConfig(
                output_dir="/app/models",
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                learning_rate=2e-4,
                num_train_epochs=TRAIN_EPOCHS,
                max_steps=TRAIN_MAX_STEPS,
                logging_steps=10,
                save_strategy="no",
                use_cpu=True,
                bf16=False,
                fp16=False,
                torch_compile=False,
                ddp_find_unused_parameters=False,
                remove_unused_columns=True,
                dataloader_num_workers=0,
            ),
        )
        logger.info("Trainer configured successfully")

        logger.info("Starting training...")
        trainer.train()
        logger.info("Training completed successfully!")

        logger.info("Merging LoRA weights into base model...")

        # The model object is already the trained PeftModel (LoRA adapted model)
        # Use merge_and_unload to convert the PeftModel wrapper into a standard AutoModelForCausalLM
        # with weights baked in. This is much safer and easier than loading from a specific checkpoint path.
        merged_model = model.merge_and_unload()

        logger.info("LoRA weights merged successfully")

        logger.info("Saving merged model and tokenizer...")
        merged_model.save_pretrained("/app/models/merged")
        tokenizer.save_pretrained("/app/models/merged")

        logger.info("Model and tokenizer saved successfully to /app/models/merged")

        return {
            "status": "success",
            "message": "Model finetuned successfully",
        }

    except Exception as e:
        logger.error("Error during model training: %s", e, exc_info=True)
        raise
