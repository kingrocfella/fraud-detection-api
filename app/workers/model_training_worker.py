"""Worker functions for processing model training jobs."""

from typing import Any, Dict

import torch  # Import torch for dtype
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from app.config import (
    LOW_CPU_MEM_USAGE,
    MODEL_NAME,
    TRAIN_EPOCHS,
    TRAIN_MAX_STEPS,
    logger,
)
from app.utils import generate_prompts_from_dataset


def process_model_training_job_sync(_job_data: Dict[str, Any]) -> Dict[str, Any]:
    """Optimized synchronous implementation of model training job processing for CPU-only, low-RAM systems."""
    logger.info("Starting optimized model training job on CPU")

    try:
        # --- 1. Load Tokenizer ---
        logger.info("Loading tokenizer for %s...", MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer loaded successfully")

        # --- 2. Configure Quantization (Memory Optimization) ---
        # Although BitsAndBytes is typically for GPU, we can use torch_dtype=torch.bfloat16
        # if the CPU supports it, or use the most memory-efficient dtype available.
        # For strict low-RAM CPU, we stick to torch.float32 unless we are sure of bfloat16 support.
        # We will use the model's default low precision loading where possible.

        # --- 3. Load Base Model on CPU with Dtype Optimization ---
        logger.info("Loading model %s...", MODEL_NAME)

        # Use a lower precision data type if your CPU supports it, otherwise stick to the most
        # memory-conservative default loading which low_cpu_mem_usage=True helps with.
        # We enforce torch.float32 as a safe default for CPU fine-tuning.
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map={"": "cpu"},
            low_cpu_mem_usage=LOW_CPU_MEM_USAGE,
            torch_dtype=torch.float32,  # Explicitly use float32 or torch.bfloat16 if supported
        )
        logger.info("Model loaded successfully")

        # --- 4. Configure LoRA (Critical for Low-RAM) ---
        logger.info("Configuring LoRA...")
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.05,
            target_modules="all-linear",  # Target all linear layers for maximum effect
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        logger.info("LoRA configured. Trainable params:")
        model.print_trainable_parameters()

        # --- 5. Load and Prepare Dataset ---
        logger.info("Loading dataset...")
        dataset = generate_prompts_from_dataset()
        logger.info("Dataset loaded with %d samples", len(dataset))

        # Limit dataset to keep training time short on CPU
        max_train_examples = min(len(dataset), TRAIN_MAX_STEPS)
        if max_train_examples < len(dataset):
            dataset = dataset.select(range(max_train_examples))
            logger.info("Truncated dataset to %d samples to match TRAIN_MAX_STEPS", max_train_examples)

        # --- 6. Tokenize Dataset (single process to avoid worker deadlocks) ---
        logger.info("Tokenizing dataset (single process)...")

        # Use fewer tokens/shorter max_length to save memory during training
        MAX_SEQ_LENGTH = 384

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
                max_length=MAX_SEQ_LENGTH,  # Reduced max_length to save RAM
                return_tensors=None,
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        tokenized_dataset = dataset.map(
            tokenize_func,
            batched=True,
            remove_columns=["instruction", "output"],  # Remove original columns to save memory
        )
        logger.info("Dataset tokenized successfully")

        # --- 7. Gradient Checkpointing (Memory Optimization) ---
        logger.info("Enabling gradient checkpointing...")
        # Note: This increases training time but drastically reduces peak memory usage.
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled successfully")

        # --- 8. Configure Trainer (Efficiency and Memory Settings) ---
        logger.info("Configuring trainer...")

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=tokenized_dataset,
            args=SFTConfig(
                output_dir="/app/models",
                per_device_train_batch_size=1,
                gradient_accumulation_steps=8,  # Keep small but reasonable on CPU
                learning_rate=2e-4,
                num_train_epochs=TRAIN_EPOCHS,
                max_steps=TRAIN_MAX_STEPS,
                logging_steps=10,
                save_strategy="no",  # Avoid checkpoint I/O on CPU
                # CPU-specific settings
                use_cpu=True,
                # Disable all GPU-specific or high-precision training
                bf16=False,
                fp16=False,
                torch_compile=False,
                ddp_find_unused_parameters=False,
                # Memory saving flag
                # This helps aggressively remove tensors that are no longer needed
                # Can sometimes be slower, but critical for low-RAM.
                remove_unused_columns=True,
                dataloader_num_workers=0,  # Avoid multiprocessing dataloader deadlocks on CPU
            ),
        )
        logger.info("Trainer configured successfully")

        # --- 9. Train ---
        logger.info("Starting training...")
        trainer.train()
        logger.info("Training completed successfully!")

        logger.info("Merging LoRA weights into base model...")

        # The model object is already the trained PeftModel (LoRA adapted model)
        # Use merge_and_unload to convert the PeftModel wrapper into a standard AutoModelForCausalLM
        # with weights baked in. This is much safer and easier than loading from a specific checkpoint path.
        merged_model = model.merge_and_unload()

        logger.info("LoRA weights merged successfully")

        # Save the merged model and tokenizer
        logger.info("Saving merged model and tokenizer...")

        # Save the full, merged model
        merged_model.save_pretrained("/app/models/merged")

        # Save the tokenizer (which was already configured)
        tokenizer.save_pretrained("/app/models/merged")

        logger.info("Model and tokenizer saved successfully to /app/models/merged")

        return {
            "status": "success",
            "message": "Model finetuned successfully",
        }

    except Exception as e:
        logger.error("Error during model training: %s", e, exc_info=True)
        # Re-raise the exception to ensure the job fails
        raise
