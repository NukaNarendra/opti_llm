import os

# --- CRITICAL FIX FOR TENSOR CORRUPTION ---
# Hides the second GPU so the Trainer runs in pure single-GPU mode.
# Prevents DataParallel from scrambling memory and causing the n_classes crash!
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NCCL_NVML_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
# ------------------------------------------

import json
import logging
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional
from datasets import Dataset as HFDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
    Trainer
)
from transformers.trainer_utils import get_last_checkpoint
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class TrainingConfig:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "data"
        self.processed_dir = self.data_dir / "processed"
        self.output_dir = self.base_dir / "models" / "opti_student_model"
        self.input_file = self.processed_dir / "distilled_train.jsonl"
        self.student_model_id = "microsoft/Phi-3.5-mini-instruct"
        self.max_seq_length = 2048
        self.per_device_train_batch_size = 4
        self.gradient_accumulation_steps = 4
        self.learning_rate = 2e-4
        self.num_train_epochs = 3
        self.warmup_ratio = 0.05
        self.logging_steps = 10
        self.save_steps = 200
        self.lora_r = 16
        self.lora_alpha = 32
        self.lora_dropout = 0.05
        self.target_modules = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
        self._prepare_directories()

    def _prepare_directories(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

class PerformanceMonitor(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            loss = logs.get("loss", 0)
            step = state.global_step
            logger.info(f"Training Step {step} | Loss: {loss:.4f}")

class DataFormatter:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.system_prompt = (
            "You are an expert AI financial analyst. You must respond ONLY with valid, raw JSON. "
            "Do not include markdown formatting, backticks, or any conversational text. "
            "Ensure the JSON is strictly formatted."
        )

    def _build_user_prompt(self, task_type: str, text: str, ticker: str) -> str:
        context = f"COMPANY: {ticker}\nTEXT:\n{text}\n\n"
        if task_type == "summarize":
            instruction = "Summarize the core financial and operational insights."
        elif task_type == "extract_risk":
            instruction = "Extract specific risks."
        elif task_type == "analyze_controls":
            instruction = "Identify internal controls or accounting metrics."
        elif task_type == "generate_hallucination":
            instruction = "Generate a highly plausible but completely fake financial statistic or risk based on the text context."
        else:
            instruction = "Extract insights."
        return f"{context}INSTRUCTION:\n{instruction}"

    def prepare_dataset(self) -> HFDataset:
        logger.info(f"Loading distilled data from {self.config.input_file}")
        valid_records = []
        error_count = 0
        
        if not self.config.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.config.input_file}")
            
        with open(self.config.input_file, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                    teacher_response = record.get("teacher_response", {})
                    
                    if not teacher_response or teacher_response.get("parse_error"):
                        error_count += 1
                        continue
                        
                    user_prompt = self._build_user_prompt(
                        record["task_type"], 
                        record["input_text"], 
                        record["ticker"]
                    )
                    
                    assistant_response = json.dumps(teacher_response)
                    
                    formatted_record = {
                        "messages": [
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": user_prompt},
                            {"role": "assistant", "content": assistant_response}
                        ]
                    }
                    valid_records.append(formatted_record)
                except Exception as e:
                    error_count += 1
                    
        logger.info(f"Successfully loaded {len(valid_records)} clean training records. Skipped {error_count} corrupted outputs.")
        return HFDataset.from_list(valid_records)

class StudentModelBuilder:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def initialize_tokenizer(self) -> AutoTokenizer:
        logger.info("Initializing Student Tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.student_model_id,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = 32000  # <|endoftext|> for Phi
            tokenizer.pad_token = "<|endoftext|>"
            
        tokenizer.padding_side = "right"
        return tokenizer

    def initialize_model(self) -> AutoModelForCausalLM:
        logger.info("Initializing Student Base Model with 4-bit Quantization")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            self.config.student_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        model.config.use_cache = False
        model.config.pretraining_tp = 1
        model = prepare_model_for_kbit_training(model)
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules
        )
        
        logger.info("Applying LoRA Adapters to Base Model")
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        return model

class SafeDataCollator:
    """
    A custom, foolproof collator designed to prevent 'CUDA device-side assert' 
    errors by mathematically guaranteeing that no out-of-bounds tokens or 
    unmasked padding tokens ever enter the loss function.
    """
    def __init__(self, tokenizer, vocab_size):
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size

    def __call__(self, features):
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt"
        )
        
        input_ids = batch["input_ids"]
        
        # 1. Force all input_ids strictly into the safe embedding range
        input_ids = torch.clamp(input_ids, min=0, max=self.vocab_size - 1)
        batch["input_ids"] = input_ids
        
        # 2. Create labels and explicitly mask out padded areas
        labels = input_ids.clone()
        labels[batch["attention_mask"] == 0] = -100
        
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
            
        # 3. The Ultimate Fail-Safe: Neutralize any rogue IDs that survived
        labels[(labels < 0) & (labels != -100)] = -100
        labels[labels >= self.vocab_size] = -100
        
        batch["labels"] = labels
        return batch

class OptiTrainer:
    def __init__(self, base_dir: str):
        self.config = TrainingConfig(base_dir)
        self.formatter = DataFormatter(self.config)
        self.builder = StudentModelBuilder(self.config)

    def execute(self) -> None:
        dataset = self.formatter.prepare_dataset()
        tokenizer = self.builder.initialize_tokenizer()
        model = self.builder.initialize_model()

        # Get the strict, true vocabulary size directly from the active model layer
        true_vocab_size = model.get_input_embeddings().weight.shape[0]
        logger.info(f"Tokenizing dataset... (Deploying SafeDataCollator for vocab size {true_vocab_size})")
        
        def tokenize_function(examples):
            texts = [tokenizer.apply_chat_template(msg, tokenize=False) for msg in examples["messages"]]
            return tokenizer(
                texts,
                truncation=True,
                max_length=self.config.max_seq_length,
                padding=False # Padding is safely deferred to our custom collator
            )

        processed_dataset = dataset.map(
            tokenize_function, 
            batched=True, 
            remove_columns=["messages"]
        )
        
        training_args = TrainingArguments(
            output_dir=str(self.config.output_dir),
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            logging_steps=self.config.logging_steps,
            max_steps=-1,
            num_train_epochs=self.config.num_train_epochs,
            save_steps=self.config.save_steps,
            fp16=False,
            bf16=True,
            optim="paged_adamw_32bit",
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type="cosine",
            ddp_find_unused_parameters=False,
            report_to="none",
            save_total_limit=2
        )

        # Deploy our custom collision-proof collator
        collator = SafeDataCollator(tokenizer=tokenizer, vocab_size=true_vocab_size)

        logger.info("Initializing Standard Trainer with Safe Collator")
        trainer = Trainer(
            model=model,
            train_dataset=processed_dataset,
            args=training_args,
            data_collator=collator,
            callbacks=[PerformanceMonitor()]
        )

        # --- DYNAMIC CHECKPOINT RESUMPTION (UPGRADED) ---
        last_checkpoint = get_last_checkpoint(str(self.config.output_dir))
        if last_checkpoint is not None:
            logger.info(f"Existing checkpoint found: {last_checkpoint}. Resuming training...")
        else:
            logger.info("No valid checkpoints found. Starting from scratch...")

        logger.info("Beginning Student Model Distillation Training")
        trainer.train(resume_from_checkpoint=last_checkpoint)

        logger.info(f"Training Complete. Saving LoRA adapters to {self.config.output_dir}")
        trainer.model.save_pretrained(self.config.output_dir)
        tokenizer.save_pretrained(self.config.output_dir)
        
        logger.info("Phase 3 Execution Completed Successfully")

if __name__ == "__main__":
    trainer_engine = OptiTrainer(os.getcwd())
    trainer_engine.execute()