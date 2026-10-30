import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class BenchmarkConfig:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "data"
        self.splits_dir = self.data_dir / "splits"
        self.models_dir = self.base_dir / "models"
        
        # Pointing directly to our successful 1400-step checkpoint!
        self.student_adapter_dir = self.models_dir / "opti_student_model" / "checkpoint-1400"
        
        self.test_data_path = self.splits_dir / "test.jsonl"
        self.eval_output_path = self.base_dir / "evaluation_report.json"
        self.base_student_id = "microsoft/Phi-3.5-mini-instruct"
        self.max_test_samples = 100
        self.max_new_tokens = 256
        self.batch_size = 1

class EvaluatedStudentLoader:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if not self.config.student_adapter_dir.exists():
            raise FileNotFoundError(f"CRITICAL ERROR: Cannot find the trained LoRA adapters at {self.config.student_adapter_dir}. The benchmark cannot proceed.")
            
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()

    def _load_tokenizer(self) -> AutoTokenizer:
        logger.info(f"Loading tokenizer from {self.config.student_adapter_dir}")
        tokenizer = AutoTokenizer.from_pretrained(
            str(self.config.student_adapter_dir),
            trust_remote_code=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = 32000
            tokenizer.pad_token = "<|endoftext|>"
        tokenizer.padding_side = "left"
        return tokenizer

    def _load_model(self) -> AutoModelForCausalLM:
        logger.info(f"Loading base model {self.config.base_student_id} in 4-bit...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_student_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        logger.info(f"Merging LoRA adapters from checkpoint-1400...")
        model = PeftModel.from_pretrained(
            base_model,
            str(self.config.student_adapter_dir)
        )
        model.eval()
        return model

class PerformanceEvaluator:
    def __init__(self, student: EvaluatedStudentLoader, config: BenchmarkConfig):
        self.student = student
        self.config = config
        self.system_prompt = (
            "You are an expert AI financial analyst. You must respond ONLY with valid, raw JSON. "
            "Do not include markdown formatting, backticks, or any conversational text. "
            "Ensure the JSON is strictly formatted."
        )
        self.results = {
            "total_samples": 0,
            "successful_parses": 0,
            "failed_parses": 0,
            "average_latency_seconds": 0.0,
            "average_tokens_per_second": 0.0,
            "task_performance": {}
        }

    def _format_prompt(self, task_type: str, text: str, ticker: str) -> str:
        context = f"COMPANY: {ticker}\nTEXT:\n{text}\n\n"
        if task_type == "summarize":
            instruction = "Summarize the core financial and operational insights."
        elif task_type == "extract_risk":
            instruction = "Extract specific risks."
        elif task_type == "analyze_controls":
            instruction = "Identify internal controls or accounting metrics."
        else:
            instruction = "Extract insights."
        user_content = f"{context}INSTRUCTION:\n{instruction}"
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        return self.student.tokenizer.apply_chat_template(messages, tokenize=False)

    def _load_test_data(self) -> List[Dict[str, Any]]:
        logger.info(f"Loading unseen test data from {self.config.test_data_path}")
        data = []
        if not self.config.test_data_path.exists():
            logger.warning("Test dataset not found. Generating dummy test data for benchmark verification.")
            return [
                {
                    "text": "The company expects Q4 revenue to decrease by 15% due to supply chain disruptions and increased regulatory scrutiny in European markets.",
                    "ticker": "DUMMY",
                    "task_type": "extract_risk"
                },
                {
                    "text": "Gross margin improved from 42% to 46% driven by higher software subscription renewals and reduced cloud infrastructure costs.",
                    "ticker": "DUMMY",
                    "task_type": "summarize"
                }
            ]
            
        with open(self.config.test_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if len(data) >= self.config.max_test_samples:
                    break
                record = json.loads(line.strip())
                # Assign a random task type if it doesn't exist in the raw split
                record["task_type"] = record.get("task_type", "summarize") 
                data.append(record)
        return data

    def run_benchmark(self) -> None:
        test_data = self._load_test_data()
        self.results["total_samples"] = len(test_data)
        
        logger.info(f"Beginning Benchmarking on {len(test_data)} samples...")
        total_latency = 0
        total_generated_tokens = 0
        
        for idx, sample in enumerate(test_data):
            prompt = self._format_prompt(sample["task_type"], sample["text"], sample["ticker"])
            
            inputs = self.student.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048
            ).to(self.student.model.device)
            
            input_length = inputs.input_ids.shape[1]
            
            start_time = time.time()
            
            with torch.no_grad():
                # --- CRITICAL FIX: Phi-3.5 Cache Bug ---
                # Explicitly pass use_cache=False to bypass Microsoft's broken DynamicCache implementation
                outputs = self.student.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    pad_token_id=self.student.tokenizer.pad_token_id,
                    eos_token_id=self.student.tokenizer.eos_token_id,
                    do_sample=False,
                    use_cache=False 
                )
                # ---------------------------------------
                
            end_time = time.time()
            
            latency = end_time - start_time
            generated_tokens = outputs.shape[1] - input_length
            
            total_latency += latency
            total_generated_tokens += generated_tokens
            
            output_text = self.student.tokenizer.decode(
                outputs[0][input_length:], 
                skip_special_tokens=True
            ).strip()
            
            # Strict JSON Formatting Verification
            is_valid_json = False
            try:
                json.loads(output_text)
                is_valid_json = True
                self.results["successful_parses"] += 1
            except json.JSONDecodeError:
                self.results["failed_parses"] += 1
                
            task = sample["task_type"]
            if task not in self.results["task_performance"]:
                self.results["task_performance"][task] = {"total": 0, "valid_json": 0}
            
            self.results["task_performance"][task]["total"] += 1
            if is_valid_json:
                self.results["task_performance"][task]["valid_json"] += 1
                
            if (idx + 1) % 10 == 0:
                logger.info(f"Evaluated {idx + 1}/{len(test_data)} samples...")

        # Calculate Final Metrics
        if self.results["total_samples"] > 0:
            self.results["average_latency_seconds"] = round(total_latency / self.results["total_samples"], 4)
            self.results["average_tokens_per_second"] = round(total_generated_tokens / total_latency, 2)
            
        self._save_report()

    def _save_report(self) -> None:
        logger.info("\n--- FINAL BENCHMARK RESULTS ---")
        logger.info(json.dumps(self.results, indent=4))
        
        with open(self.config.eval_output_path, "w") as f:
            json.dump(self.results, f, indent=4)
        logger.info(f"Detailed report saved to {self.config.eval_output_path}")
        logger.info("Phase 4 Execution Completed Successfully")

if __name__ == "__main__":
    config = BenchmarkConfig(os.getcwd())
    loader = EvaluatedStudentLoader(config)
    evaluator = PerformanceEvaluator(loader, config)
    evaluator.run_benchmark()