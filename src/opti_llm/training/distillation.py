import os
import json
import time
import logging
import gc
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class DistillationConfig:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "data"
        self.splits_dir = self.data_dir / "splits"
        self.processed_dir = self.data_dir / "processed"
        self.input_file = self.splits_dir / "train.jsonl"
        self.output_file = self.processed_dir / "distilled_train.jsonl"
        self.checkpoint_file = self.processed_dir / "distillation_checkpoint.json"
        self.teacher_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.initial_batch_size = 8
        self.max_input_length = 1536
        self.max_new_tokens = 512
        self.temperature = 0.2
        self.top_p = 0.95
        self.top_k = 50
        self.save_interval = 50
        self.tasks = ["summarize", "extract_risk", "analyze_controls", "generate_hallucination"]
        self.quarantined_indices = set(list(range(11600, 11620)) + list(range(17700, 17720)))
        self._prepare_directories()

    def _prepare_directories(self) -> None:
        self.processed_dir.mkdir(parents=True, exist_ok=True)

class PromptLibrary:
    def __init__(self):
        self.system_prompt = (
            "You are an expert AI financial analyst. You must respond ONLY with valid, raw JSON. "
            "Do not include markdown formatting, backticks, or any conversational text. "
            "Ensure the JSON is strictly formatted."
        )

    def get_prompt(self, task_type: str, text: str, ticker: str, section: str) -> str:
        context = f"COMPANY: {ticker}\nSECTION: {section}\nTEXT:\n{text}\n\n"
        
        if task_type == "summarize":
            instruction = (
                "Summarize the core financial and operational insights. "
                "Output JSON format: {\"task\": \"summary\", \"content\": \"<summary text>\"}"
            )
        elif task_type == "extract_risk":
            instruction = (
                "Extract specific risks. "
                "Output JSON format: {\"task\": \"risks\", \"items\": [\"<risk 1>\", \"<risk 2>\"]}"
            )
        elif task_type == "analyze_controls":
            instruction = (
                "Identify internal controls or accounting metrics. "
                "Output JSON format: {\"task\": \"controls\", \"metrics\": [\"<metric 1>\"]}"
            )
        elif task_type == "generate_hallucination":
            instruction = (
                "Generate a highly plausible but completely fake financial statistic or risk based on the text context. "
                "Output JSON format: {\"task\": \"hallucination\", \"fake_insight\": \"<fake text>\", \"reasoning\": \"<why it looks real>\"}"
            )
        else:
            instruction = (
                "Extract insights. "
                "Output JSON format: {\"task\": \"general\", \"content\": \"<insight>\"}"
            )
            
        return f"{self.system_prompt}\n\n{context}INSTRUCTION:\n{instruction}"

class SECDataset(Dataset):
    def __init__(self, data_path: Path):
        self.data = []
        self._load_data(data_path)

    def _load_data(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found at {path}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    try:
                        self.data.append(json.loads(stripped))
                    except json.JSONDecodeError:
                        continue

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]

class DataIO:
    def __init__(self, config: DistillationConfig):
        self.config = config

    def load_checkpoint(self) -> int:
        if self.config.checkpoint_file.exists():
            try:
                with open(self.config.checkpoint_file, "r") as f:
                    data = json.load(f)
                    return data.get("last_processed_index", 0)
            except Exception:
                return 0
        return 0

    def save_checkpoint(self, index: int) -> None:
        tmp_file = self.config.checkpoint_file.with_suffix('.tmp')
        with open(tmp_file, "w") as f:
            json.dump({"last_processed_index": index, "timestamp": time.time()}, f)
        tmp_file.replace(self.config.checkpoint_file)

    def append_results(self, results: List[Dict[str, Any]]) -> None:
        with open(self.config.output_file, "a", encoding="utf-8") as f:
            for res in results:
                f.write(json.dumps(res) + "\n")

class OutputSanitizer:
    @staticmethod
    def clean_json_string(raw_output: str) -> Optional[Dict[str, Any]]:
        cleaned = raw_output.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            try:
                start_idx = cleaned.find('{')
                end_idx = cleaned.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    return json.loads(cleaned[start_idx:end_idx])
            except Exception:
                pass
            return {"raw_fallback": cleaned, "parse_error": True}

class TeacherLLM:
    def __init__(self, config: DistillationConfig):
        self.config = config
        self.tokenizer = self._initialize_tokenizer()
        self.model = self._initialize_model()

    def _initialize_tokenizer(self) -> AutoTokenizer:
        logger.info(f"Loading tokenizer: {self.config.teacher_model_id}")
        tokenizer = AutoTokenizer.from_pretrained(self.config.teacher_model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return tokenizer

    def _initialize_model(self) -> AutoModelForCausalLM:
        logger.info("Initializing Teacher Model with 4-bit quantization and bfloat16")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.config.teacher_model_id,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        model.eval()
        return model

    def format_chat(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate_batch(self, prompts: List[str]) -> Tuple[List[str], int]:
        chat_prompts = [self.format_chat(p) for p in prompts]
        inputs = self.tokenizer(
            chat_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_input_length
        ).to(self.model.device)
        
        input_token_count = inputs.input_ids.numel()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
        generated_texts = []
        total_new_tokens = 0
        
        for i in range(len(prompts)):
            input_length = inputs.input_ids[i].shape[0]
            generated_tokens = outputs[i][input_length:]
            total_new_tokens += len(generated_tokens)
            decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            generated_texts.append(decoded.strip())
            
        return generated_texts, total_new_tokens

class DistillationEngine:
    def __init__(self, base_dir: str):
        self.config = DistillationConfig(base_dir)
        self.io = DataIO(self.config)
        self.prompter = PromptLibrary()
        self.sanitizer = OutputSanitizer()
        self.dataset = SECDataset(self.config.input_file)
        self.current_batch_size = self.config.initial_batch_size
        
    def _create_batches(self, start_idx: int) -> List[Tuple[int, List[Dict[str, Any]]]]:
        batches = []
        current_batch = []
        for i in range(start_idx, len(self.dataset)):
            if i in self.config.quarantined_indices:
                continue
            current_batch.append((i, self.dataset[i]))
            if len(current_batch) == self.current_batch_size:
                batches.append(current_batch)
                current_batch = []
        if current_batch:
            batches.append(current_batch)
        return batches

    def _handle_oom(self) -> None:
        self.current_batch_size = max(1, self.current_batch_size // 2)
        logger.warning(f"OOM Error encountered. Reducing batch size to {self.current_batch_size}")
        gc.collect()
        torch.cuda.empty_cache()

    def run(self) -> None:
        start_index = self.io.load_checkpoint()
        total_samples = len(self.dataset)
        
        if start_index in self.config.quarantined_indices:
            start_index = max(self.config.quarantined_indices) + 1
            
        if start_index >= total_samples:
            logger.info("Distillation already completed.")
            return

        logger.info(f"Starting distillation at index {start_index}/{total_samples}")
        
        try:
            self.llm = TeacherLLM(self.config)
        except Exception as e:
            logger.error(f"Teacher Model initialization failed: {str(e)}")
            return

        while start_index < total_samples:
            batches = self._create_batches(start_index)
            if not batches:
                break
                
            batch_data = batches[0]
            prompts = []
            metadata = []
            
            for idx, item in batch_data:
                task = self.config.tasks[idx % len(self.config.tasks)]
                prompt = self.prompter.get_prompt(task, item["text"], item["ticker"], item["section"])
                prompts.append(prompt)
                metadata.append({
                    "id": item["id"], 
                    "ticker": item["ticker"],
                    "task": task, 
                    "original_text": item["text"]
                })
                
            try:
                t0 = time.time()
                responses, new_tokens = self.llm.generate_batch(prompts)
                t1 = time.time()
                
                tps = new_tokens / (t1 - t0)
                
                current_results = []
                for meta, response in zip(metadata, responses):
                    parsed_response = self.sanitizer.clean_json_string(response)
                    result_record = {
                        "id": meta["id"],
                        "ticker": meta["ticker"],
                        "task_type": meta["task"],
                        "input_text": meta["original_text"],
                        "teacher_response": parsed_response
                    }
                    current_results.append(result_record)
                    
                self.io.append_results(current_results)
                
                start_index = batch_data[-1][0] + 1
                
                if start_index % self.config.save_interval == 0 or start_index >= total_samples:
                    self.io.save_checkpoint(start_index)
                    logger.info(f"Saved: {start_index}/{total_samples} | Speed: {tps:.2f} tok/s | Batch: {self.current_batch_size}")
                    
            except torch.cuda.OutOfMemoryError:
                self._handle_oom()
                if self.current_batch_size == 1:
                    logger.error("OOM even at batch size 1. Skipping this specific sample.")
                    start_index += 1
            except Exception as e:
                logger.error(f"Critical error during generation: {str(e)}")
                break

        self.io.save_checkpoint(start_index)
        logger.info("Phase 2 Execution Completed")

if __name__ == "__main__":
    engine = DistillationEngine(os.getcwd())
    engine.run()