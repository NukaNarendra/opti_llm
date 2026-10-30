import os
import re
import json
import time
import random
import logging
import requests
import threading
import warnings
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class PipelineConfig:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "data"
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.splits_dir = self.data_dir / "splits"
        self.user_agent = "OptiLLM_Research_System/3.0 (admin@optillm.local)"
        self.max_workers = 8
        self.chunk_size = 512
        self.overlap = 64
        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.test_ratio = 0.1
        self.target_tickers = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", 
            "TSLA", "NVDA", "JPM", "V", "WMT", 
            "BAC", "MA", "HD", "PG", "JNJ", 
            "UNH", "XOM", "INTC", "CSCO", "NFLX"
        ]
        self.target_forms = ["10-K", "10-Q"]
        self.min_paragraph_words = 20
        self.max_docs_per_ticker = 10
        self._create_directories()

    def _create_directories(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.splits_dir.mkdir(parents=True, exist_ok=True)

class SECRateLimiter:
    def __init__(self, max_requests_per_second: int = 8):
        self.interval = 1.0 / max_requests_per_second
        self.lock = threading.Lock()
        self.last_request_time = 0.0

    def wait(self):
        with self.lock:
            current_time = time.time()
            elapsed = current_time - self.last_request_time
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self.last_request_time = time.time()

class EdgarFetcher:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.headers = {"User-Agent": self.config.user_agent, "Accept-Encoding": "gzip, deflate"}
        self.rate_limiter = SECRateLimiter()
        self.ticker_to_cik_url = "https://www.sec.gov/files/company_tickers.json"
        self.submissions_url = "https://data.sec.gov/submissions/CIK{cik}.json"
        self.document_url = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
        self.cik_mapping = self._fetch_cik_mapping()

    def _make_request(self, url: str) -> requests.Response:
        max_retries = 3
        for attempt in range(max_retries):
            self.rate_limiter.wait()
            try:
                response = requests.get(url, headers=self.headers, timeout=15)
                if response.status_code == 429:
                    time.sleep((attempt + 1) * 3)
                    continue
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(2)
        raise requests.exceptions.RequestException("Max retries exceeded")

    def _fetch_cik_mapping(self) -> Dict[str, str]:
        response = self._make_request(self.ticker_to_cik_url)
        data = response.json()
        mapping = {}
        for entry in data.values():
            mapping[entry["ticker"]] = str(entry["cik_str"]).zfill(10)
        return mapping

    def fetch_submissions(self, cik: str) -> Dict[str, Any]:
        url = self.submissions_url.format(cik=cik)
        response = self._make_request(url)
        return response.json()

    def download_filing(self, ticker: str, cik: str, accession: str, document: str) -> str:
        accession_no_dashes = accession.replace("-", "")
        url = self.document_url.format(cik=cik.lstrip("0"), accession=accession_no_dashes, document=document)
        file_path = self.config.raw_dir / f"{ticker}_{accession}.htm"
        if file_path.exists():
            return str(file_path)
        response = self._make_request(url)
        with open(file_path, "wb") as f:
            f.write(response.content)
        return str(file_path)

    def process_ticker(self, ticker: str) -> List[str]:
        downloaded_files = []
        if ticker not in self.cik_mapping:
            return downloaded_files
        cik = self.cik_mapping[ticker]
        try:
            subs = self.fetch_submissions(cik)
            filings = subs.get("filings", {}).get("recent", {})
            if not filings:
                return downloaded_files
            accessions = filings.get("accessionNumber", [])
            forms = filings.get("form", [])
            documents = filings.get("primaryDocument", [])
            docs_downloaded = 0
            for idx, form in enumerate(forms):
                if form in self.config.target_forms:
                    accession = accessions[idx]
                    document = documents[idx]
                    if not document or not document.endswith((".htm", ".html", ".xml")):
                        continue
                    try:
                        filepath = self.download_filing(ticker, cik, accession, document)
                        downloaded_files.append(filepath)
                        docs_downloaded += 1
                        if docs_downloaded >= self.config.max_docs_per_ticker:
                            break
                    except Exception as e:
                        logger.error(f"Failed filing {ticker} {accession}: {str(e)}")
        except Exception as e:
            logger.error(f"Failed submission {ticker}: {str(e)}")
        return downloaded_files

class IXBRLCleaner:
    def __init__(self):
        self.hidden_pattern = re.compile(r"<ix:hidden>.*?</ix:hidden>", re.IGNORECASE | re.DOTALL)
        self.numeric_ratio_threshold = 0.35

    def is_mostly_numeric(self, text: str) -> bool:
        tokens = text.split()
        if not tokens:
            return True
        numeric_count = sum(1 for t in tokens if any(char.isdigit() for char in t))
        return (numeric_count / len(tokens)) > self.numeric_ratio_threshold

    def clean_document(self, html_content: str) -> str:
        content = self.hidden_pattern.sub("", html_content)
        soup = BeautifulSoup(content, features="xml")
        for tag in soup(["script", "style", "meta", "link", "head", "noscript", "ix:header"]):
            tag.decompose()
        for table in soup.find_all("table"):
            table_text = table.get_text(separator=" ", strip=True)
            if self.is_mostly_numeric(table_text):
                table.decompose()
        paragraphs = []
        for element in soup.find_all(["p", "div", "span", "ix:nonnumeric", "text"]):
            text = element.get_text(separator=" ", strip=True)
            text = re.sub(r"\s+", " ", text)
            if len(text.split()) > 10 and not self.is_mostly_numeric(text):
                paragraphs.append(text)
        return "\n".join(paragraphs)

class FinancialFilingParser:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.cleaner = IXBRLCleaner()
        self.item_pattern = re.compile(r"(?i)^(?:item|part)\s*([1-9][a-z]?)\b")
        self.section_mapping = {
            "1": "business",
            "1a": "risk_factors",
            "2": "properties",
            "3": "legal_proceedings",
            "7": "mda",
            "7a": "market_risk",
            "8": "financial_statements",
            "9": "controls_procedures"
        }

    def extract_sections(self, raw_html: str) -> Dict[str, str]:
        clean_text = self.cleaner.clean_document(raw_html)
        sections = {}
        current_section = "general_narrative"
        sections[current_section] = []
        lines = clean_text.split("\n")
        for line in lines:
            line_clean = line.strip().replace("&#160;", " ").replace("&nbsp;", " ")
            if len(line_clean) < 60:
                match = self.item_pattern.search(line_clean)
                if match:
                    item_num = match.group(1).lower()
                    if item_num in self.section_mapping:
                        current_section = self.section_mapping[item_num]
                        if current_section not in sections:
                            sections[current_section] = []
                        continue
            words = line_clean.split()
            if len(words) >= self.config.min_paragraph_words:
                sections[current_section].append(line_clean)
        consolidated = {}
        for key, text_list in sections.items():
            combined_text = " ".join(text_list)
            if len(combined_text.split()) > 100:
                consolidated[key] = combined_text
        return consolidated

class TextChunker:
    def __init__(self, chunk_size: int, overlap: int):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        if len(words) <= self.chunk_size:
            if len(words) > 50:
                return [" ".join(words)]
            return []
        step = self.chunk_size - self.overlap
        i = 0
        while i < len(words):
            chunk = words[i:i + self.chunk_size]
            if len(chunk) > 50:
                chunks.append(" ".join(chunk))
            i += step
        return chunks

class DatasetManager:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.parser = FinancialFilingParser(config)
        self.chunker = TextChunker(config.chunk_size, config.overlap)
        self.dataset = []
        self.stats = defaultdict(int)
        self.section_stats = defaultdict(int)
        self.lock = threading.Lock()

    def process_raw_file(self, file_path: str) -> None:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            sections = self.parser.extract_sections(content)
            filename = Path(file_path).stem
            ticker = filename.split("_")[0]
            local_records = []
            for section_name, section_text in sections.items():
                chunks = self.chunker.chunk_text(section_text)
                for idx, chunk in enumerate(chunks):
                    record = {
                        "id": f"{filename}_{section_name}_{idx}",
                        "ticker": ticker,
                        "section": section_name,
                        "text": chunk,
                        "word_count": len(chunk.split())
                    }
                    local_records.append(record)
            with self.lock:
                self.dataset.extend(local_records)
                self.stats[ticker] += len(local_records)
                for r in local_records:
                    self.section_stats[r["section"]] += 1
        except Exception as e:
            logger.error(f"Processing failed {file_path}: {str(e)}")

    def build_splits(self) -> None:
        if not self.dataset:
            logger.warning("Dataset is completely empty after processing.")
            return
        random.shuffle(self.dataset)
        total = len(self.dataset)
        train_end = int(total * self.config.train_ratio)
        val_end = train_end + int(total * self.config.val_ratio)
        train_data = self.dataset[:train_end]
        val_data = self.dataset[train_end:val_end]
        test_data = self.dataset[val_end:]
        self._write_jsonl(train_data, self.config.splits_dir / "train.jsonl")
        self._write_jsonl(val_data, self.config.splits_dir / "validation.jsonl")
        self._write_jsonl(test_data, self.config.splits_dir / "test.jsonl")
        self._write_stats()

    def _write_jsonl(self, data: List[Dict[str, Any]], output_path: Path) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    def _write_stats(self) -> None:
        stats_data = {
            "total_samples": len(self.dataset),
            "ticker_distribution": dict(self.stats),
            "section_distribution": dict(self.section_stats)
        }
        with open(self.config.data_dir / "dataset_stats.json", "w") as f:
            json.dump(stats_data, f, indent=4)
        logger.info(f"Dataset compiled. Total samples generated: {len(self.dataset)}")
        logger.info(f"Section breakdown: {dict(self.section_stats)}")

def execute_pipeline(base_dir: str):
    config = PipelineConfig(base_dir)
    fetcher = EdgarFetcher(config)
    manager = DatasetManager(config)
    all_raw_files = []
    logger.info("Initializing iXBRL Data Download")
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        future_to_ticker = {executor.submit(fetcher.process_ticker, ticker): ticker for ticker in config.target_tickers}
        for future in as_completed(future_to_ticker):
            try:
                files = future.result()
                all_raw_files.extend(files)
            except Exception:
                pass
    fallback_files = list(config.raw_dir.glob("*.htm"))
    files_to_process = list(set(all_raw_files + [str(p) for p in fallback_files]))
    logger.info(f"Initializing iXBRL Parsing for {len(files_to_process)} files")
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        for file_path in files_to_process:
            executor.submit(manager.process_raw_file, file_path)
    logger.info("Initializing Dataset Splitting")
    manager.build_splits()
    logger.info("Phase 1 Execution Completed")

if __name__ == "__main__":
    execute_pipeline(os.getcwd())