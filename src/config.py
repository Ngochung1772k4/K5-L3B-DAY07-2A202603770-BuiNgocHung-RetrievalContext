"""Shared paths and runtime configuration for Day 7."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
STARTER_DIR = ROOT / "starter"

DAY6_REPO_DIR = Path(
    os.getenv(
        "DAY6_REPO_DIR",
        r"D:\AIInAction\Day6\K5-L3B-DAY06-BuiNgocHung-2A202603770-DataIngestion",
    )
)
DAY6_CHROMA_DIR = Path(
    os.getenv("DAY6_CHROMA_DIR", str(DAY6_REPO_DIR / "starter" / "results" / "chroma"))
)
LOCAL_CHROMA_DIR = ROOT / "results" / "chroma_cache"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
DAY6_COLLECTION = "day6_b"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
AGENT_MODEL = os.getenv("AGENT_MODEL", "qwen2.5:7b")

CHUNKS_PATH = DATA_DIR / "day6_chunks.json"
QUESTIONS_PATH = DATA_DIR / "day6_questions.json"
AGENT_CASES_PATH = DATA_DIR / "agent_cases.json"
