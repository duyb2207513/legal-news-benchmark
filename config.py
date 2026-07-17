"""Đọc .env, hằng số dùng chung, bảng rank cấp văn bản (Component level)."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Neo4j AuraDB ---
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# --- Google Cloud / Vertex AI ---
GCP_PROJECT = os.getenv("GCP_PROJECT", "")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
LLM_MODEL_HEAVY = os.getenv("LLM_MODEL_HEAVY", "gemini-3.5-flash")
LLM_MODEL_LIGHT = os.getenv("LLM_MODEL_LIGHT", "gemini-3.1-flash-lite")

# --- Hugging Face dataset ---
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "th1nhng0/vietnamese-legal-documents")

# --- Đường dẫn dữ liệu trung gian ---
DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()
RAW_DIR = DATA_DIR / "raw"
TRANSFORMED_DIR = DATA_DIR / "transformed"
EMBEDDED_DIR = DATA_DIR / "embedded"

BM25_DIR = DATA_DIR / "bm25_index"

for _dir in (RAW_DIR, TRANSFORMED_DIR, EMBEDDED_DIR, BM25_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# --- Retrieval (retrieval/pipeline.py, benchmark/) ---
# TOP_K tăng 5 -> 15: cần đủ ứng viên thô cho reranker chọn lọc, trước đây
# top_k=5 mỗi nhánh (vector/BM25) quá hẹp nên rerank gần như không có gì để
# lọc — đây là thay đổi so với baseline dùng để benchmark 4 mode cũ, nên
# nếu cần so sánh lại với baseline thì set TOP_K=5 qua .env.
TOP_K = int(os.getenv("TOP_K", "15"))
MAX_COMPONENTS = int(os.getenv("MAX_COMPONENTS", "5"))
MAX_CHARS_PER_UNIT = int(os.getenv("MAX_CHARS_PER_UNIT", "1000"))
MAX_PROMPT_TOKENS = int(os.getenv("MAX_PROMPT_TOKENS", "6000"))
BM25_BACKEND = os.getenv("BM25_BACKEND", "whoosh")

# --- Rerank (retrieval/reranker.py) ---
# Bật/tắt rerank sau RRF — tắt để so sánh A/B với pipeline cũ khi benchmark.
USE_RERANK = os.getenv("USE_RERANK", "true").lower() == "true"
RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "0.0"))

# --- Bảng rank cấp văn bản (Component) — số nhỏ hơn = cấp cao hơn (nông hơn trong cây) ---
LEVEL_RANK = {
    "Phan": 0,
    "Chuong": 1,
    "Muc": 2,
    "TieuMuc": 3,
    "Dieu": 4,
    "Khoan": 5,
    "Diem": 6,
}

# --- Batch size dùng cho ghi Neo4j (UNWIND + MERGE) ---
NEO4J_BATCH_SIZE = 500