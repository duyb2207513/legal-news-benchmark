"""Embed câu hỏi (query) qua Vertex AI — dùng task_type=RETRIEVAL_QUERY.

Tách riêng khỏi embed/gemini_embedding.py vì đó là batch-embed TextUnit lúc
build index (task_type mặc định RETRIEVAL_DOCUMENT), còn đây là embed 1 câu
hỏi tại thời điểm truy vấn — 2 việc khác nhau dù cùng dùng chung model.
"""
from __future__ import annotations

import logging

from config import EMBEDDING_MODEL, GCP_LOCATION, GCP_PROJECT

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
    return _client


def embed_question(question: str, task_type: str = "RETRIEVAL_QUERY") -> list[float]:
    """Trả về vector embedding cho 1 câu hỏi.

    Ném exception nếu GCP_PROJECT chưa cấu hình hoặc lỗi gọi API — để
    caller (retrieval/pipeline.py) tự quyết định fallback (vd bỏ qua
    nhánh vector, chỉ dùng BM25).
    """
    if not GCP_PROJECT:
        raise RuntimeError("GCP_PROJECT chưa được cấu hình trong .env")
    client = _get_client()
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[question],
        config={"task_type": task_type},
    )
    return list(result.embeddings[0].values)