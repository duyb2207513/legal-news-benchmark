# """Trích từ khoá PHÁP LÝ từ câu hỏi (tên luật, loại văn bản, khái niệm) —
# dùng để làm query cho bm25_search() khi USE_KEYWORD_EXTRACTION=True, vì
# BM25 match theo từ khoá chính xác nên câu hỏi tự nhiên đầy đủ (có cả tình
# huống cá nhân: số tiền, tên công ty...) thường match kém hơn là chỉ trích
# riêng phần từ khoá pháp lý.
# """
from __future__ import annotations

from langchain_google_vertexai import ChatVertexAI

from config import GCP_LOCATION, GCP_PROJECT, LLM_MODEL_HEAVY

_llm = None


def _get_llm() -> ChatVertexAI:
    global _llm
    if _llm is None:
        _llm = ChatVertexAI(
            model=LLM_MODEL_HEAVY,
            project=GCP_PROJECT,
            location=GCP_LOCATION,
            temperature=0,
        )
    return _llm



import traceback
def extract_legal_keywords(question: str, llm=None) -> str:
    llm = llm or _get_llm()
    prompt = f"""Trích từ khóa PHÁP LÝ (tên luật, loại văn bản, khái niệm pháp lý cụ thể —
      không lấy từ khóa mô tả tình huống cá nhân như số tiền, tên công ty).
      Chỉ trả từ khóa, cách nhau bởi dấu cách.

      Câu hỏi: {question} 
      """
    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else response

        # Xử lý trường hợp content là list các block (thay vì string đơn)
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and "text" in block:
                    parts.append(block["text"])
            content = " ".join(parts)

        return content.strip()
    except Exception as e:
        print("=== LỖI THẬT SỰ KHI GỌI LLM ===")
        print(f"Loại lỗi: {type(e).__name__}")
        print(f"Nội dung: {e}")
        import traceback
        traceback.print_exc()
        print("================================")
        return question