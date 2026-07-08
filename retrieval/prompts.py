"""Prompt template + hàm sinh câu trả lời cuối cùng (RAG) từ context đã build.

Model dùng LLM_MODEL_HEAVY trong config.py (mặc định gemini-3.5-flash) thay
vì hardcode như notebook gốc — đổi model chỉ cần sửa .env, không sửa code.
"""
from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_vertexai import ChatVertexAI

from config import GCP_LOCATION, GCP_PROJECT, LLM_MODEL_HEAVY

RAG_PROMPT = PromptTemplate(
    input_variables=["question", "context"],
    template="""Bạn là chuyên gia pháp luật Việt Nam. Dựa vào context dưới đây, trả lời câu hỏi.

Context:
{context}

Câu hỏi: {question}

Yêu cầu: nêu rõ phạm vi áp dụng nếu context chỉ phản ánh 1 trường hợp/thời kỳ cụ thể; ưu tiên văn bản/nội dung còn hiệu lực (nêu rõ nếu đã bị sửa/hủy/đình chỉ); trích số hiệu văn bản + citation cụ thể; nếu context chưa đủ thì nói rõ.""",
)

_llm_answer = None


def get_llm() -> ChatVertexAI:
    global _llm_answer
    if _llm_answer is None:
        _llm_answer = ChatVertexAI(
            model=LLM_MODEL_HEAVY,
            project=GCP_PROJECT,
            location=GCP_LOCATION,
            temperature=0,
        )
    return _llm_answer


def answer(question: str, context: str) -> str:
    """Sinh câu trả lời cuối cùng từ context đã build (flat hoặc graph)."""
    chain = RAG_PROMPT | get_llm() | StrOutputParser()
    return chain.invoke({"question": question, "context": context})


def count_tokens(text: str, model: str = LLM_MODEL_HEAVY, use_api: bool = False) -> int:
    """Đếm token cho prompt.

    Mặc định dùng ước lượng local (len // 3) — KHÔNG gọi API, vì hàm này
    chạy trên mỗi request chỉ để quyết định có cắt context hay không, gọi
    API Vertex chỉ để đếm token là 1 round-trip network lãng phí.

    use_api=True: dùng khi cần số liệu chính xác (vd benchmark/report cuối
    cùng) — gọi thật client.models.count_tokens, fallback về ước lượng nếu lỗi.
    """
    if not use_api:
        return len(text) // 3
    try:
        from google import genai
        client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
        result = client.models.count_tokens(model=model, contents=text)
        return result.total_tokens
    except Exception as e:
        print(f"Không đếm được token, dùng ước lượng: {e}")
        return len(text) // 3