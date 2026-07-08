"""Metric cần ground truth (gold_citations/gold_answer trong EVAL_SET):

- recall_at_k / mrr / ndcg_at_k: chấm chất lượng RETRIEVAL (so citation lấy
  được với gold_citations) — trả None nếu câu đó chưa có gold_citations.
- llm_judge: chấm chất lượng CÂU TRẢ LỜI cuối (so với gold_answer) qua LLM,
  thang 1-5 cho correctness + faithfulness — bỏ qua nếu chưa có gold_answer.

Khác nhóm metric ở benchmark/deepeval_judge.py (không cần ground truth, chấm
trực tiếp trên context+answer) — 2 nhóm bổ sung cho nhau, không thay thế.
"""
from __future__ import annotations

import json
import math

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from retrieval.prompts import get_llm


def _hit(retrieved_citations: list, gold_citations: list) -> list[int]:
    gold_set = set(gold_citations)
    return [1 if c in gold_set else 0 for c in retrieved_citations]


def recall_at_k(retrieved_citations: list, gold_citations: list) -> float | None:
    if not gold_citations:
        return None
    hits = set(retrieved_citations) & set(gold_citations)
    return len(hits) / len(gold_citations)


def mrr(retrieved_citations: list, gold_citations: list) -> float:
    for i, c in enumerate(retrieved_citations, start=1):
        if c in gold_citations:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_citations: list, gold_citations: list) -> float:
    gold_set = set(gold_citations)
    dcg = sum((1 if c in gold_set else 0) / math.log2(i + 1) for i, c in enumerate(retrieved_citations, start=1))
    ideal_hits = min(len(gold_set), len(retrieved_citations))
    idcg = sum(1 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


JUDGE_PROMPT = PromptTemplate(
    input_variables=["question", "gold_answer", "model_answer"],
    template="""Bạn là giám khảo pháp lý. So sánh câu trả lời của mô hình với đáp án chuẩn.
Chấm điểm 1-5 cho 2 tiêu chí:
- correctness: nội dung pháp lý có đúng và đủ không
- faithfulness: có bịa thông tin ngoài context/pháp luật không (5 = không bịa)

Câu hỏi: {question}
Đáp án chuẩn (tham khảo): {gold_answer}
Câu trả lời mô hình: {model_answer}

Trả về CHỈ một JSON dạng: {{"correctness": <1-5>, "faithfulness": <1-5>}}""",
)


def llm_judge(question: str, gold_answer: str, model_answer: str) -> dict:
    """Trả {"correctness": int|None, "faithfulness": int|None}. None khi LLM
    trả về không phải JSON hợp lệ (không chặn benchmark vì 1 câu lỗi)."""
    chain = JUDGE_PROMPT | get_llm() | StrOutputParser()
    raw = chain.invoke({"question": question, "gold_answer": gold_answer, "model_answer": model_answer})
    try:
        return json.loads(raw.strip().strip("`").replace("json\n", ""))
    except Exception:
        return {"correctness": None, "faithfulness": None}