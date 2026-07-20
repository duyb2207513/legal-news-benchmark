"""Chấm điểm bằng RAGAS, dùng chung Gemini (qua Vertex AI) làm judge model +
embedding model — bổ sung cho benchmark/deepeval_judge.py, KHÔNG thay thế
(2 framework chấm độc lập, dùng để đối chiếu chéo kết quả với nhau).

Metric KHÔNG cần ground truth (chạy được ngay, giống nhóm DeepEval hiện tại):
- faithfulness: answer có bịa ngoài context không
- answer_relevancy: answer có bám sát question không (cần embedding)
- context_precision: trong context lấy được, tỷ lệ phần liên quan xếp hạng
  cao (dùng biến thể "without reference" — ước lượng qua LLM, không cần
  gold_answer)

Metric CẦN ground truth (chỉ chạy nếu EVAL_SET có "gold_answer", tự động bỏ
qua nếu không có — không chặn benchmark):
- context_recall: context lấy được có phủ đủ thông tin trong gold_answer không
- answer_correctness: answer cuối cùng đúng so với gold_answer không

Dùng LangchainLLMWrapper/LangchainEmbeddingsWrapper để bọc lại đúng
ChatVertexAI/VertexAIEmbeddings đã dùng trong retrieval/prompts.py và
retrieval/embedder.py — tránh phải cấu hình OpenAI key (mặc định của RAGAS).
"""
from __future__ import annotations

import asyncio
import logging

from datasets import Dataset
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerCorrectness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from config import EMBEDDING_MODEL, GCP_LOCATION, GCP_PROJECT, LLM_MODEL_LIGHT

logger = logging.getLogger(__name__)

# Dùng LLM_MODEL_LIGHT (giống GeminiJudge trong deepeval_judge.py) — judge
# không cần model mạnh nhất, và tránh cạnh tranh quota với LLM_MODEL_HEAVY
# đang dùng để sinh answer trong retrieval/prompts.py.
_ragas_llm = None
_ragas_embeddings = None

NO_GT_METRIC_NAMES = ("faithfulness_ragas", "answer_relevancy_ragas", "context_precision_ragas")
GT_METRIC_NAMES = ("context_recall_ragas", "answer_correctness_ragas")
ALL_METRIC_NAMES = NO_GT_METRIC_NAMES + GT_METRIC_NAMES


def _get_ragas_llm() -> LangchainLLMWrapper:
    global _ragas_llm
    if _ragas_llm is None:
        chat = ChatVertexAI(model=LLM_MODEL_LIGHT, project=GCP_PROJECT, location=GCP_LOCATION, temperature=0)
        _ragas_llm = LangchainLLMWrapper(chat)
    return _ragas_llm


def _get_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    global _ragas_embeddings
    if _ragas_embeddings is None:
        emb = VertexAIEmbeddings(model_name=EMBEDDING_MODEL, project=GCP_PROJECT, location=GCP_LOCATION)
        _ragas_embeddings = LangchainEmbeddingsWrapper(emb)
    return _ragas_embeddings


def to_ragas_row(result: dict, gold_answer: str | None = None) -> dict:
    """Chuyển output run_pipeline() (+ gold_answer tuỳ chọn từ EVAL_SET)
    sang 1 row đúng schema RAGAS cần: question/answer/contexts (list[str])
    + ground_truth khi có."""
    chunks = [r["text"] for r in result.get("retrieved", []) if r.get("text")]
    row = {
        "question": result["question"],
        "answer": result["answer"],
        "contexts": chunks if chunks else [result["context"]],
    }
    if gold_answer:
        row["ground_truth"] = gold_answer
    return row


def score_with_ragas(
    result: dict,
    gold_answer: str | None = None,
    metric_names: tuple[str, ...] = ALL_METRIC_NAMES,
) -> dict:
    """Chấm 1 result bằng RAGAS, trả dict phẳng để gộp vào row của
    benchmark/run_benchmark.py — cùng kiểu dùng như score_with_deepeval().

    Metric cần ground truth (context_recall/answer_correctness) tự động bỏ
    qua (giá trị None) nếu gold_answer=None, KHÔNG raise lỗi — cho phép
    chạy RAGAS ngay cả khi EVAL_SET chưa có ground truth, giống cách
    recall_at_k trong benchmark/metrics.py trả None khi thiếu gold_citations.

    Mỗi lần gọi tạo Dataset 1 dòng — kém hiệu quả hơn batch cả eval set 1
    lần (evaluate() của RAGAS vốn thiết kế để chạy batch), nhưng giữ cùng
    pattern per-row như score_with_deepeval() để cắm thẳng vào
    _run_one_question() trong run_benchmark.py mà không phải đổi kiến trúc
    song song hoá theo câu hỏi hiện tại. Nếu cần benchmark eval set lớn,
    nên chuyển sang gọi evaluate() 1 lần cho cả DataFrame (xem gợi ý ở
    docstring run_benchmark.py)."""
    active_no_gt = [m for m in metric_names if m in NO_GT_METRIC_NAMES]
    active_gt = [m for m in metric_names if m in GT_METRIC_NAMES] if gold_answer else []

    scores: dict = {name: None for name in metric_names}
    if not active_no_gt and not active_gt:
        return scores

    row = to_ragas_row(result, gold_answer)
    dataset = Dataset.from_list([row])

    llm = _get_ragas_llm()
    embeddings = _get_ragas_embeddings()

    metric_map = {
        "faithfulness_ragas": Faithfulness(llm=llm),
        "answer_relevancy_ragas": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision_ragas": ContextPrecision(llm=llm),
        "context_recall_ragas": ContextRecall(llm=llm),
        "answer_correctness_ragas": AnswerCorrectness(llm=llm, embeddings=embeddings),
    }
    metrics = [metric_map[name] for name in (active_no_gt + active_gt)]

    try:
        report = evaluate(dataset, metrics=metrics, llm=llm, embeddings=embeddings, show_progress=False)
        df = report.to_pandas()
        for name in active_no_gt + active_gt:
            col = name.replace("_ragas", "")  # cột trả về từ RAGAS không có hậu tố _ragas
            scores[name] = float(df.iloc[0][col]) if col in df.columns else None
    except Exception as e:
        logger.warning("RAGAS lỗi cho câu hỏi '%s...': %s", result["question"][:80], e)
        for name in active_no_gt + active_gt:
            scores[f"{name}_error"] = str(e)

    return scores
