"""Chấm điểm KHÔNG cần ground truth, dùng DeepEval + Gemini làm judge model:

- answer_relevancy: câu trả lời có bám sát câu hỏi không
- faithfulness: câu trả lời có bịa ngoài context đã đưa không
- contextual_relevancy: context lấy được có liên quan tới câu hỏi không

Bổ sung cho benchmark/metrics.py (cần gold_citations/gold_answer) — dùng
được ngay cả khi EVAL_SET chưa có ground truth, phù hợp benchmark nhanh.
"""
from __future__ import annotations

import threading

from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric, FaithfulnessMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from google.genai.types import GenerateContentConfig

from config import GCP_LOCATION, GCP_PROJECT, LLM_MODEL_LIGHT

# Delimiter dùng để tách context thành từng chunk (1 Component/TextUnit) cho
# ContextualRelevancyMetric chấm chính xác hơn — PHẢI khớp với delimiter thật
# sự dùng trong retrieval/context_builder.py (build_context_flat/graph nối
# các đoạn bằng "---", KHÔNG phải "============================" như bản
# notebook gốc ghi nhầm trong docstring — đã sửa lại đúng ở đây).
_CONTEXT_CHUNK_DELIMITER = "---"


class GeminiJudge(DeepEvalBaseLLM):
    """Wrapper để dùng Gemini (qua Vertex AI) làm judge model cho DeepEval,
    thay vì mặc định OpenAI mà DeepEval built-in hỗ trợ."""

    def __init__(self, model_name: str = LLM_MODEL_LIGHT):
        from google import genai
        from google.genai.types import HttpOptions

        self.model_name = model_name
        # timeout=120s — không set sẽ treo vô thời hạn nếu Vertex AI phản hồi
        # chậm/quá tải, trông giống loop vô tận nhưng thực ra đang chờ mạng.
        self.client = genai.Client(
            vertexai=True,
            project=GCP_PROJECT,
            location=GCP_LOCATION,
            http_options=HttpOptions(timeout=120_000),
        )

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=GenerateContentConfig(temperature=0),
        )
        return response.text

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name


_judge_llm = None
_judge_llm_lock = threading.Lock()

ALL_METRIC_NAMES = ("answer_relevancy", "faithfulness_deepeval", "contextual_relevancy")


def _get_metrics(metric_names: tuple[str, ...] = ALL_METRIC_NAMES, include_reason: bool = False) -> dict:
    """Tạo metric MỚI mỗi lần gọi — KHÔNG cache dùng chung giữa các thread
    nữa (bản trước dùng _metrics_cache toàn cục, khiến mọi thread trong
    ThreadPoolExecutor của run_benchmark.py cùng lấy về CHUNG 1 instance
    metric). Metric object của DeepEval lưu score/reason làm INSTANCE STATE
    sau khi .measure() — gọi song song trên cùng 1 object từ nhiều thread
    sẽ ghi đè chéo nhau (race condition): quan sát được thực tế qua
    score/reason lệch nhau trong cùng 1 row benchmark (vd contextual_relevancy
    ghi 0.61 nhưng reason lại giải thích cho điểm 0.08 của 1 câu hỏi khác).

    Tạo mới ở đây chỉ là khởi tạo object Python (không gọi API), nên không
    tốn thêm chi phí đáng kể — mỗi thread giờ có instance riêng, an toàn.

    judge_llm (client gọi Vertex AI) vẫn tái sử dụng giữa các thread vì nó
    KHÔNG giữ state per-measure (chỉ là HTTP client, mỗi generate_content()
    độc lập) — khởi tạo qua lock để tránh 2 thread cùng tạo GeminiJudge()
    khi gọi lần đầu gần như đồng thời (double-checked locking).
    """
    global _judge_llm
    if _judge_llm is None:
        with _judge_llm_lock:
            if _judge_llm is None:
                _judge_llm = GeminiJudge()

    metrics = {}
    if "answer_relevancy" in metric_names:
        metrics["answer_relevancy"] = AnswerRelevancyMetric(threshold=0.7, model=_judge_llm, include_reason=include_reason)
    if "faithfulness_deepeval" in metric_names:
        metrics["faithfulness_deepeval"] = FaithfulnessMetric(threshold=0.7, model=_judge_llm, include_reason=include_reason)
    if "contextual_relevancy" in metric_names:
        metrics["contextual_relevancy"] = ContextualRelevancyMetric(threshold=0.6, model=_judge_llm, include_reason=include_reason)
    return metrics


def to_test_case(result: dict) -> LLMTestCase:
    """Chuyển output của retrieval.pipeline.run_pipeline() sang LLMTestCase
    cho DeepEval — dùng thẳng result["retrieved"] (list seed thô, mỗi phần
    tử 1 chunk sạch, không lẫn header/citation/action-annotation của
    context_builder.py và không bị cắt ngang bởi truncation theo
    max_tokens) thay vì split result["context"] theo delimiter."""
    chunks = [r["text"] for r in result.get("retrieved", []) if r.get("text")]
    return LLMTestCase(
        input=result["question"],
        actual_output=result["answer"],
        retrieval_context=chunks if chunks else [result["context"]],  # fallback nếu retrieved rỗng
    )


def score_with_deepeval(
    result: dict,
    metric_names: tuple[str, ...] = ALL_METRIC_NAMES,
    include_reason: bool = False,
) -> dict:
    """Chạy các metric DeepEval được chọn (mặc định cả 3) trên 1 result của
    run_pipeline(), trả về dict phẳng để gộp thẳng vào row của
    benchmark/run_benchmark.py.

    Chỉ chạy 1 metric (vd metric_names=("answer_relevancy",)) sẽ giảm số
    lần gọi LLM xuống ~1/3 so với chạy cả 3 — dùng khi test nhanh hoặc
    quota/chi phí hạn chế.

    Mỗi metric lỗi độc lập (vd timeout Vertex AI) không chặn metric còn
    lại — ghi None + lý do lỗi vào *_reason để debug sau."""
    metrics = _get_metrics(metric_names, include_reason)
    test_case = to_test_case(result)
    scores = {}
    for name, metric in metrics.items():
        try:
            metric.measure(test_case)
            scores[name] = metric.score
            scores[f"{name}_reason"] = metric.reason
        except Exception as e:
            scores[name] = None
            scores[f"{name}_reason"] = f"lỗi: {e}"
    return scores