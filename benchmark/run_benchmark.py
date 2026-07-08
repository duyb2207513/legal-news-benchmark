"""Entrypoint benchmark: python -m benchmark.run_benchmark --eval-set ...

So sánh 4 mode retrieval (bm25/vector/hybrid/graphrag) trên cùng 1 EVAL_SET,
chấm bằng DeepEval (answer_relevancy, faithfulness, contextual_relevancy —
không cần ground truth), gộp thành bảng, lưu CSV vào benchmark/results/.
"""
from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from benchmark.deepeval_judge import score_with_deepeval
from benchmark.eval_set import load_eval_set
from retrieval.pipeline import run_pipeline

logger = logging.getLogger(__name__)

MODES = ["bm25", "vector", "hybrid", "graphrag"]

RESULTS_DIR = Path(__file__).parent / "results"

# Số câu hỏi chạy song song cùng lúc. Mỗi câu hỏi vẫn chạy 4 mode TUẦN TỰ bên
# trong (an toàn hơn về rate limit LLM/embedding và tránh 1 câu hỏi chiếm hết
# quota). Đừng để quá cao — mỗi worker gọi Vertex AI (embedding + LLM_HEAVY +
# LLM_LIGHT nếu còn rerank) + Neo4j, dễ dính rate limit nếu chạy quá nhiều
# song song. 4-5 là mức an toàn cho hầu hết quota mặc định.
DEFAULT_MAX_WORKERS = 4


def _run_one_question(item: dict, modes: list[str], include_reason: bool = False) -> list[dict]:
    """Chạy tuần tự tất cả mode cho 1 câu hỏi, trả list row (1 row/mode).

    Tách riêng thành hàm để mỗi câu hỏi có thể chạy trên 1 thread độc lập,
    còn các mode trong CÙNG câu hỏi vẫn chạy tuần tự — tránh tăng concurrency
    quá mức cần thiết (N câu hỏi song song đã đủ để giảm latency đáng kể).
    """
    question = item["question"]
    rows = []
    for mode in modes:
        logger.info("Chạy mode=%s cho câu hỏi: %s...", mode, question[:80])
        result = run_pipeline(question, mode)
        deepeval_scores = score_with_deepeval(result, include_reason=include_reason)

        rows.append({
            "question": question,
            "category": item.get("category"),
            "mode": mode,
            "answer_relevancy": deepeval_scores.get("answer_relevancy"),
            "faithfulness_deepeval": deepeval_scores.get("faithfulness_deepeval"),
            "contextual_relevancy": deepeval_scores.get("contextual_relevancy"),
            "context_len_chars": len(result["context"]),
            "prompt_tokens": result["prompt_tokens"],
            "latency_sec": round(result["latency_sec"], 2),
            # giữ lại reason để debug/đọc thủ công, không show trong summary
            "_answer_relevancy_reason": deepeval_scores.get("answer_relevancy_reason"),
            "_faithfulness_reason": deepeval_scores.get("faithfulness_deepeval_reason"),
            "_contextual_relevancy_reason": deepeval_scores.get("contextual_relevancy_reason"),
        })
    return rows


def run_benchmark(
    eval_set: list[dict],
    modes: list[str] = MODES,
    max_workers: int = DEFAULT_MAX_WORKERS,
    include_reason: bool = False,
) -> pd.DataFrame:
    """Chạy run_pipeline() cho mỗi (câu hỏi, mode), chấm bằng DeepEval, trả
    DataFrame 1 dòng/(câu hỏi, mode).

    Song song hoá theo CÂU HỎI (tối đa `max_workers` câu hỏi cùng lúc) bằng
    ThreadPoolExecutor — mỗi câu hỏi vẫn chạy 4 mode tuần tự bên trong.
    max_workers=1 để quay lại chạy hoàn toàn tuần tự như bản gốc (hữu ích
    khi debug hoặc khi rate limit quá chặt).

    include_reason=True: DeepEval sinh thêm giải thích (_*_reason trong
    CSV) cho TỪNG metric — tốn thêm ~1 lần gọi LLM/metric/câu hỏi nên mặc
    định tắt. Bật lên khi cần hiểu TẠI SAO 1 metric thấp (vd faithfulness
    thấp: reason sẽ chỉ đúng câu/nhận định nào trong answer không có căn cứ
    trong context) thay vì chỉ nhìn con số.
    """
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_run_one_question, item, modes, include_reason): item for item in eval_set}
        for future in as_completed(futures):
            item = futures[future]
            try:
                rows.extend(future.result())
            except Exception:
                logger.exception("Lỗi khi chạy câu hỏi: %s...", item["question"][:80])
                raise

    return pd.DataFrame(rows)


def summarize(df_results: pd.DataFrame) -> pd.DataFrame:
    """Trung bình các metric số theo mode — bảng chính để so sánh 4 mode."""
    return df_results.groupby("mode")[[
        "answer_relevancy", "faithfulness_deepeval", "contextual_relevancy",
        "context_len_chars", "prompt_tokens", "latency_sec",
    ]].mean(numeric_only=True).round(3)


def summarize_by_category(df_results: pd.DataFrame) -> pd.DataFrame | None:
    """Trung bình theo category x mode — GraphRAG kỳ vọng nổi bật ở nhóm câu
    hỏi cần suy luận nhiều văn bản. Trả None nếu EVAL_SET không có category."""
    if "category" not in df_results.columns:
        return None
    return df_results.groupby(["category", "mode"])[[
        "answer_relevancy", "faithfulness_deepeval", "contextual_relevancy",
    ]].mean(numeric_only=True).round(3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark 4 mode retrieval (bm25/vector/hybrid/graphrag) bằng DeepEval")
    parser.add_argument("--eval-set", required=True, help="Path jsonl EVAL_SET (xem benchmark/eval_set.py)")
    parser.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    parser.add_argument(
        "--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
        help="Số câu hỏi chạy song song cùng lúc (mỗi câu hỏi vẫn chạy các mode "
             "tuần tự bên trong). Dùng 1 để chạy tuần tự hoàn toàn.",
    )
    parser.add_argument(
        "--include-reason", action="store_true",
        help="Bật DeepEval sinh giải thích cho từng metric (cột _*_reason trong "
             "CSV) — tốn thêm ~1 lần gọi LLM/metric/câu hỏi. Mặc định tắt.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    eval_set = load_eval_set(args.eval_set)
    df_results = run_benchmark(eval_set, args.modes, max_workers=args.max_workers, include_reason=args.include_reason)
    summary = summarize(df_results)

    print("=== Trung bình theo mode (toàn bộ eval set) ===")
    print(summary)

    summary_by_cat = summarize_by_category(df_results)
    if summary_by_cat is not None:
        print("\n=== Trung bình theo category x mode ===")
        print(summary_by_cat)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(out_dir / "benchmark_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "benchmark_summary.csv", encoding="utf-8-sig")
    print(f"\nĐã lưu benchmark_results.csv và benchmark_summary.csv vào {out_dir}")


if __name__ == "__main__":
    main()