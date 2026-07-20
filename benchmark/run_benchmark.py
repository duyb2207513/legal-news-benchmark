"""Entrypoint benchmark: python -m benchmark.run_benchmark --eval-set ...

So sánh 5 mode retrieval (bm25/vector/hybrid/graphrag/vector_graph) trên cùng 1 EVAL_SET,
chấm bằng DeepEval (answer_relevancy, faithfulness, contextual_relevancy —
không cần ground truth), gộp thành bảng, lưu CSV vào benchmark/results/.
"""
from __future__ import annotations

import argparse
import ast
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ast
import pandas as pd

from benchmark.deepeval_judge import score_with_deepeval
from benchmark.ragas_judge import score_with_ragas
from benchmark.eval_set import load_eval_set
from retrieval.pipeline import run_pipeline

logger = logging.getLogger(__name__)

MODES = ["bm25", "vector", "hybrid", "graphrag", "vector_graph"]

RESULTS_DIR = Path(__file__).parent / "results"

# Số câu hỏi chạy song song cùng lúc. Mỗi câu hỏi vẫn chạy 4 mode TUẦN TỰ bên
# trong (an toàn hơn về rate limit LLM/embedding và tránh 1 câu hỏi chiếm hết
# quota). Đừng để quá cao — mỗi worker gọi Vertex AI (embedding + LLM_HEAVY +
# LLM_LIGHT nếu còn rerank) + Neo4j, dễ dính rate limit nếu chạy quá nhiều
# song song. 4-5 là mức an toàn cho hầu hết quota mặc định.
DEFAULT_MAX_WORKERS = 4


def load_precomputed_results(path: str | Path) -> dict[str, dict]:
    """Đọc 1 file CSV kết quả run_pipeline đã chạy sẵn (vd
    data/results/results_hybrid.csv, các cột: id, mode, question, result,
    error), trả dict {question: result_dict} để tra cứu nhanh trong
    _run_one_question thay vì gọi lại run_pipeline().

    Bỏ qua các dòng có lỗi (cột "error" khác rỗng/NaN) — những câu hỏi này
    sẽ tự động fallback sang gọi run_pipeline() bình thường vì không có
    trong dict trả về.
    """
    df = pd.read_csv(path)
    lookup: dict[str, dict] = {}
    for _, row in df.iterrows():
        if pd.notna(row.get("error")):
            continue
        question = row["question"]
        try:
            result = ast.literal_eval(row["result"])
        except (ValueError, SyntaxError):
            logger.warning("Không parse được cột 'result' cho câu hỏi: %s...", str(question)[:80])
            continue
        lookup[question] = result
    return lookup


def _parse_precomputed(items: list[str]) -> dict[str, str]:
    """Parse list "mode=path.csv" -> {mode: path}. Raise nếu format sai."""
    result = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--precomputed phải dạng mode=path.csv, nhận: {item!r}")
        mode, path = item.split("=", 1)
        result[mode] = path
    return result


def _load_precomputed_result(path: str, question: str) -> dict | None:
    """Đọc 1 dòng result đã chạy sẵn từ CSV (khớp theo question), trả về
    None nếu không tìm thấy — caller sẽ tự fallback chạy run_pipeline()."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    matched = df[df["question"] == question]
    if matched.empty:
        return None
    row = matched.iloc[0]
    result = row["result"] if "result" in row else row.to_dict()
    if isinstance(result, str):
        result = ast.literal_eval(result)
    return result


def _run_one_question(
    item: dict,
    modes: list[str],
    include_reason: bool = False,
    precomputed: dict[str, dict[str, dict]] | None = None,
    with_ragas: bool = False,
) -> list[dict]:
    """Chạy tuần tự tất cả mode cho 1 câu hỏi, trả list row (1 row/mode).

    Tách riêng thành hàm để mỗi câu hỏi có thể chạy trên 1 thread độc lập,
    còn các mode trong CÙNG câu hỏi vẫn chạy tuần tự — tránh tăng concurrency
    quá mức cần thiết (N câu hỏi song song đã đủ để giảm latency đáng kể).

    precomputed: {mode: {question: result_dict}} — nếu mode có mặt ở đây VÀ
    tìm thấy câu hỏi tương ứng, dùng lại result đã chạy sẵn (đọc từ CSV)
    thay vì gọi run_pipeline() (tránh chạy lại retrieval tốn kém/tốn quota).
    Mode không có trong precomputed, hoặc câu hỏi không tìm thấy trong đó,
    vẫn chạy run_pipeline() như cũ.
    """
    question = item["question"]
    rows = []
    for mode in modes:
        cached = (precomputed or {}).get(mode, {}).get(question)
        if cached is not None:
            logger.info("Dùng kết quả có sẵn (precomputed) mode=%s cho câu hỏi: %s...", mode, question[:80])
            result = cached
        else:
            logger.info("Chạy mode=%s cho câu hỏi: %s...", mode, question[:80])
            result = run_pipeline(question, mode,max_components=5,top_k=5, use_rerank=False)
        deepeval_scores = score_with_deepeval(result, include_reason=include_reason)

        row = {
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
        }

        if with_ragas:
            # gold_answer tuỳ chọn trong EVAL_SET -> tự bật thêm
            # context_recall/answer_correctness, bỏ qua (None) nếu không có
            ragas_scores = score_with_ragas(result, gold_answer=item.get("gold_answer"))
            row.update({
                "faithfulness_ragas": ragas_scores.get("faithfulness_ragas"),
                "answer_relevancy_ragas": ragas_scores.get("answer_relevancy_ragas"),
                "context_precision_ragas": ragas_scores.get("context_precision_ragas"),
                "context_recall_ragas": ragas_scores.get("context_recall_ragas"),
                "answer_correctness_ragas": ragas_scores.get("answer_correctness_ragas"),
            })

        rows.append(row)
    return rows


def run_benchmark(
    eval_set: list[dict],
    modes: list[str] = MODES,
    max_workers: int = DEFAULT_MAX_WORKERS,
    include_reason: bool = False,
    precomputed: dict[str, dict[str, dict]] | None = None,
    with_ragas: bool = False,
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

    precomputed: {mode: {question: result_dict}}, dùng khi đã có sẵn kết
    quả run_pipeline() lưu ở CSV (vd data/results/results_hybrid.csv) và
    chỉ muốn chấm điểm DeepEval lại mà KHÔNG chạy lại retrieval cho mode
    đó. Dùng hàm load_precomputed_results() để build dict này từ CSV.
    """
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_run_one_question, item, modes, include_reason, precomputed, with_ragas): item
            for item in eval_set
        }
        for future in as_completed(futures):
            item = futures[future]
            try:
                rows.extend(future.result())
            except Exception:
                logger.exception("Lỗi khi chạy câu hỏi: %s...", item["question"][:80])
                raise

    return pd.DataFrame(rows)


def summarize(df_results: pd.DataFrame) -> pd.DataFrame:
    """Trung bình các metric số theo mode — bảng chính để so sánh các mode."""
    cols = [
        "answer_relevancy", "faithfulness_deepeval", "contextual_relevancy",
        "context_len_chars", "prompt_tokens", "latency_sec",
    ]
    ragas_cols = [
        "faithfulness_ragas", "answer_relevancy_ragas", "context_precision_ragas",
        "context_recall_ragas", "answer_correctness_ragas",
    ]
    cols += [c for c in ragas_cols if c in df_results.columns]
    return df_results.groupby("mode")[cols].mean(numeric_only=True).round(3)


def summarize_by_category(df_results: pd.DataFrame) -> pd.DataFrame | None:
    """Trung bình theo category x mode — GraphRAG kỳ vọng nổi bật ở nhóm câu
    hỏi cần suy luận nhiều văn bản. Trả None nếu EVAL_SET không có category."""
    if "category" not in df_results.columns:
        return None
    return df_results.groupby(["category", "mode"])[[
        "answer_relevancy", "faithfulness_deepeval", "contextual_relevancy",
    ]].mean(numeric_only=True).round(3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark 5 mode retrieval (bm25/vector/hybrid/graphrag/vector_graph) bằng DeepEval")
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
    parser.add_argument(
        "--with-ragas", action="store_true",
        help="Chấm thêm bằng RAGAS (faithfulness, answer_relevancy, "
             "context_precision — không cần ground truth; context_recall, "
             "answer_correctness tự bật thêm nếu EVAL_SET có 'gold_answer'). "
             "Tốn thêm lệnh gọi LLM/embedding nên mặc định tắt.",
    )
    parser.add_argument(
        "--precomputed", nargs="+", default=[], metavar="MODE=PATH",
        help="Dùng kết quả run_pipeline() đã chạy sẵn, lưu ở CSV, thay vì "
             "chạy lại retrieval cho mode đó. Truyền dạng mode=path, có thể "
             "lặp lại nhiều mode. VD: --precomputed hybrid=data/results/results_hybrid.csv. "
             "Mode/câu hỏi nào không có trong CSV vẫn tự chạy run_pipeline() bình thường.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    precomputed: dict[str, dict[str, dict]] = {}
    for spec in args.precomputed:
        mode, _, path = spec.partition("=")
        if not path:
            raise ValueError(f"--precomputed phải có dạng mode=path, nhận được: {spec!r}")
        precomputed[mode] = load_precomputed_results(path)
        logger.info("Đã nạp %d kết quả có sẵn cho mode=%s từ %s", len(precomputed[mode]), mode, path)

    eval_set = load_eval_set(args.eval_set)
    df_results = run_benchmark(
        eval_set, args.modes, max_workers=args.max_workers,
        include_reason=args.include_reason, precomputed=precomputed,
        with_ragas=args.with_ragas,
    )
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