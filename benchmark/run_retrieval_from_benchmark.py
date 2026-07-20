"""Chấm điểm (DeepEval + tuỳ chọn RAGAS) dựa HOÀN TOÀN trên kết quả retrieval
đã lưu sẵn trong data/retrieval/ (do benchmark/run_retrieval.py tạo ra),
KHÔNG gọi lại run_pipeline() — chỉ còn tốn lệnh gọi LLM cho phần judge.

Tự động tìm các file "{stem}__{mode}.csv" trong data/retrieval/ khớp với
--input, nạp qua load_precomputed_results() (có sẵn trong
benchmark/run_benchmark.py) rồi gọi thẳng run_benchmark() — mode/câu hỏi
nào lỡ THIẾU trong CSV sẽ tự fallback gọi run_pipeline() sống, không chặn
benchmark (hành vi có sẵn của run_benchmark.py).

Usage:
    # đã chạy trước: python -m benchmark.run_retrieval --input data/questions.jsonl
    python -m benchmark.run_benchmark_from_retrieval --input data/questions.jsonl
    python -m benchmark.run_benchmark_from_retrieval --input data/questions.jsonl --with-ragas
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from benchmark.eval_set import load_eval_set
from benchmark.run_retrieval import MODES, RETRIEVAL_DIR
from benchmark.run_benchmark import (
    DEFAULT_MAX_WORKERS,
    RESULTS_DIR,
    load_precomputed_results,
    run_benchmark,
    summarize,
    summarize_by_category,
)

logger = logging.getLogger(__name__)


def find_retrieval_csvs(input_stem: str, modes: list[str], retrieval_dir: str | Path = RETRIEVAL_DIR) -> dict[str, Path]:
    """Tìm các file data/retrieval/{input_stem}__{mode}.csv đã có sẵn (do
    run_retrieval.py tạo). Mode nào không tìm thấy file bị BỎ QUA (không
    raise) — run_benchmark() sẽ tự chạy run_pipeline() sống cho mode đó."""
    retrieval_dir = Path(retrieval_dir)
    found: dict[str, Path] = {}
    for mode in modes:
        path = retrieval_dir / f"{input_stem}__{mode}.csv"
        if path.exists():
            found[mode] = path
        else:
            logger.warning(
                "Không tìm thấy %s — mode=%s sẽ tự chạy run_pipeline() sống (không dùng precomputed).",
                path, mode,
            )
    return found


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark bằng DeepEval (+ tuỳ chọn RAGAS) dựa trên kết quả "
                     "retrieval đã lưu sẵn trong data/retrieval/ (xem benchmark/run_retrieval.py)."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path jsonl gốc đã dùng để chạy run_retrieval.py (vd data/questions.jsonl) "
             "— dùng để suy ra tên file CSV trong data/retrieval/ và để lấy lại "
             "category/gold_answer cho từng câu hỏi.",
    )
    parser.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    parser.add_argument("--retrieval-dir", default=str(RETRIEVAL_DIR))
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--include-reason", action="store_true")
    parser.add_argument(
        "--with-ragas", action="store_true",
        help="Chấm thêm bằng RAGAS (xem benchmark/ragas_judge.py). Mặc định tắt.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    input_stem = Path(args.input).stem
    csv_paths = find_retrieval_csvs(input_stem, args.modes, args.retrieval_dir)
    if not csv_paths:
        raise SystemExit(
            f"Không tìm thấy CSV nào trong {args.retrieval_dir} cho '{input_stem}' — "
            f"chạy trước: python -m benchmark.run_retrieval --input {args.input}"
        )

    precomputed = {}
    for mode, path in csv_paths.items():
        precomputed[mode] = load_precomputed_results(path)
        logger.info("Đã nạp %d kết quả precomputed cho mode=%s từ %s", len(precomputed[mode]), mode, path)

    eval_set = load_eval_set(args.input)
    df_results = run_benchmark(
        eval_set, args.modes, max_workers=args.max_workers,
        include_reason=args.include_reason, precomputed=precomputed,
        with_ragas=args.with_ragas,
    )
    summary = summarize(df_results)

    print("=== Trung bình theo mode (toàn bộ eval set, dùng data/retrieval/ có sẵn) ===")
    print(summary)

    summary_by_cat = summarize_by_category(df_results)
    if summary_by_cat is not None:
        print("\n=== Trung bình theo category x mode ===")
        print(summary_by_cat)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(out_dir / f"{input_stem}_benchmark_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / f"{input_stem}_benchmark_summary.csv", encoding="utf-8-sig")
    print(f"\nĐã lưu {input_stem}_benchmark_results.csv và {input_stem}_benchmark_summary.csv vào {out_dir}")


if __name__ == "__main__":
    main()