"""Chạy lại benchmark CHỈ cho các dòng bị lỗi trong 1 file benchmark_results.csv
đã có sẵn (vd lỗi 429 RESOURCE_EXHAUSTED khi hết quota giữa chừng), rồi merge
kết quả mới đè lên các dòng lỗi cũ — không cần chạy lại toàn bộ eval set.

CSV không có cột "error" riêng — 1 dòng được coi là LỖI nếu:
  - bất kỳ cột metric nào (answer_relevancy/faithfulness_deepeval/
    contextual_relevancy) là NaN, HOẶC
  - bất kỳ cột _*_reason nào chứa chuỗi "lỗi:" (do deepeval_judge.py ghi khi
    metric.measure() raise exception).

Usage:
    python -m benchmark.retry_failed \
        --results-csv benchmark/results/benchmark_results.csv \
        --eval-set data/questions.jsonl \
        --max-workers 1
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from benchmark.eval_set import load_eval_set
from benchmark.run_benchmark import DEFAULT_MAX_WORKERS, MODES, run_benchmark

logger = logging.getLogger(__name__)

METRIC_COLS = ["answer_relevancy", "faithfulness_deepeval", "contextual_relevancy"]


def find_failed_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Trả về các dòng bị lỗi: metric NaN hoặc reason chứa 'lỗi:'."""
    metric_cols = [c for c in METRIC_COLS if c in df.columns]
    reason_cols = [c for c in df.columns if c.startswith("_") and c.endswith("_reason")]

    is_failed = pd.Series(False, index=df.index)
    if metric_cols:
        is_failed |= df[metric_cols].isna().any(axis=1)
    for col in reason_cols:
        is_failed |= df[col].astype(str).str.contains("lỗi:", na=False)

    return df[is_failed]


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy lại benchmark cho các dòng bị lỗi trong CSV kết quả")
    parser.add_argument("--results-csv", required=True, help="Path benchmark_results.csv đã chạy trước đó (có dòng lỗi)")
    parser.add_argument("--eval-set", required=True, help="Path jsonl EVAL_SET gốc (để lấy lại category/... cho các câu lỗi)")
    parser.add_argument("--modes", nargs="+", default=None, choices=MODES,
                         help="Giới hạn mode cần retry. Mặc định: tự suy ra từ các mode có dòng lỗi trong CSV.")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--include-reason", action="store_true")
    parser.add_argument("--out-csv", default=None,
                         help="Path CSV kết quả sau merge. Mặc định: ghi đè lên --results-csv.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    df_old = pd.read_csv(args.results_csv, encoding="utf-8-sig")
    df_failed = find_failed_rows(df_old)

    if df_failed.empty:
        logger.info("Không có dòng nào lỗi trong %s — không cần chạy lại gì.", args.results_csv)
        return

    failed_modes = sorted(df_failed["mode"].unique().tolist()) if "mode" in df_failed.columns else MODES
    modes_to_run = args.modes or failed_modes
    failed_questions = set(df_failed["question"].unique().tolist())

    logger.info(
        "Tìm thấy %d dòng lỗi (%d câu hỏi khác nhau, mode: %s) — chạy lại...",
        len(df_failed), len(failed_questions), failed_modes,
    )

    full_eval_set = load_eval_set(args.eval_set)
    retry_eval_set = [item for item in full_eval_set if item["question"] in failed_questions]

    missing = failed_questions - {item["question"] for item in retry_eval_set}
    if missing:
        logger.warning(
            "%d câu hỏi lỗi trong CSV không khớp được với --eval-set (có thể lệch "
            "whitespace/nội dung câu hỏi) — sẽ KHÔNG được chạy lại: %s",
            len(missing), [q[:60] for q in missing],
        )

    df_retry = run_benchmark(
        retry_eval_set, modes_to_run, max_workers=args.max_workers,
        include_reason=args.include_reason, precomputed={},
    )

    # Bỏ các dòng lỗi cũ (đúng question+mode vừa chạy lại), rồi nối kết quả mới vào.
    retried_keys = set(zip(df_retry["question"], df_retry["mode"]))
    keep_mask = ~df_old.apply(lambda r: (r["question"], r["mode"]) in retried_keys, axis=1)
    df_merged = pd.concat([df_old[keep_mask], df_retry], ignore_index=True)

    still_failed = find_failed_rows(df_retry)
    if not still_failed.empty:
        logger.warning(
            "%d dòng vẫn còn lỗi sau khi chạy lại (chạy lại lệnh này lần nữa nếu cần): %s",
            len(still_failed), still_failed["question"].str[:60].tolist(),
        )

    out_path = Path(args.out_csv) if args.out_csv else Path(args.results_csv)
    df_merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(
        "Đã merge: %d dòng chạy lại, %d dòng giữ nguyên từ trước, tổng %d dòng -> %s",
        len(df_retry), keep_mask.sum(), len(df_merged), out_path,
    )


if __name__ == "__main__":
    main()