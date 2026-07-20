"""Chạy retrieval.pipeline.run_pipeline() cho TOÀN BỘ câu hỏi trong 1 file
jsonl bất kỳ nằm trong data/ (data/questions.jsonl, data/related_questions.jsonl,
data/short_questions.jsonl, hoặc file jsonl mới nào khác chỉ cần có cột
"question"), lưu kết quả riêng từng mode vào data/retrieval/ dạng CSV.

Vì sao tách riêng bước này khỏi benchmark/run_benchmark.py:
- run_pipeline() (gọi Neo4j + Vertex AI embedding/LLM để retrieve + sinh
  answer) là phần TỐN THỜI GIAN/QUOTA nhất, và kết quả của nó không đổi
  giữa các lần chấm điểm lại bằng framework khác nhau (DeepEval, RAGAS, hay
  thêm framework mới sau này).
- Tách ra để chạy retrieval 1 LẦN, rồi chấm đi chấm lại nhiều lần (đổi
  metric, đổi judge model, so DeepEval vs RAGAS...) mà KHÔNG phải chạy lại
  retrieval — dùng benchmark/run_benchmark_from_retrieval.py để đọc lại.

CSV lưu ra đúng schema mà benchmark/run_benchmark.py đã hỗ trợ sẵn qua
load_precomputed_results()/--precomputed (cột id, mode, question, result,
error — "result" là repr() của dict trả về từ run_pipeline(), đọc lại bằng
ast.literal_eval) — nên cũng dùng trực tiếp được với run_benchmark.py hiện
tại (--precomputed mode=path), không chỉ với run_benchmark_from_retrieval.py.

Usage:
    python -m benchmark.run_retrieval --input data/questions.jsonl
    python -m benchmark.run_retrieval --input data/related_questions.jsonl \
        --modes vector hybrid graphrag --max-workers 4
"""
from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from benchmark.eval_set import load_eval_set
from load.neo4j_client import Neo4jClient
from retrieval.pipeline import run_pipeline

logger = logging.getLogger(__name__)

MODES = ["bm25", "vector", "hybrid", "graphrag", "vector_graph"]

RETRIEVAL_DIR = Path(__file__).parent.parent / "data" / "retrieval"

DEFAULT_MAX_WORKERS = 4


def _run_one(item: dict, mode: str, idx: int, client: Neo4jClient) -> dict:
    """Chạy run_pipeline() cho 1 câu hỏi, trả 1 row đúng schema CSV mà
    benchmark/run_benchmark.py.load_precomputed_results() đọc được."""
    question = item["question"]
    try:
        result = run_pipeline(question, mode, client=client)
        return {
            "id": item.get("id", idx),
            "mode": mode,
            "question": question,
            "result": repr(result),
            "error": "",
        }
    except Exception as e:
        logger.exception("Lỗi mode=%s cho câu hỏi: %s...", mode, question[:80])
        return {
            "id": item.get("id", idx),
            "mode": mode,
            "question": question,
            "result": "",
            "error": str(e),
        }


def run_retrieval_for_mode(
    eval_set: list[dict],
    mode: str,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> pd.DataFrame:
    """Chạy run_pipeline() cho 1 mode trên toàn bộ eval_set, dùng CHUNG 1
    Neo4jClient cho cả batch (khác cách run_benchmark.py._run_one_question
    hiện đang gọi run_pipeline() không kèm client, nên mỗi câu hỏi tự
    mở/đóng driver riêng) — giảm overhead mở driver hàng chục/hàng trăm lần
    khi chạy 1 file jsonl lớn.

    Song song hoá theo CÂU HỎI bằng ThreadPoolExecutor, giống run_benchmark.py.
    """
    client = Neo4jClient()
    rows: list[dict] = []
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(_run_one, item, mode, idx, client): idx
                for idx, item in enumerate(eval_set)
            }
            for future in as_completed(futures):
                rows.append(future.result())
    finally:
        client.close()
    return pd.DataFrame(rows)


def run_retrieval(
    input_path: str | Path,
    modes: list[str] = MODES,
    out_dir: str | Path = RETRIEVAL_DIR,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> dict[str, Path]:
    """Đọc 1 file jsonl bất kỳ, chạy run_pipeline() cho từng mode, lưu mỗi
    mode thành 1 CSV riêng trong out_dir: "{stem}__{mode}.csv" (vd input
    data/related_questions.jsonl, mode=hybrid -> data/retrieval/
    related_questions__hybrid.csv). Trả dict {mode: path_đã_lưu}."""
    eval_set = load_eval_set(input_path)
    stem = Path(input_path).stem
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, Path] = {}
    for mode in modes:
        logger.info("Chạy retrieval mode=%s cho %d câu hỏi từ %s", mode, len(eval_set), input_path)
        df = run_retrieval_for_mode(eval_set, mode, max_workers=max_workers)
        out_path = out_dir / f"{stem}__{mode}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        n_err = int((df["error"] != "").sum())
        logger.info("Đã lưu %s (%d dòng, %d lỗi)", out_path, len(df), n_err)
        saved[mode] = out_path
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chạy run_pipeline() cho 1 file jsonl bất kỳ trong data/, "
                     "lưu kết quả từng mode vào data/retrieval/ để dùng cho benchmark sau."
    )
    parser.add_argument("--input", required=True, help="Path jsonl bất kỳ (mỗi dòng ít nhất có 'question')")
    parser.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    parser.add_argument("--out-dir", default=str(RETRIEVAL_DIR))
    parser.add_argument(
        "--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
        help="Số câu hỏi chạy song song cùng lúc cho MỖI mode (các mode vẫn chạy tuần tự với nhau).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    saved = run_retrieval(args.input, args.modes, args.out_dir, args.max_workers)

    print(f"\nĐã chạy xong retrieval cho {args.input}, lưu vào {args.out_dir}:")
    for mode, path in saved.items():
        print(f"  {mode}: {path}")
    print(
        "\nDùng lại bằng benchmark/run_benchmark_from_retrieval.py, hoặc thủ công qua "
        "run_benchmark.py --precomputed " + " ".join(f"{m}={p}" for m, p in saved.items())
    )


if __name__ == "__main__":
    main()