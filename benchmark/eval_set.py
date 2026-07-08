"""Tạo/đọc EVAL_SET dùng cho benchmark/run_benchmark.py.

Chỉ cần "question" — không cần ground truth (gold_citations/gold_answer),
vì benchmark chỉ dùng DeepEval (answer_relevancy, faithfulness,
contextual_relevancy đều chấm trực tiếp trên context + answer, không cần
đáp án chuẩn).
"""
from __future__ import annotations

import json
import random
from pathlib import Path


def build_eval_set_from_jsonl(
    jsonl_path: str | Path,
    sample_size: int = 10,
    seed: int = 42,
    category: str = "general",
) -> list[dict]:
    """Random sample câu hỏi từ 1 file jsonl thô (vd file extract), ghép
    title + body thành câu hỏi. Không cần điền gì thêm — dùng chấm ngay
    bằng DeepEval."""
    random.seed(seed)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    samples = random.sample(data, min(sample_size, len(data)))
    eval_set = []
    for item in samples:
        title = item.get("title", "").strip()
        body = item.get("body", "").strip()
        question = " ".join(x for x in [title, body] if x)
        eval_set.append({"question": question, "category": category})
    return eval_set


def load_eval_set(path: str | Path) -> list[dict]:
    """Đọc EVAL_SET đã chuẩn bị sẵn (jsonl, mỗi dòng ít nhất có "question",
    "category" tuỳ chọn để so sánh theo nhóm)."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]