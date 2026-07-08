"""Reciprocal Rank Fusion (RRF) — hợp nhất kết quả từ nhiều nguồn retrieval
(vector search, BM25) thành 1 danh sách xếp hạng duy nhất, dùng cho mode
"hybrid" và "graphrag" trong retrieval/pipeline.py.

RRF thay vì cộng trực tiếp điểm số vì vector score (cosine similarity) và
BM25 score (BM25F) không cùng thang đo — RRF chỉ dựa vào RANK (thứ hạng)
của mỗi kết quả trong từng nguồn, nên công bằng giữa 2 nguồn khác thang.
Vì lý do tương tự, KHÔNG được so sánh raw "score" (cosine similarity vs
BM25F) ở bất kỳ bước nào khác của hàm này (chọn dòng đại diện, tie-break...)
— chỉ rank hoặc rrf_score (đã cùng thang) mới được dùng để so sánh giữa
2 nguồn.
"""
from __future__ import annotations


def merge_search_results(*result_lists: list[dict], k: int = 60) -> list[dict]:
    """Hợp nhất nhiều list kết quả (mỗi list đã sort theo score giảm dần)
    bằng RRF: rrf(row) = sum(1 / (k + rank_i)) qua các nguồn i chứa row.

    Nguồn đầu tiên (result_lists[0], quy ước là vector search) được thưởng
    hệ số 1.2 — vector search thường có precision cao hơn BM25 cho câu hỏi
    tự nhiên (so với từ khoá), nên ưu tiên nhẹ khi 2 nguồn đồng thuận.

    Dòng đại diện cho mỗi Component: chọn dòng có RANK tốt nhất (nhỏ nhất)
    trong nguồn của nó — không so sánh raw score giữa 2 nguồn (khác thang,
    xem docstring module) vì BM25F score gần như luôn lớn hơn cosine
    similarity một cách "giả tạo", dễ chọn nhầm dòng BM25 làm đại diện dù
    vector search xếp hạng nó tốt hơn.

    Kết quả cuối: sort theo (còn hiệu lực trước, rrf_score giảm dần, rồi
    best_rank tăng dần làm tie-break — rank thấp hơn = tốt hơn), và loại
    trùng theo (norm_id, citation) — giữ dòng có rrf_score cao nhất cho
    mỗi Component.
    """
    rrf_scores: dict[str, float] = {}
    best_row: dict[str, dict] = {}
    best_rank: dict[str, float] = {}

    for source_idx, results in enumerate(result_lists):
        for rank, row in enumerate(results, start=1):
            comp_id = row.get("comp_id")
            if not comp_id:
                continue
            rrf = 1.0 / (k + rank)
            if source_idx == 0:  # nguồn đầu tiên (vector) được thưởng nhẹ
                rrf *= 1.2
            rrf_scores[comp_id] = rrf_scores.get(comp_id, 0) + rrf

            # Chọn dòng đại diện + best_rank theo RANK (so sánh công bằng
            # giữa 2 nguồn), KHÔNG theo raw score (khác thang, xem docstring).
            if comp_id not in best_rank or rank < best_rank[comp_id]:
                best_rank[comp_id] = rank
                best_row[comp_id] = row

    merged = []
    for comp_id, score in rrf_scores.items():
        row = dict(best_row[comp_id])
        row["rrf_score"] = score
        row["original_score"] = row.get("score", 0)  # giữ lại để debug, KHÔNG dùng để sort/so sánh
        row["score"] = score
        row["best_rank"] = best_rank[comp_id]
        merged.append(row)

    merged.sort(key=lambda r: (r.get("validity_status") != "Còn hiệu lực", -r["score"], r["best_rank"]))

    filtered, seen = [], set()
    for row in merged:
        key = (row.get("norm_id"), row.get("citation"))
        if key in seen:
            continue
        seen.add(key)
        filtered.append(row)
    return filtered