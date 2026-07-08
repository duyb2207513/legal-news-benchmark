"""Ráp toàn bộ module retrieval/ thành 2 hàm dùng trực tiếp:

- retrieve(question, mode): chỉ lấy seeds (Component) — dùng khi chỉ cần
  test/benchmark chất lượng retrieval, chưa cần sinh câu trả lời.
- run_pipeline(question, mode): retrieve + build context + sinh câu trả
  lời — hàm end-to-end dùng cho cả app thật lẫn benchmark/run_benchmark.py.

4 mode:
- "vector":   chỉ vector search (embedding similarity)
- "bm25":     chỉ BM25 (từ khoá chính xác)
- "hybrid":   vector + BM25 (trên từ khoá đã trích) hợp nhất qua RRF
- "graphrag": giống hybrid, nhưng build context có mở rộng qua graph
              (Norm cha, Action sửa đổi/bị sửa đổi) thay vì context phẳng

>>> BẢN NÀY CÓ THÊM LOG TIMING (print [TIMING] ...) ĐỂ DEBUG LATENCY. <<<
Xoá/hoặc đổi print() thành logger.debug() sau khi đã xác định xong bottleneck.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from config import (
    MAX_CHARS_PER_UNIT,
    MAX_COMPONENTS,
    MAX_PROMPT_TOKENS,
    RERANK_MIN_SCORE,
    TOP_K,
    USE_RERANK,
)
from load.neo4j_client import Neo4jClient
from retrieval.bm25_index import bm25_search
from retrieval.context_builder import build_context_flat, build_context_graph
from retrieval.embedder import embed_question
from retrieval.fusion import merge_search_results
from retrieval.graph_expand import expand_graph
from retrieval.keyword_extractor import extract_legal_keywords
from retrieval.prompts import RAG_PROMPT, answer, count_tokens
from retrieval.reranker import rerank
from retrieval.vector_search import vector_search


def _log(label: str, t_start: float) -> float:
    """In ra thời gian đã trôi qua kể từ t_start, trả về mốc thời gian mới
    (để log tiếp bước sau). Chỉ dùng cho debug — xoá khi đã xong.

    flush=True: bắt buộc để log hiện ra NGAY trong Jupyter/notebook, tránh
    bị buffer và chỉ hiện hết 1 lần khi cell chạy xong (dễ gây cảm giác
    "không in ra gì" trong lúc đang chạy).
    """
    now = time.time()
    print(f"[TIMING] {label}: {now - t_start:.2f}s", flush=True)
    return now


def retrieve(
    question: str,
    mode: str,
    top_k: int = TOP_K,
    max_components: int = MAX_COMPONENTS,
    use_rerank: bool = USE_RERANK,
    client: Neo4jClient | None = None,
    embedding: list[float] | None = None,
    keywords: str | None = None,
) -> list[dict]:
    """Trả về tối đa `max_components` Component liên quan nhất.

    embedding/keywords: nếu đã tính sẵn ở caller (vd benchmark tính 1 lần
    cho cả 3 mode dùng chung câu hỏi), truyền vào để bỏ qua việc gọi lại
    embed_question()/extract_legal_keywords(). Nếu không truyền, tự tính.

    Rerank (nếu use_rerank=True): sau khi có seeds thô, chấm điểm liên quan
    bằng LLM và lọc bỏ candidate dưới RERANK_MIN_SCORE.
    """
    t = time.time()

    if mode == "vector":
        embedding = embedding if embedding is not None else embed_question(question)
        t = _log("embed_question", t)
        seeds = vector_search(embedding, top_k, client=client)
        t = _log("vector_search", t)

    elif mode == "bm25":
        seeds = bm25_search(question, top_k)
        t = _log("bm25_search", t)

    elif mode in ("hybrid", "graphrag"):
        need_embedding = embedding is None
        need_keywords = keywords is None

        if need_embedding and need_keywords:
            with ThreadPoolExecutor(max_workers=2) as ex:
                f_embed = ex.submit(embed_question, question)
                f_keywords = ex.submit(extract_legal_keywords, question)
                embedding = f_embed.result()
                keywords = f_keywords.result()
        elif need_embedding:
            embedding = embed_question(question)
        elif need_keywords:
            keywords = extract_legal_keywords(question)
        t = _log("embed_question + extract_legal_keywords (song song)", t)

        vec_results = vector_search(embedding, top_k, client=client)
        t = _log("vector_search", t)

        bm25_results = bm25_search(keywords, top_k)
        t = _log("bm25_search", t)

        seeds = merge_search_results(vec_results, bm25_results)
        t = _log("merge_search_results (RRF, local)", t)

        print(f"[TIMING] seeds trước rerank: {len(seeds)} candidate", flush=True)

    else:
        raise ValueError(f"Mode không hợp lệ: {mode}")

    if use_rerank:
        # Rerank trên toàn bộ seeds thô (top_k*2 nguồn), giữ dư hơn max_components
        # 1 chút (max_components + 2) để bước sort validity_status vẫn có lựa chọn.
        candidates_chars = sum(len((row.get("text") or "")[:MAX_CHARS_PER_UNIT]) for row in seeds)
        print(f"[TIMING] rerank prompt ước tính ~{candidates_chars} ký tự cho {len(seeds)} candidate", flush=True)
        seeds = rerank(question, seeds, top_n=max_components + 2, min_score=RERANK_MIN_SCORE)
        t = _log("rerank (LLM call)", t)

    seeds = sorted(seeds, key=lambda r: (r.get("validity_status") != "Còn hiệu lực", -r.get("score", 0)))
    return seeds[:max_components]


def run_pipeline(
    question: str,
    mode: str,
    top_k: int = TOP_K,
    max_components: int = MAX_COMPONENTS,
    max_chars: int = MAX_CHARS_PER_UNIT,
    max_tokens: int = MAX_PROMPT_TOKENS,
    client: Neo4jClient | None = None,
    embedding: list[float] | None = None,
    keywords: str | None = None,
) -> dict:
    """End-to-end: retrieve → build context (flat hoặc graph tuỳ mode) →
    cắt bớt context nếu vượt max_tokens → sinh câu trả lời.
    """
    t0 = time.time()
    t = t0

    owns_client = client is None
    client = client or Neo4jClient()
    t = _log("mở Neo4jClient", t)
    try:
        seeds = retrieve(
            question, mode, top_k, max_components,
            client=client, embedding=embedding, keywords=keywords,
        )
        t = _log("retrieve() TỔNG", t)

        if mode == "graphrag":
            component_ids = [row["comp_id"] for row in seeds if row.get("comp_id")]
            subgraph = expand_graph(component_ids, client=client)
            t = _log("expand_graph", t)
            context = build_context_graph(subgraph, max_chars)
        else:
            context = build_context_flat(seeds, max_chars)
        t = _log("build_context (local)", t)
    finally:
        if owns_client:
            client.close()

    prompt_text = RAG_PROMPT.format(question=question, context=context)
    n_tokens = count_tokens(prompt_text)  # ước lượng local, không gọi API
    if n_tokens > max_tokens:
        ratio = max_tokens / n_tokens
        context = context[: int(len(context) * ratio)]
        n_tokens = count_tokens(RAG_PROMPT.format(question=question, context=context))
    t = _log("count_tokens + cắt context (local)", t)

    print(f"[TIMING] prompt_tokens gửi cho answer(): {n_tokens}", flush=True)
    answer_text = answer(question, context)
    t = _log("answer (LLM call)", t)

    latency = time.time() - t0
    print(f"[TIMING] ===== TỔNG run_pipeline: {latency:.2f}s =====", flush=True)

    return {
        "question": question,
        "mode": mode,
        "answer": answer_text,
        "context": context,
        "retrieved": seeds,
        "prompt_tokens": n_tokens,
        "latency_sec": latency,
        "retrieved_citations": [(r.get("norm_number"), r.get("citation")) for r in seeds],
    }