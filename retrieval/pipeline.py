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

Quan trọng:
- 1 lần gọi run_pipeline() chỉ mở 1 Neo4jClient (1 driver) dùng chung cho
  cả vector_search lẫn expand_graph — tránh mở/đóng driver lặp lại.
- run_pipeline() và retrieve() đều nhận `client`, `embedding`, `keywords`
  optional từ bên ngoài — cho phép benchmark/run_benchmark.py tính
  embedding/keywords 1 lần/câu hỏi rồi tái sử dụng cho nhiều mode (vector/
  hybrid/graphrag đều cần embedding; hybrid/graphrag đều cần keywords),
  thay vì mỗi mode tự gọi lại API. Khi dùng lẻ (không truyền), hành vi cũ
  vẫn giữ nguyên — tự tính bên trong.
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
    embed_question()/extract_legal_keywords() — 2 lệnh gọi API/LLM tốn
    nhất trong bước retrieve. Nếu không truyền, hàm tự tính như cũ.

    mode="hybrid"/"graphrag": nếu cả embedding lẫn keywords đều chưa có,
    tính song song qua ThreadPoolExecutor (2 việc độc lập). Nếu chỉ thiếu
    1 trong 2, chỉ gọi API cho phần còn thiếu.

    Rerank (nếu use_rerank=True): sau khi có seeds thô (RRF hoặc 1 nguồn
    đơn với mode vector/bm25), chấm điểm liên quan bằng LLM và lọc bỏ
    candidate dưới RERANK_MIN_SCORE trước khi cắt còn max_components —
    khắc phục Contextual Relevancy thấp do RRF chỉ xếp theo rank, không
    đánh giá lại độ liên quan ngữ nghĩa thật với câu hỏi.

    Validity_status vẫn ưu tiên sau cùng: rerank chọn candidate liên quan,
    rồi trong số đó văn bản "Còn hiệu lực" được xếp lên đầu.
    """
    if mode == "vector":
        embedding = embedding if embedding is not None else embed_question(question)
        seeds = vector_search(embedding, top_k, client=client)

    elif mode == "bm25":
        seeds = bm25_search(question, top_k)

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

        vec_results = vector_search(embedding, top_k, client=client)
        bm25_results = bm25_search(keywords, top_k)
        seeds = merge_search_results(vec_results, bm25_results)

    else:
        raise ValueError(f"Mode không hợp lệ: {mode}")

    if use_rerank:
        # Rerank trên toàn bộ seeds thô (top_k*2 nguồn), giữ dư hơn max_components
        # 1 chút (max_components + 2) để bước sort validity_status vẫn có lựa chọn.
        seeds = rerank(question, seeds, top_n=max_components + 2, min_score=RERANK_MIN_SCORE)

    # Sort theo rerank_score nếu có (use_rerank=True) — trước đây luôn sort theo
    # "score" (RRF score cũ), khiến bước cắt còn max_components chọn nhầm theo
    # thứ hạng RRF thay vì độ liên quan thật đã tính lại ở rerank(), vô hiệu hoá
    # một phần tác dụng của rerank ngay tại bước chọn cuối cùng. Fallback về
    # "score" khi use_rerank=False (không có rerank_score) để giữ hành vi cũ.
    seeds = sorted(
        seeds,
        key=lambda r: (r.get("validity_status") != "Còn hiệu lực", -r.get("rerank_score", r.get("score", 0))),
    )
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

    client: nếu không truyền, tự mở/đóng 1 Neo4jClient dùng riêng cho lần
    gọi này (hành vi cũ, phù hợp dùng lẻ/app thật — 1 request = 1 driver).
    Khi chạy benchmark nhiều câu hỏi liên tiếp, LUÔN truyền 1 client dùng
    chung cho cả batch (xem benchmark/run_benchmark.py) để tránh mở/đóng
    driver hàng chục/hàng trăm lần.

    embedding/keywords: xem docstring retrieve() — truyền vào để tái sử
    dụng giữa các mode của cùng 1 câu hỏi, tránh gọi lại API embedding/LLM.

    Trả dict gồm answer, context, retrieved (seeds thô — để tính recall/mrr
    trong benchmark/metrics.py), prompt_tokens, latency_sec.
    """
    t0 = time.time()

    owns_client = client is None
    client = client or Neo4jClient()
    try:
        seeds = retrieve(
            question, mode, top_k, max_components,
            client=client, embedding=embedding, keywords=keywords,
        )

        if mode == "graphrag":
            component_ids = [row["comp_id"] for row in seeds if row.get("comp_id")]
            subgraph = expand_graph(component_ids, client=client)
            context = build_context_graph(subgraph, max_chars)
        else:
            context = build_context_flat(seeds, max_chars)
    finally:
        if owns_client:
            client.close()

    prompt_text = RAG_PROMPT.format(question=question, context=context)
    n_tokens = count_tokens(prompt_text)  # ước lượng local, không gọi API (xem prompts.py)
    if n_tokens > max_tokens:
        ratio = max_tokens / n_tokens
        context = context[: int(len(context) * ratio)]
        n_tokens = count_tokens(RAG_PROMPT.format(question=question, context=context))

    answer_text = answer(question, context)
    latency = time.time() - t0

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