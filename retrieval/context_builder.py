"""Ghép seed results (sau fusion/graph_expand) thành 1 chuỗi context để đưa
vào RAG_PROMPT. 2 chế độ tương ứng 2 nhóm mode trong retrieval/pipeline.py:

- build_context_flat: dùng cho mode "vector"/"bm25"/"hybrid" — mỗi seed là
  1 đoạn text độc lập, không có quan hệ graph.
- build_context_graph: dùng cho mode "graphrag" — mỗi seed kèm theo Norm
  cha, các Action liên quan (sửa đổi/bị sửa đổi) lấy từ graph_expand().

Cả 2 đều ưu tiên văn bản "Còn hiệu lực" lên đầu (sort trước khi ghép).
"""
from __future__ import annotations

from config import MAX_CHARS_PER_UNIT


def build_context_flat(seeds: list[dict], max_chars_per_unit: int = MAX_CHARS_PER_UNIT) -> str:
    """Ghép context từ danh sách seed phẳng (vector/bm25/hybrid) — không có
    thông tin graph (Action, Norm cha nhiều cấp)."""
    seeds = sorted(seeds, key=lambda r: r.get("validity_status") != "Còn hiệu lực")
    parts = []
    for row in seeds:
        text = (row.get("text") or "")[:max_chars_per_unit]
        parts.append(
            f"[{row.get('norm_number','')}] {row.get('norm_title','')} "
            f"({row.get('validity_status','')})\n"
            f"{row.get('level','')} {row.get('citation','')} {row.get('title_text') or ''}\n"
            f"{text}"
        )
        parts.append("---")
    return "\n".join(parts)


def build_context_graph(subgraph: list[dict], max_chars_per_unit: int = MAX_CHARS_PER_UNIT) -> str:
    """Ghép context từ kết quả graph_expand() — mỗi item là 1 Component kèm
    Norm cha, text_units, và các Action liên quan 2 chiều.

    Khác build_context_flat: có thêm dòng "Tác động ra ngoài" / "Bị tác động"
    khi Component này từng sửa đổi hoặc bị sửa đổi bởi văn bản khác — đây
    chính là phần "Graph" giúp GraphRAG trả lời được câu hỏi kiểu "quy định
    này còn hiệu lực không, bị thay bởi văn bản nào".

    item["n"] có thể là None (component không tìm được Norm tổ tiên trong 7
    tầng — xem graph_expand.py) — vẫn giữ Component trong context (đã có
    log cảnh báo ở expand_graph), chỉ bỏ phần header Norm/validity_status
    thay vì loại bỏ hẳn cả đoạn văn bản liên quan.
    """
    subgraph = sorted(
        subgraph,
        key=lambda item: (item["n"] or {}).get("validity_status") != "Còn hiệu lực",
    )
    context = []
    for item in subgraph:
        norm, component = item.get("n") or {}, item["c"]
        header = f"[{norm.get('norm_number','')}] {norm.get('title','')}".strip()
        if not norm:
            header = "[Không xác định được văn bản gốc]"
        elif norm.get("validity_status"):
            header += f" ({norm['validity_status']})"
        header += f"\n{component.get('level','')} {component.get('citation','')} {component.get('title_text') or ''}".rstrip()
        context.append(header)

        for tu in item["text_units"]:
            if tu and tu.get("accumulated_text"):
                text = tu["accumulated_text"]
                if len(text) > max_chars_per_unit:
                    text = text[:max_chars_per_unit] + "..."
                context.append(text)

        actions_from = [a for a in item["actions_from_this"] if a]
        if actions_from:
            context.append("Tác động ra ngoài: " + "; ".join(
                f"{a.get('relation_type')}→{a.get('amending_doc_number')}" for a in actions_from))

        actions_to = [p for p in item["actions_applied_to_this"] if p.get("action")]
        if actions_to:
            parts = []
            for p in actions_to:
                a, src = p["action"], p.get("source_comp")
                s = f"{a.get('relation_type')} bởi {a.get('amending_doc_number')}"
                if src:
                    s += f" ({src.get('citation')})"
                parts.append(s)
            context.append("Bị tác động: " + "; ".join(parts))

        context.append("---")
    return "\n".join(context)