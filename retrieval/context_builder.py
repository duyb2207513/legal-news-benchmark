# """Ghép seed results (sau fusion/graph_expand) thành 1 chuỗi context để đưa
# vào RAG_PROMPT. 2 chế độ tương ứng 2 nhóm mode trong retrieval/pipeline.py:

# - build_context_flat: dùng cho mode "vector"/"bm25"/"hybrid" — mỗi seed là
#   1 đoạn text độc lập, không có quan hệ graph.
# - build_context_graph: dùng cho mode "graphrag" — mỗi seed kèm theo Norm
#   cha, các Action liên quan (sửa đổi/bị sửa đổi) lấy từ graph_expand().

# Không sort theo validity_status nữa — giữ nguyên thứ tự trả về từ
# retrieve()/graph_expand() (đã sort theo rerank/score ở đó). Việc phân biệt
# còn/hết hiệu lực để LLM tự đọc trong header (validity_status vẫn được in
# ra) và xử lý theo RAG_PROMPT, không ép cứng bằng thứ tự nữa.
# """
# from __future__ import annotations

# from config import MAX_CHARS_PER_UNIT


# # Định nghĩa thứ tự ưu tiên — số nhỏ hơn = ưu tiên cao hơn
# _VALIDITY_RANK = {
#     "Còn hiệu lực": 0,
#     "Hết hiệu lực một phần": 1,
# }
# _DEFAULT_VALIDITY_RANK = 2  # "Hết hiệu lực", "Không xác định", None, hoặc giá trị lạ khác


# def _validity_sort_key(status: str | None) -> int:
#     return _VALIDITY_RANK.get(status, _DEFAULT_VALIDITY_RANK)


# def build_context_flat(seeds: list[dict], max_chars_per_unit: int = MAX_CHARS_PER_UNIT) -> str:
#     parts = []
#     for row in seeds:
#         text = (row.get("text") or "")[:max_chars_per_unit]
#         parts.append(
#             f"[{row.get('norm_number','')}] {row.get('norm_title','')} "
#             f"({row.get('validity_status','')})\n"
#             f"{row.get('level','')} {row.get('citation','')} {row.get('title_text') or ''}\n"
#             f"{text}"
#         )
#         parts.append("---")
#     return "\n".join(parts)


# def build_context_graph(subgraph: list[dict], max_chars_per_unit: int = MAX_CHARS_PER_UNIT) -> str:
#     """Ghép context từ kết quả graph_expand() — mỗi item là 1 Component kèm
#     Norm cha, text_units, và các Action liên quan 2 chiều.

#     Khác build_context_flat: có thêm dòng "Tác động ra ngoài" / "Bị tác động"
#     khi Component này từng sửa đổi hoặc bị sửa đổi bởi văn bản khác — đây
#     chính là phần "Graph" giúp GraphRAG trả lời được câu hỏi kiểu "quy định
#     này còn hiệu lực không, bị thay bởi văn bản nào".

#     item["n"] có thể là None (component không tìm được Norm tổ tiên trong 7
#     tầng — xem graph_expand.py) — vẫn giữ Component trong context (đã có
#     log cảnh báo ở expand_graph), chỉ bỏ phần header Norm/validity_status
#     thay vì loại bỏ hẳn cả đoạn văn bản liên quan.
#     """
#     context = []
#     for item in subgraph:
#         norm, component = item.get("n") or {}, item["c"]
#         header = f"[{norm.get('norm_number','')}] {norm.get('title','')}".strip()
#         if not norm:
#             header = "[Không xác định được văn bản gốc]"
#         elif norm.get("validity_status"):
#             header += f" ({norm['validity_status']})"
#         header += f"\n{component.get('level','')} {component.get('citation','')} {component.get('title_text') or ''}".rstrip()
#         context.append(header)

#         for tu in item["text_units"]:
#             if tu and tu.get("accumulated_text"):
#                 text = tu["accumulated_text"]
#                 if len(text) > max_chars_per_unit:
#                     text = text[:max_chars_per_unit] + "..."
#                 context.append(text)

#         actions_from = [a for a in item["actions_from_this"] if a]
#         if actions_from:
#             context.append("Tác động ra ngoài: " + "; ".join(
#                 f"{a.get('relation_type')}→{a.get('amending_doc_number')}" for a in actions_from))

#         actions_to = [p for p in item["actions_applied_to_this"] if p.get("action")]
#         if actions_to:
#             parts = []
#             for p in actions_to:
#                 a, src = p["action"], p.get("source_comp")
#                 s = f"{a.get('relation_type')} bởi {a.get('amending_doc_number')}"
#                 if src:
#                     s += f" ({src.get('citation')})"
#                 parts.append(s)
#             context.append("Bị tác động: " + "; ".join(parts))

#         context.append("---")
#     return "\n".join(context)
"""Ghép seed results (sau fusion/graph_expand) thành 1 chuỗi context để đưa
vào RAG_PROMPT. 2 chế độ tương ứng 2 nhóm mode trong retrieval/pipeline.py:

- build_context_flat: dùng cho mode "vector"/"bm25"/"hybrid" — mỗi seed là
  1 đoạn text độc lập, không có quan hệ graph.
- build_context_graph: dùng cho mode "graphrag" — mỗi seed kèm theo Norm
  cha, các Action liên quan (sửa đổi/bị sửa đổi) lấy từ graph_expand().

Không sort theo validity_status nữa — giữ nguyên thứ tự trả về từ
retrieve()/graph_expand() (đã sort theo rerank/score ở đó). Việc phân biệt
còn/hết hiệu lực để LLM tự đọc trong header (validity_status vẫn được in
ra) và xử lý theo RAG_PROMPT, không ép cứng bằng thứ tự nữa.

Format text dùng chung retrieval/citation_formatter.py::format_citations()
(kiểu "Trích dẫn N") thay vì tự build string thủ công.
"""
from __future__ import annotations

from config import MAX_CHARS_PER_UNIT
from retrieval.citation_formatter import format_citations


# Định nghĩa thứ tự ưu tiên — số nhỏ hơn = ưu tiên cao hơn
_VALIDITY_RANK = {
    "Còn hiệu lực": 0,
    "Hết hiệu lực một phần": 1,
}
_DEFAULT_VALIDITY_RANK = 2  # "Hết hiệu lực", "Không xác định", None, hoặc giá trị lạ khác


def _validity_sort_key(status: str | None) -> int:
    return _VALIDITY_RANK.get(status, _DEFAULT_VALIDITY_RANK)


def build_context_flat(seeds: list[dict], max_chars_per_unit: int = MAX_CHARS_PER_UNIT) -> str:
    citations = [
        {
            "norm_title": row.get("norm_title", ""),
            "norm_number": row.get("norm_number", ""),
            "validity_status": row.get("validity_status", ""),
            "citation": row.get("citation", "") or row.get("level", ""),
            "text": (row.get("text") or "")[:max_chars_per_unit],
        }
        for row in seeds
    ]
    return format_citations(citations)


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

    citation_formatter.format_citations() không có field riêng cho "Tác
    động ra ngoài"/"Bị tác động" — nối trực tiếp vào cuối "text" trước khi
    đưa vào format_citations() để không phải sửa citation_formatter.py.
    """
    citations = []
    for item in subgraph:
        norm, component = item.get("n") or {}, item["c"]

        if not norm:
            norm_title, norm_number, validity_status = "Không xác định được văn bản gốc", "", ""
        else:
            norm_title, norm_number = norm.get("title", ""), norm.get("norm_number", "")
            validity_status = norm.get("validity_status", "")

        text_parts = []
        for tu in item["text_units"]:
            if tu and tu.get("accumulated_text"):
                text = tu["accumulated_text"]
                if len(text) > max_chars_per_unit:
                    text = text[:max_chars_per_unit] + "..."
                text_parts.append(text)
        text = " ".join(text_parts)

        actions_from = [a for a in item["actions_from_this"] if a]
        if actions_from:
            text += "\nTác động ra ngoài: " + "; ".join(
                f"{a.get('relation_type')}→{a.get('amending_doc_number')}" for a in actions_from)

        actions_to = [p for p in item["actions_applied_to_this"] if p.get("action")]
        if actions_to:
            parts = []
            for p in actions_to:
                a, src = p["action"], p.get("source_comp")
                s = f"{a.get('relation_type')} bởi {a.get('amending_doc_number')}"
                if src:
                    s += f" ({src.get('citation')})"
                parts.append(s)
            text += "\nBị tác động: " + "; ".join(parts)

        citations.append({
            "norm_title": norm_title,
            "norm_number": norm_number,
            "validity_status": validity_status,
            "citation": component.get("citation", "") or component.get("level", ""),
            "text": text,
        })

    return format_citations(citations)