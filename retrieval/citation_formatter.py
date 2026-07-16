"""Format text context kiểu "Trích dẫn N" — tách riêng khỏi retrieval/
context_builder.py để dùng lại độc lập (vd hiển thị lại context đã lưu
trong file JSON/CSV từ benchmark, không cần chạy lại pipeline/Neo4j).

Dùng chung format với retrieval/context_builder.py::build_context_flat(),
nhưng nhận input là list dict "sạch" (không phụ thuộc field name của
Neo4j row) — chỉ cần các key: norm_title, norm_number, validity_status,
citation, text. published_date là optional.
"""
from __future__ import annotations


def format_citations(citations: list[dict]) -> str:
    """Build text context từ list citation dict, theo format:

    [NGỮ CẢNH PHÁP LÝ ĐƯỢC CUNG CẤP]

    --- Trích dẫn 1 ---
    - Văn bản: <norm_title> (Số: <norm_number>)
    - Trạng thái: <validity_status>
    - Điều/Khoản: <citation>
    - Nội dung: "<text>"

    Mỗi dict trong `citations` cần các key: norm_title, norm_number,
    validity_status, citation, text. Key `published_date` là optional —
    nếu có, thêm " (ban hành <published_date>)" ngay sau (Số: ...).
    """
    parts = ["[NGỮ CẢNH PHÁP LÝ ĐƯỢC CUNG CẤP]"]
    for i, c in enumerate(citations, start=1):
        ban_hanh = f" (ban hành {c['published_date']})" if c.get("published_date") else ""
        parts.append(
            f"\n--- Trích dẫn {i} ---\n"
            f"- Văn bản: {c.get('norm_title', '')} (Số: {c.get('norm_number', '')}){ban_hanh}\n"
            f"- Trạng thái: {c.get('validity_status', '')}\n"
            f"- Điều/Khoản: {c.get('citation', '')}\n"
            f'- Nội dung: "{c.get("text", "")}"'
        )
    return "\n".join(parts)