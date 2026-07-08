"""Mở rộng seed Component (từ vector/BM25 search) qua graph traversal —
lấy thêm Norm cha, TextUnit nội dung, và các Action (sửa đổi/bị sửa đổi)
liên quan. Đây là phần "Graph" trong GraphRAG, phân biệt với retrieval
phẳng (flat) chỉ dựa vào text_units độc lập.

Lưu ý: Neo4jClient.run_read() dùng session.run(...).data(), tự động
convert Node/Relationship/Map lồng nhau thành dict thuần Python — nên các
hàm downstream (context_builder.build_context_graph) truy cập bằng
item["c"], item["n"], item["text_units"] như dict bình thường, không cần
import kiểu neo4j.graph.Node.

Cypher dùng CALL (c) { ... } subquery riêng cho từng nhánh (text_units,
actions_from_this, actions_applied_to_this) thay vì OPTIONAL MATCH nối
tiếp + collect(DISTINCT ...) — 3 nhánh này độc lập với nhau, nối tiếp sẽ
bị Neo4j nhân chéo (cartesian) trước khi collect gộp lại (vd 5 text_units
x 3 action_from x 2 action_to = 30 dòng trung gian cho 1 Component). Tách
subquery tránh nhân chéo, không cần DISTINCT nữa vì không còn trùng giả.

*1..7 thay vì *1.. (không trần): cây Component tối đa 7 tầng theo
ComponentLevel (Phan..Diem) — chặn trần giúp query planner ước lượng chi
phí path chính xác hơn, tránh traversal vô hạn nếu data ingest có lỗi tạo
vòng lặp CONTAINS ngoài ý muốn.

QUAN TRỌNG: (n:Norm)-[:CONTAINS*1..7]->(c) dùng OPTIONAL MATCH (không phải
MATCH bắt buộc) — nếu là MATCH thường, bất kỳ Component nào không tìm được
Norm tổ tiên trong 7 tầng (data lỗi, chuỗi CONTAINS đứt, cây sâu hơn 7 cấp)
sẽ bị loại khỏi kết quả HOÀN TOÀN mà không có cảnh báo gì — seed đã được
vector/BM25 chọn nhưng biến mất âm thầm khỏi context graphrag, trong khi
seeds gốc vẫn được ghi nhận đủ ở retrieved (dùng tính recall/MRR benchmark),
gây lệch pha khó debug giữa "đã retrieve" và "thực sự có trong context".
"""
from __future__ import annotations

import logging

from load.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

_EXPAND_CYPHER = """
MATCH (c:Component) WHERE c.comp_id IN $ids
OPTIONAL MATCH (n:Norm)-[:CONTAINS*1..7]->(c)
CALL (c) {
  OPTIONAL MATCH (c)-[:HAS_TEXTUNIT]->(tu:TextUnit {type: 'noi_dung'})
  RETURN collect(tu) AS text_units
}
CALL (c) {
  OPTIONAL MATCH (c)-[:HAS_ACTION]->(action_from:Action)
  RETURN collect(action_from) AS actions_from_this
}
CALL (c) {
  OPTIONAL MATCH (action_to:Action)-[:APPLY_TO]->(c)
  OPTIONAL MATCH (source_comp:Component)-[:HAS_ACTION]->(action_to)
  RETURN collect({action: action_to, source_comp: source_comp}) AS actions_applied_to_this
}
RETURN c, n, text_units, actions_from_this, actions_applied_to_this
"""


def expand_graph(seed_ids: list[str], client: Neo4jClient | None = None) -> list[dict]:
    """Từ danh sách comp_id (kết quả seed của vector/BM25/fusion), lấy về
    Norm chứa nó, các TextUnit nội dung, và Action liên quan 2 chiều
    (component này sửa đổi cái gì / bị cái gì sửa đổi).

    Trả list[dict] với keys: c, n, text_units, actions_from_this,
    actions_applied_to_this — dùng trực tiếp cho context_builder.build_context_graph().
    `n` có thể là None nếu Component không tìm được Norm tổ tiên trong 7
    tầng (xem docstring module) — build_context_graph() phải xử lý case này.

    Nếu comp_id nào trong seed_ids hoàn toàn không xuất hiện trong kết quả
    trả về (vd comp_id không tồn tại trong DB), hoặc có xuất hiện nhưng
    n=None, log cảnh báo để dễ debug thay vì âm thầm mất dữ liệu.
    """
    if not seed_ids:
        return []
    owns_client = client is None
    client = client or Neo4jClient()
    try:
        rows = client.run_read(_EXPAND_CYPHER, ids=seed_ids)
    finally:
        if owns_client:
            client.close()

    returned_ids = {row["c"]["comp_id"] for row in rows if row.get("c")}
    missing_ids = set(seed_ids) - returned_ids
    if missing_ids:
        logger.warning(
            "expand_graph: %d/%d seed comp_id KHÔNG tồn tại trong DB (không match nổi "
            "MATCH (c:Component)): %s", len(missing_ids), len(seed_ids), missing_ids,
        )

    no_norm_ids = {row["c"]["comp_id"] for row in rows if row.get("c") and not row.get("n")}
    if no_norm_ids:
        logger.warning(
            "expand_graph: %d component KHÔNG tìm được Norm tổ tiên trong 7 tầng "
            "(chuỗi CONTAINS đứt hoặc cây sâu hơn 7 cấp), vẫn giữ trong context nhưng "
            "thiếu header Norm: %s", len(no_norm_ids), no_norm_ids,
        )

    return rows