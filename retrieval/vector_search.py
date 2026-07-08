"""Vector search trên Neo4j vector index — tìm TextUnit gần nghĩa nhất với câu hỏi.

Khác với eval/retriever.py (bản rút gọn, chỉ phục vụ 1 test), hàm ở đây trả
đủ metadata (comp_id, level, citation, validity_status...) để dùng cho
graph_expand.py và context_builder.py ở bước sau trong pipeline retrieval đầy đủ.
"""
from __future__ import annotations

from load.neo4j_client import Neo4jClient

_VECTOR_INDEX_NAME = "textunit_embedding_index"

# *1..7 thay vì *1.. (không trần) — đồng bộ với graph_expand.py: chặn trần
# giúp query planner ước lượng chi phí path chính xác hơn, tránh traversal
# không giới hạn nếu data ingest lỡ tạo vòng lặp CONTAINS ngoài ý muốn.
_CYPHER = f"""
CALL db.index.vector.queryNodes('{_VECTOR_INDEX_NAME}', $top_k, $embedding)
YIELD node AS tu, score
WHERE tu.type = 'noi_dung'
MATCH (c:Component)-[:HAS_TEXTUNIT]->(tu)
MATCH (n:Norm)-[:CONTAINS*1..7]->(c)
RETURN
  tu.unit_id AS textunit_id, score, tu.accumulated_text AS text,
  c.comp_id AS comp_id, c.level AS level, c.citation AS citation,
  c.title_text AS title_text, n.norm_id AS norm_id, n.title AS norm_title,
  n.norm_number AS norm_number, n.validity_status AS validity_status
ORDER BY score DESC
"""


def vector_search(question_embedding: list[float], top_k: int = 10, client: Neo4jClient | None = None) -> list[dict]:
    """Trả về top_k TextUnit (type='noi_dung') gần nhất với embedding câu hỏi.

    Nếu không truyền `client`, tự mở/đóng 1 connection tạm — tiện cho dùng
    lẻ (script/test). Trong pipeline.py LUÔN truyền client dùng chung (mở 1
    lần ở run_pipeline) để tránh mở/đóng driver mới cho mỗi câu hỏi — chi
    phí handshake tới Aura (remote, TLS) có thể tốn hàng trăm ms mỗi lần.
    """
    owns_client = client is None
    client = client or Neo4jClient()
    try:
        return client.run_read(_CYPHER, embedding=question_embedding, top_k=top_k)
    finally:
        if owns_client:
            client.close()