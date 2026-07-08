"""BM25 index (Whoosh) trên toàn bộ TextUnit — bổ trợ cho vector_search khi
câu hỏi dùng từ khoá/số hiệu văn bản chính xác mà vector search có thể bỏ lỡ.

Khác notebook gốc: WHOOSH_DIR đổi từ path Colab (/home/claude/whoosh_index)
sang config.BM25_DIR (./data/bm25_index/) cho khớp convention data/raw|
transformed|embedded đã có trong repo.

Cách dùng:
    from retrieval.bm25_index import build_index, bm25_search
    build_index()                       # chạy 1 lần (hoặc khi data đổi)
    bm25_search("bảo hiểm xã hội", top_k=5)
"""
from __future__ import annotations

import logging
import re
import shutil

from config import BM25_DIR, TOP_K
from load.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

# *1..7 — đồng bộ với graph_expand.py/vector_search.py, tránh traversal
# không giới hạn khi build index (chạy 1 lần nhưng vẫn nên nhất quán).
_FETCH_CYPHER = """
MATCH (c:Component)-[:HAS_TEXTUNIT]->(tu:TextUnit {type:'noi_dung'})
MATCH (n:Norm)-[:CONTAINS*1..7]->(c)
RETURN
  tu.unit_id AS textunit_id, tu.accumulated_text AS text,
  c.comp_id AS comp_id, c.level AS level, c.citation AS citation,
  c.title_text AS title_text, n.norm_id AS norm_id, n.title AS norm_title,
  n.norm_number AS norm_number, n.validity_status AS validity_status
"""

# Ký tự đặc biệt trong cú pháp query Whoosh (?, *, :, ^, ~, ngoặc...) — loại bỏ
# khỏi câu hỏi trước khi parse, tránh bị hiểu nhầm thành wildcard/toán tử.
_SPECIAL_CHARS_RE = re.compile(r'[?!.,;:*^~\[\]{}()"]')

_ix = None  # cache index đã mở, tránh open_dir lại mỗi lần search


def fetch_all_textunits(client: Neo4jClient | None = None) -> list[dict]:
    """Kéo toàn bộ TextUnit type='noi_dung' + metadata từ Neo4j để đánh index."""
    owns_client = client is None
    client = client or Neo4jClient()
    try:
        rows = client.run_read(_FETCH_CYPHER, timeout=60.0)  # query lớn, timeout dài hơn
        logger.info("Đã lấy %d TextUnit để đánh index BM25.", len(rows))
        return rows
    finally:
        if owns_client:
            client.close()


def build_index(rows: list[dict] | None = None) -> None:
    """Build lại toàn bộ Whoosh index từ đầu — XOÁ index cũ ở BM25_DIR nếu có.

    Chạy sau mỗi lần load/ ghi dữ liệu mới vào Neo4j (embed/load pipeline),
    tương tự cách embed index vector cần rebuild khi data đổi.
    """
    from whoosh.analysis import StandardAnalyzer
    from whoosh.fields import ID, STORED, TEXT, Schema
    from whoosh.index import create_in

    rows = rows if rows is not None else fetch_all_textunits()

    if BM25_DIR.exists():
        shutil.rmtree(BM25_DIR)
    BM25_DIR.mkdir(parents=True, exist_ok=True)

    schema = Schema(
        textunit_id=ID(stored=True, unique=True),
        text=TEXT(stored=True, analyzer=StandardAnalyzer()),
        comp_id=STORED, level=STORED, citation=STORED, title_text=STORED,
        norm_id=STORED, norm_title=STORED, norm_number=STORED, validity_status=STORED,
    )
    ix = create_in(str(BM25_DIR), schema)
    writer = ix.writer(limitmb=256, procs=1)
    skipped = 0
    for row in rows:
        if not row.get("text"):
            skipped += 1
            continue
        writer.add_document(
            textunit_id=str(row.get("textunit_id")),
            text=row.get("text") or "",
            comp_id=row.get("comp_id"),
            level=row.get("level"),
            citation=row.get("citation"),
            title_text=row.get("title_text"),
            norm_id=row.get("norm_id"),
            norm_title=row.get("norm_title"),
            norm_number=row.get("norm_number"),
            validity_status=row.get("validity_status"),
        )
    writer.commit()
    logger.info("Đã build xong Whoosh BM25 index (%d doc, %d bị bỏ qua vì text rỗng).", len(rows) - skipped, skipped)


def _get_index():
    global _ix
    if _ix is None:
        from whoosh.index import open_dir
        if not BM25_DIR.exists() or not any(BM25_DIR.iterdir()):
            raise FileNotFoundError(
                f"Chưa có Whoosh index tại {BM25_DIR} — chạy build_index() trước."
            )
        _ix = open_dir(str(BM25_DIR))
    return _ix


def bm25_search(question: str, top_k: int = TOP_K) -> list[dict]:
    """Tìm kiếm BM25F trên index đã build. Group=OR giữa các từ trong câu hỏi
    (đúng chuẩn BM25 — chấm điểm theo độ trùng khớp, thay vì AND mặc định của
    Whoosh khiến câu hỏi dài gần như không bao giờ match được document nào)."""
    from whoosh import scoring
    from whoosh.qparser import OrGroup, QueryParser

    ix = _get_index()
    cleaned = _SPECIAL_CHARS_RE.sub(" ", question)

    with ix.searcher(weighting=scoring.BM25F()) as searcher:
        qp = QueryParser("text", schema=ix.schema, group=OrGroup)
        q = qp.parse(cleaned)
        hits = searcher.search(q, limit=top_k)
        results = []
        for h in hits:
            d = dict(h)
            d["score"] = h.score
            results.append(d)
        return results