# AI Legal Retrieval Benchmark

Hệ thống RAG (Retrieval-Augmented Generation) trả lời câu hỏi pháp luật Việt Nam, xây dựng trên nền **Legal Knowledge Graph** (Neo4j) và so sánh 5 chiến lược retrieval: `bm25`, `vector`, `hybrid`, `graphrag`, `vector_graph`.

Tài liệu chi tiết về pipeline `retrieval/` và `benchmark/` (input/output, ý nghĩa từng hàm): xem [`RETRIEVAL_BENCHMARK.md`](./RETRIEVAL_BENCHMARK.md).

---

## 1. Kiến trúc tổng quan

```
data/ (raw jsonl)
      │
      ▼
schema/  (định nghĩa Node/Edge: Norm, Component, TextUnit, Action, NormRelation...)
      │
      ▼
load/    (nạp dữ liệu vào Neo4j AuraDB theo đúng thứ tự phụ thuộc)
      │
      ▼
retrieval/  (embed câu hỏi → vector/BM25/hybrid/graphrag search → build context → RAG answer)
      │
      ▼
benchmark/  (chạy retrieval/pipeline.py trên eval set → chấm điểm bằng DeepEval/LLM-judge → CSV)
```

Legal Knowledge Graph gồm các loại node chính:

- **Norm**: văn bản pháp luật (Luật, Nghị định, Thông tư...).
- **Component**: đơn vị cấu trúc bên trong văn bản (Phần/Chương/Mục/Tiểu mục/Điều/Khoản/Điểm), quan hệ `CONTAINS` lồng nhau.
- **TextUnit**: nội dung văn bản gắn với Component (`type='noi_dung'`), được embed để phục vụ vector search.
- **Action**: sự kiện sửa đổi/bãi bỏ/thay thế giữa các Component (quan hệ `HAS_ACTION`, `APPLY_TO`).

## 2. Cấu trúc thư mục

```
ai-legal-retrieval-benchmark/
├── config.py                # đọc .env, hằng số dùng chung (TOP_K, MAX_COMPONENTS, LEVEL_RANK...)
├── requirements.txt
├── schema/                  # Pydantic model cho Node (nodes.py), Edge (edges.py), Enum (enums.py)
├── load/                    # nạp dữ liệu vào Neo4j
│   ├── neo4j_client.py      #   wrapper driver Neo4j (run_read/run_write)
│   ├── loaders.py           #   load_norms/load_components/load_component_textunits/
│   │                        #   load_actions/load_action_edges/load_relations
│   └── schema_init.cypher   #   tạo constraint/index (bao gồm vector index) trong Neo4j
├── retrieval/                # xem chi tiết trong RETRIEVAL_BENCHMARK.md
│   ├── embedder.py, keyword_extractor.py
│   ├── vector_search.py, bm25_index.py, fusion.py, reranker.py
│   ├── graph_expand.py, context_builder.py, citation_formatter.py, prompts.py
│   ├── retry_utils.py        # retry tự động khi LLM trả lỗi 429 (quota)
│   ├── pipeline.py           # retrieve() + run_pipeline() — entrypoint chính
│   └── pipeline_debbug.py    # bản pipeline.py có thêm log [TIMING], chỉ dùng để debug latency
├── benchmark/                 # xem chi tiết trong RETRIEVAL_BENCHMARK.md
│   ├── eval_set.py, deepeval_judge.py, metrics.py
│   ├── run_benchmark.py       # entrypoint: python -m benchmark.run_benchmark
│   ├── retry_failed.py        # chạy lại benchmark chỉ cho các dòng bị lỗi trong CSV đã có
│   └── results/                # CSV kết quả benchmark đã chạy
├── data/                      # dữ liệu thô/eval set (jsonl)
└── RETRIEVAL_BENCHMARK.md     # tài liệu chi tiết pipeline retrieval + benchmark
```

## 3. Cài đặt

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tạo file `.env` ở thư mục gốc với các biến (xem `config.py` để biết đầy đủ + giá trị mặc định):

```env
# Neo4j AuraDB
NEO4J_URI=neo4j+s://xxxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=xxxx

# Google Cloud / Vertex AI
GCP_PROJECT=your-gcp-project
GCP_LOCATION=us-central1
EMBEDDING_MODEL=gemini-embedding-001
LLM_MODEL_HEAVY=gemini-3.5-flash
LLM_MODEL_LIGHT=gemini-3.1-flash-lite

# Retrieval
TOP_K=15
MAX_COMPONENTS=5
MAX_CHARS_PER_UNIT=1000
MAX_PROMPT_TOKENS=6000
USE_RERANK=true
RERANK_MIN_SCORE=0.0
```

## 4. Quy trình sử dụng

### 4.1 Nạp dữ liệu vào Neo4j (chạy 1 lần / khi có dữ liệu mới)

Thứ tự bắt buộc (theo phụ thuộc node/edge — xem docstring `load/loaders.py`):

```python
from load.neo4j_client import Neo4jClient
from load.loaders import (
    load_norms, load_components, load_component_textunits,
    load_actions, load_action_edges, load_relations,
)

client = Neo4jClient()
load_norms(client, norms)
load_components(client, components)
load_component_textunits(client, textunits)
load_actions(client, actions)
load_action_edges(client, action_edges)
load_relations(client, relations)
```

Chạy `load/schema_init.cypher` trước để tạo constraint + vector index (`textunit_embedding_index`) trên Neo4j.

### 4.2 Build BM25 index (chạy 1 lần / khi dữ liệu Neo4j thay đổi)

```python
from retrieval.bm25_index import build_index
build_index()
```

### 4.3 Truy vấn thử (retrieval/pipeline.py)

```python
from retrieval.pipeline import run_pipeline

result = run_pipeline("Người lao động nghỉ thai sản được hưởng chế độ gì?", mode="graphrag")
print(result["answer"])
print(result["retrieved_citations"])
```

Mode `vector_graph` giống `vector` ở bước lấy seed (chỉ vector search), nhưng build context có mở rộng qua graph giống `graphrag` — dùng để tách riêng tác dụng của "mở rộng qua graph" khỏi tác dụng của "hợp nhất BM25+vector" khi so sánh kết quả benchmark.

### 4.4 Chạy benchmark so sánh 5 mode

```bash
python -m benchmark.run_benchmark \
  --eval-set data/questions.jsonl \
  --modes bm25 vector hybrid graphrag vector_graph \
  --max-workers 4
```

Kết quả:

- `benchmark/results/benchmark_results.csv` — chi tiết từng (câu hỏi, mode).
- `benchmark/results/benchmark_summary.csv` — trung bình theo mode (`answer_relevancy`, `faithfulness_deepeval`, `contextual_relevancy`, `context_len_chars`, `prompt_tokens`, `latency_sec`).

Xem thêm cờ `--include-reason` để DeepEval sinh giải thích cho từng metric (tốn thêm 1 lần gọi LLM/metric/câu hỏi), và `--precomputed mode=path.csv` để chấm lại điểm DeepEval trên kết quả `run_pipeline()` đã chạy sẵn thay vì gọi lại retrieval (xem `benchmark/retry_failed.py` để chỉ chạy lại các dòng bị lỗi).

> Lưu ý: `_run_one_question()` trong `run_benchmark.py` hiện gọi `run_pipeline(..., max_components=5, top_k=5, use_rerank=False)` cố định — các giá trị `TOP_K`/`MAX_COMPONENTS`/`USE_RERANK` trong `.env` **không** áp dụng khi chạy qua `benchmark.run_benchmark`, chỉ áp dụng khi gọi `run_pipeline()`/`retrieve()` trực tiếp.

## 5. Tài liệu liên quan

- [`RETRIEVAL_BENCHMARK.md`](./RETRIEVAL_BENCHMARK.md) — mô tả chi tiết pipeline retrieval (4 mode, input/output/ý nghĩa từng hàm) và pipeline benchmark (DeepEval + metric cần ground truth).
