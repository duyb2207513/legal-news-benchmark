# Retrieval & Benchmark Pipeline

Tài liệu mô tả chi tiết 2 module chính trong dự án: `retrieval/` (truy xuất + sinh câu trả lời) và `benchmark/` (đo chất lượng retrieval).

---

## 1. Module `retrieval/`

### 1.1 Sơ đồ pipeline tổng quát

```
question
   │
   ├─► embed_question()          (Vertex AI embedding, task=RETRIEVAL_QUERY)
   ├─► extract_legal_keywords()  (LLM trích từ khoá pháp lý — chỉ cho hybrid/graphrag)
   │
   ▼
retrieve(question, mode)
   │  mode="vector"       → vector_search()
   │  mode="bm25"         → bm25_search()
   │  mode="hybrid"       → vector_search() + bm25_search() → merge_search_results() (RRF)
   │  mode="graphrag"     → giống hybrid, seeds dùng để expand_graph() ở bước sau
   │  mode="vector_graph" → giống vector, seeds dùng để expand_graph() ở bước sau
   │
   ▼ (tuỳ chọn) rerank() — lọc theo coverage từ khoá (có trọng số IDF), sort lại
   ▼ sort ưu tiên rerank_score (nếu có rerank) hoặc score, cắt còn max_components
   │
   ▼
seeds (list[dict] Component)
   │
   ├─ mode="graphrag"/"vector_graph": expand_graph(seeds) → build_context_graph()
   └─ mode khác:                                            build_context_flat()
   │
   ▼
context (string, format "Trích dẫn N" — xem citation_formatter.py)
   │
   ▼ cắt bớt nếu vượt MAX_PROMPT_TOKENS
   ▼
answer(question, context)  → RAG_PROMPT → LLM_MODEL_HEAVY
   │
   ▼
{answer, context, retrieved, prompt_tokens, latency_sec, retrieved_citations}
```

Có **5 mode retrieval**:

| Mode           | Nguồn seed                                                                  | Context builder |
| -------------- | --------------------------------------------------------------------------- | --------------- |
| `vector`       | chỉ vector search (embedding similarity)                                    | flat            |
| `bm25`         | chỉ BM25 (từ khoá chính xác)                                                | flat            |
| `hybrid`       | vector + BM25 hợp nhất bằng RRF                                             | flat            |
| `graphrag`     | giống hybrid, nhưng mở rộng qua graph (Norm cha, Action sửa đổi/bị sửa đổi) | graph           |
| `vector_graph` | giống `vector` (chỉ vector search), nhưng mở rộng qua graph như `graphrag`  | graph           |

`vector_graph` cho phép tách riêng tác dụng của "mở rộng qua graph" khỏi tác dụng của "hợp nhất BM25+vector qua RRF" khi so sánh 5 mode trong benchmark.

### 1.2 Chi tiết từng file

#### `embedder.py` — Embed câu hỏi

- **Hàm:** `embed_question(question, task_type="RETRIEVAL_QUERY") -> list[float]`
- **Input:** câu hỏi dạng text.
- **Output:** vector embedding (list[float]) từ model `EMBEDDING_MODEL` (Vertex AI Gemini embedding).
- **Ý nghĩa:** Tách riêng khỏi việc embed hàng loạt lúc build index (task_type=`RETRIEVAL_DOCUMENT`) vì embed 1 câu hỏi tại thời điểm query dùng task_type khác (`RETRIEVAL_QUERY`) để tối ưu độ khớp truy vấn ↔ tài liệu. Nếu `GCP_PROJECT` chưa cấu hình hoặc lỗi API, ném exception để tầng gọi (`pipeline.py`) tự quyết định fallback.

#### `keyword_extractor.py` — Trích từ khoá pháp lý

- **Hàm:** `extract_legal_keywords(question, llm=None) -> str`
- **Input:** câu hỏi tự nhiên (có thể lẫn chi tiết cá nhân: số tiền, tên công ty...).
- **Output:** chuỗi 3–6 từ khoá pháp lý (tên luật, loại văn bản, khái niệm), cách nhau bởi dấu cách.
- **Ý nghĩa:** BM25 match theo từ đúng nghĩa đen, nên nếu đưa nguyên câu hỏi tự nhiên vào thì dễ bị nhiễu bởi các từ không mang tính pháp lý. Hàm này gọi LLM (`LLM_MODEL_HEAVY`) để lọc ra phần "lõi pháp lý" làm query cho BM25. Có fallback: nếu LLM lỗi, trả nguyên câu hỏi gốc để không chặn pipeline.

#### `vector_search.py` — Tìm kiếm theo embedding

- **Hàm:** `vector_search(question_embedding, top_k=10, client=None) -> list[dict]`
- **Input:** embedding câu hỏi, số lượng kết quả `top_k`, Neo4j client dùng chung (tuỳ chọn).
- **Output:** danh sách top-k `TextUnit` (chỉ `type='noi_dung'`) gần nghĩa nhất, mỗi phần tử gồm `textunit_id, score, text, comp_id, level, citation, title_text, norm_id, norm_title, norm_number, validity_status`.
- **Ý nghĩa:** Chạy Cypher `db.index.vector.queryNodes(...)` trên vector index Neo4j, sau đó join lên `Component` (chứa TextUnit) và `Norm` (chứa Component, tối đa 7 tầng `CONTAINS*1..7`) để trả đủ metadata cho các bước sau (graph expand, build context). Nếu không truyền `client`, tự mở/đóng connection riêng (tiện cho chạy lẻ); khi chạy trong pipeline luôn dùng chung 1 client để tránh tốn chi phí handshake TLS tới Aura mỗi lần.

#### `bm25_index.py` — Chỉ mục từ khoá (Whoosh)

- **Hàm chính:**
  - `fetch_all_textunits(client=None) -> list[dict]`: kéo toàn bộ `TextUnit(type='noi_dung')` + metadata từ Neo4j để chuẩn bị đánh index.
  - `build_index(rows=None) -> None`: xoá index Whoosh cũ (nếu có) tại `BM25_DIR`, tạo schema mới, ghi từng document vào index. Chạy 1 lần (hoặc khi dữ liệu Neo4j thay đổi).
  - `bm25_search(question, top_k=TOP_K) -> list[dict]`: tìm kiếm BM25F (group=OR giữa các từ) trên index đã build.
- **Input/Output:**
  - `build_index`: input là list dict (metadata TextUnit), output ghi ra thư mục index trên đĩa, không trả giá trị.
  - `bm25_search`: input câu hỏi/từ khoá + `top_k`; output list dict kết quả (mỗi dict có `score` + toàn bộ field STORED trong schema).
- **Ý nghĩa:** Bổ trợ cho vector search khi câu hỏi chứa từ khoá/số hiệu văn bản chính xác mà vector search dễ bỏ lỡ. Dùng `OrGroup` thay AND mặc định của Whoosh vì câu hỏi dài theo AND gần như không match được gì. Ký tự đặc biệt trong cú pháp Whoosh (`?*:^~[]{}()"`) được loại bỏ khỏi câu hỏi trước khi parse để tránh hiểu nhầm thành toán tử.

#### `fusion.py` — Hợp nhất kết quả (weighted RRF)

- **Hàm:** `merge_search_results(*result_lists, k=60, weights=(0.55, 0.45)) -> list[dict]`
- **Input:** nhiều list kết quả (mỗi list đã sort theo score giảm dần) — ví dụ `vector_search()` và `bm25_search()`; `weights` là trọng số áp cho từng nguồn theo đúng thứ tự truyền vào.
- **Output:** 1 danh sách hợp nhất, mỗi dòng có thêm `rrf_score` (cũng được gán vào `score`), `original_score` (score gốc của dòng đại diện, giữ lại để debug — không dùng để sort/so sánh), `best_rank`, đã loại trùng theo `(norm_id, citation)` (giữ dòng có `rrf_score` cao nhất).
- **Ý nghĩa:** Reciprocal Rank Fusion có trọng số — công thức `rrf(row) = Σ weight_i / (k + rank_i)` qua các nguồn chứa row đó. Dùng RANK (thứ hạng) thay vì raw score vì cosine similarity (vector) và BM25F score không cùng thang đo, so sánh trực tiếp sẽ thiên vị BM25 (score luôn lớn hơn "giả tạo"). Nguồn đầu tiên (quy ước là `vector_search()`) được nhân trọng số `weights[0]` (mặc định 0.55) và nguồn thứ hai (BM25) nhân `weights[1]` (mặc định 0.45) vì vector search thường có precision cao hơn cho câu hỏi tự nhiên; nguồn dư ngoài `len(weights)` dùng weight=1.0. Dòng đại diện cho mỗi Component chọn theo RANK tốt nhất (nhỏ nhất) trong nguồn của nó, không theo raw score. Kết quả cuối **không** tự sort theo `validity_status` trong hàm này (dòng sort đó đã bị comment trong code) — việc ưu tiên "Còn hiệu lực" được thực hiện ở bước sau, trong `retrieve()` (`pipeline.py`).
<!-- 
#### `reranker.py` — Rerank theo coverage từ khoá tuyệt đối

- **Hàm:** `rerank(question, seeds, top_n=5, min_score=4.0, llm=None) -> list[dict]`
- **Input:** câu hỏi gốc, danh sách seeds thô (sau RRF hoặc 1 nguồn đơn), `top_n` số lượng giữ lại, `min_score` ngưỡng loại (thang 0–10).
- **Output:** danh sách seed đã được gán thêm `rerank_score`, lọc bỏ dòng dưới `min_score`, sort giảm dần theo `rerank_score` (tie-break bằng BM25 nội bộ), cắt còn tối đa `top_n`. Có thể trả **danh sách rỗng** nếu toàn bộ candidate đều lạc đề.
- **Ý nghĩa:** RRF chỉ xếp theo rank giữa 2 nguồn, không đánh giá lại độ liên quan ngữ nghĩa thật. Bản cũ dùng self-referential BM25 normalization (chuẩn hoá theo max trong chính tập seeds) nên luôn đẩy 1 candidate lên ~10/10 dù nó lạc đề — không bao giờ lọc sạch được cả tập sai. Bản hiện tại dùng **coverage tuyệt đối có trọng số IDF**, không phụ thuộc các candidate khác:
  - Với mỗi từ khoá trong câu hỏi (`query_terms`, đã loại stopword tiếng Việt), tính `idf(term) = ln((n_docs + 1) / (df(term) + 1)) + 1`, trong đó `df(term)` là số candidate trong `seeds` có chứa từ đó — từ càng hiếm trong tập candidate thì trọng số càng cao.
  - `rerank_score = 10 × Σ idf(t) cho t ∈ (query_terms ∩ doc_terms) / Σ idf(t) cho t ∈ query_terms` (thang 0–10). Nếu tài liệu không chứa từ khoá nào của câu hỏi, `rerank_score = 0` và bị loại (dưới `min_score`) bất kể so với các candidate khác thế nào.
  - Raw BM25 (`rank_bm25`, coi seeds là mini-corpus) chỉ dùng làm tie-break **phụ** giữa các candidate đã cùng vượt `min_score`, không quyết định pass/fail. -->

#### `graph_expand.py` — Mở rộng seed qua đồ thị

- **Hàm:** `expand_graph(seed_ids: list[str], client=None) -> list[dict]`
- **Input:** danh sách `comp_id` (từ seeds sau retrieve/rerank).
- **Output:** list dict, mỗi phần tử gồm khoá `c` (Component), `n` (Norm cha, có thể `None`), `text_units` (list TextUnit nội dung), `actions_from_this` (Action mà Component này gây ra — sửa đổi văn bản khác), `actions_applied_to_this` (Action từ nơi khác tác động lên Component này, kèm `source_comp`).
- **Ý nghĩa:** Đây là phần "Graph" của GraphRAG — bổ sung ngữ cảnh về quan hệ pháp lý (văn bản cha, lịch sử sửa đổi 2 chiều) mà retrieval phẳng không có. Dùng 3 subquery `CALL (c) {...}` độc lập cho từng nhánh thay vì `OPTIONAL MATCH` nối tiếp để tránh nhân chéo (cartesian product) khi collect. Dùng `OPTIONAL MATCH` (không phải MATCH bắt buộc) cho quan hệ Norm–Component để tránh Component "biến mất âm thầm" khỏi context nếu chuỗi `CONTAINS` bị đứt hoặc cây sâu hơn 7 cấp; đồng thời log cảnh báo khi có `comp_id` không tồn tại hoặc không tìm được Norm tổ tiên.

#### `context_builder.py` — Ghép context cho prompt

- **Hàm:**
  - `build_context_flat(seeds, max_chars_per_unit=MAX_CHARS_PER_UNIT) -> str`: dùng cho mode `vector`/`bm25`/`hybrid`.
  - `build_context_graph(subgraph, max_chars_per_unit=MAX_CHARS_PER_UNIT) -> str`: dùng cho mode `graphrag`.
- **Input:** `build_context_flat` nhận list seed (dict); `build_context_graph` nhận output của `expand_graph()`.
- **Output:** 1 chuỗi text (context) theo format `"Trích dẫn N"` (xem `citation_formatter.py`), mỗi trích dẫn gồm Văn bản/Trạng thái/Điều-Khoản/Nội dung, để đưa vào `RAG_PROMPT`.
- **Ý nghĩa:** Cả 2 hàm build danh sách dict citation (`norm_title`, `norm_number`, `validity_status`, `citation`, `text`) rồi gọi chung `citation_formatter.format_citations()` để ra chuỗi text cuối — không tự ghép string thủ công nữa. Không sort theo `validity_status` ở bước này nữa (giữ nguyên thứ tự đã sort ở `retrieve()`/`rerank()`); LLM tự đọc trạng thái hiệu lực trong header từng trích dẫn theo hướng dẫn của `RAG_PROMPT`. `build_context_graph` nối thêm dòng "Tác động ra ngoài" / "Bị tác động" vào cuối `text` khi Component từng sửa đổi hoặc bị sửa đổi bởi văn bản khác — giúp trả lời được câu hỏi kiểu "quy định này còn hiệu lực không, bị thay bởi văn bản nào". Nếu `n` (Norm) là `None`, vẫn giữ Component trong context nhưng thay header bằng "Không xác định được văn bản gốc" thay vì loại bỏ hẳn.

#### `citation_formatter.py` — Format context kiểu "Trích dẫn N"

- **Hàm:** `format_citations(citations: list[dict]) -> str`
- **Input:** list dict "sạch" (không phụ thuộc field Neo4j), mỗi phần tử cần các key `norm_title`, `norm_number`, `validity_status`, `citation`, `text`; `published_date` là optional.
- **Output:** chuỗi bắt đầu bằng `[NGỮ CẢNH PHÁP LÝ ĐƯỢC CUNG CẤP]`, theo sau là từng khối `--- Trích dẫn N ---` liệt kê Văn bản (kèm số hiệu, và ngày ban hành nếu có), Trạng thái, Điều/Khoản, Nội dung.
- **Ý nghĩa:** Tách riêng khỏi `context_builder.py` để dùng lại độc lập — ví dụ hiển thị lại context đã lưu trong CSV/JSON kết quả benchmark mà không cần chạy lại pipeline/Neo4j. `build_context_flat`/`build_context_graph` đều gọi hàm này để ra format thống nhất.

#### `prompts.py` — Prompt template & sinh câu trả lời

- **Hàm:**
  - `answer(question, context) -> str`: chạy chain `RAG_PROMPT | LLM | StrOutputParser`, trả câu trả lời cuối.
  - `count_tokens(text, model=LLM_MODEL_HEAVY, use_api=False) -> int`: đếm token của prompt.
- **Input/Output:** `answer` nhận câu hỏi + context, trả text câu trả lời. `count_tokens` nhận text, trả số nguyên số token.
- **Ý nghĩa:** `RAG_PROMPT` yêu cầu LLM (đóng vai chuyên gia pháp luật Việt Nam) nêu rõ phạm vi áp dụng, ưu tiên văn bản còn hiệu lực, trích số hiệu + citation cụ thể, và nói rõ nếu context chưa đủ. `count_tokens` mặc định dùng ước lượng local (`len // 3`) để tránh tốn round-trip API chỉ để đếm token trên mỗi request; chỉ gọi API thật khi `use_api=True` (dùng cho báo cáo cuối, cần số chính xác).

#### `pipeline.py` — Ráp toàn bộ thành 2 hàm dùng trực tiếp

5 mode: `vector`, `bm25`, `hybrid`, `graphrag`, `vector_graph` (xem bảng mode ở mục 1.1).

- **`retrieve(question, mode, top_k=TOP_K, max_components=MAX_COMPONENTS, use_rerank=USE_RERANK, client=None, embedding=None, keywords=None) -> list[dict]`**
  - **Input:** câu hỏi, mode, các tham số cấu hình, và `embedding`/`keywords` đã tính sẵn (tuỳ chọn, để tái sử dụng giữa nhiều mode của cùng câu hỏi).
  - **Output:** tối đa `max_components` Component liên quan nhất (đã qua rerank nếu bật). Sort cuối cùng theo `rerank_score` nếu có (`use_rerank=True`), fallback về `score` (RRF/vector/BM25) nếu không rerank — **không** còn ưu tiên cứng theo `validity_status` ở bước cắt này.
  - **Ý nghĩa:** Điều phối theo mode — `vector`/`vector_graph` chỉ gọi `vector_search`; `bm25` chỉ gọi `bm25_search`; `hybrid`/`graphrag` gọi cả `vector_search` + `bm25_search` rồi `merge_search_results`. Với `hybrid`/`graphrag`, nếu cả embedding và keywords đều chưa có thì tính song song bằng `ThreadPoolExecutor` (2 việc độc lập, tiết kiệm latency); nếu chỉ thiếu 1 trong 2, chỉ gọi API cho phần còn thiếu.
- **`run_pipeline(question, mode, top_k=TOP_K, max_components=MAX_COMPONENTS, max_chars=MAX_CHARS_PER_UNIT, max_tokens=MAX_PROMPT_TOKENS, client=None, embedding=None, keywords=None, use_rerank=USE_RERANK) -> dict`**
  - **Input:** giống `retrieve` + `max_chars` (giới hạn ký tự/đoạn) và `max_tokens` (giới hạn token prompt).
  - **Output:** dict gồm `question, mode, answer, context, retrieved, prompt_tokens, latency_sec, retrieved_citations` (`retrieved_citations` là list tuple `(norm_number, citation)` của các seed).
  - **Ý nghĩa:** Hàm end-to-end — `retrieve()` → với mode `graphrag`/`vector_graph`: `expand_graph()` trên `comp_id` của seeds (gắn lại `rerank_score`/`score` từ seeds vào subgraph theo `comp_id`, vì `expand_graph()` truy vấn lại Neo4j nên không tự mang theo 2 field này) → `build_context_graph()`; mode khác → `build_context_flat()` → cắt bớt context nếu prompt vượt `max_tokens` (ước lượng token bằng `count_tokens()` cục bộ, không gọi API) → sinh câu trả lời bằng `answer()`. Chỉ mở **1 Neo4jClient** dùng chung cho cả `vector_search` lẫn `expand_graph` trong 1 lần gọi, tránh mở/đóng driver lặp lại; nếu không truyền `client` từ ngoài, tự mở/đóng riêng (phù hợp gọi lẻ). Đây là hàm được cả app thật và `benchmark/run_benchmark.py` gọi.

> `retrieval/pipeline_debbug.py` là 1 bản sao của `pipeline.py` có thêm `print("[TIMING] ...")` để debug latency từng bước — hiện chỉ có 4 mode (chưa có `vector_graph`), không phải bản dùng chính thức, chỉ giữ lại để đối chiếu khi cần soi bottleneck.

### 1.3 Bảng tổng hợp input/output nhanh

| File                    | Hàm chính                                  | Input                  | Output                              |
| ----------------------- | ------------------------------------------ | ---------------------- | ----------------------------------- |
| `embedder.py`           | `embed_question`                           | câu hỏi                | vector embedding                    |
| `keyword_extractor.py`  | `extract_legal_keywords`                   | câu hỏi                | chuỗi từ khoá pháp lý               |
| `vector_search.py`      | `vector_search`                            | embedding, top_k       | list Component/TextUnit + score     |
| `bm25_index.py`         | `bm25_search`                              | câu hỏi/từ khoá, top_k | list Component/TextUnit + score     |
| `fusion.py`             | `merge_search_results`                     | nhiều list kết quả     | list hợp nhất theo RRF              |
| `reranker.py`           | `rerank`                                   | câu hỏi, seeds         | seeds đã lọc + `rerank_score`       |
| `graph_expand.py`       | `expand_graph`                             | list comp_id           | list {c, n, text_units, actions...} |
| `context_builder.py`    | `build_context_flat`/`build_context_graph` | seeds/subgraph         | chuỗi context                       |
| `citation_formatter.py` | `format_citations`                         | list dict citation     | chuỗi context "Trích dẫn N"         |
| `prompts.py`            | `answer`                                   | câu hỏi, context       | câu trả lời text                    |
| `pipeline.py`           | `retrieve`                                 | câu hỏi, mode          | list Component (seeds)              |
| `pipeline.py`           | `run_pipeline`                             | câu hỏi, mode          | dict kết quả end-to-end             |

---

## 2. Module `benchmark/`

### 2.1 Sơ đồ pipeline tổng quát

```
eval_set.jsonl  ──► load_eval_set()  ──► list[{question, category?}]
                                              │
                          ┌───────────────────┴───────────────────┐
                          │   với mỗi câu hỏi (song song theo câu hỏi,
                          │   5 mode chạy TUẦN TỰ bên trong 1 câu hỏi)  │
                          └───────────────────┬───────────────────┘
                                              ▼
                     run_pipeline(question, mode)  (từ retrieval/pipeline.py)
                                              │
                                              ▼
                     score_with_deepeval(result)  (answer_relevancy,
                                          faithfulness_deepeval, contextual_relevancy)
                                              │
                                              ▼
                              rows (1 dòng / (câu hỏi, mode))
                                              │
                                              ▼
                              run_benchmark() → DataFrame
                                              │
                        ┌─────────────────────┴─────────────────────┐
                        ▼                                           ▼
                  summarize()                          summarize_by_category()
              (trung bình theo mode)                (trung bình theo category x mode)
                        │                                           │
                        ▼                                           ▼
        benchmark_results.csv / benchmark_summary.csv  (lưu vào benchmark/results/)
```

Ngoài ra còn 1 nhóm metric **cần ground truth** (`metrics.py`) độc lập, bổ sung cho nhóm DeepEval (không cần ground truth).

### 2.2 Chi tiết từng file

#### `eval_set.py` — Chuẩn bị bộ câu hỏi đánh giá

- **Hàm:**
  - `build_eval_set_from_jsonl(jsonl_path, sample_size=10, seed=42, category="general") -> list[dict]`: random sample câu hỏi từ 1 file jsonl thô (có `title`, `body`), ghép thành câu hỏi hoàn chỉnh.
  - `load_eval_set(path) -> list[dict]`: đọc file jsonl EVAL_SET đã chuẩn bị sẵn.
- **Input/Output:** input là đường dẫn file jsonl; output là `list[dict]`, mỗi phần tử tối thiểu có `question`, tuỳ chọn `category`.
- **Ý nghĩa:** Vì DeepEval chấm trực tiếp trên context + answer (không cần đáp án chuẩn), EVAL_SET chỉ cần cột `question` là đủ để chạy benchmark.

#### `deepeval_judge.py` — Chấm điểm không cần ground truth

- **Class `GeminiJudge(DeepEvalBaseLLM)`:** wrapper để DeepEval dùng Gemini (qua Vertex AI, `LLM_MODEL_LIGHT`) làm judge model thay vì OpenAI mặc định. Có `generate`/`a_generate`/`load_model`/`get_model_name`; timeout 120s để tránh treo vô hạn khi Vertex AI phản hồi chậm.
- **Hàm:**
  - `to_test_case(result: dict) -> LLMTestCase`: chuyển output của `run_pipeline()` thành `LLMTestCase` của DeepEval — dùng thẳng `result["retrieved"]` (chunk sạch, chưa qua header/annotation của context_builder) làm `retrieval_context`, thay vì split `result["context"]` theo delimiter.
  - `score_with_deepeval(result, metric_names=ALL_METRIC_NAMES, include_reason=False) -> dict`: chạy các metric DeepEval được chọn (mặc định cả 3), trả dict phẳng gồm điểm số + (tuỳ chọn) lý do (`*_reason`).
- **3 metric:**
  - `answer_relevancy`: câu trả lời có bám sát câu hỏi không.
  - `faithfulness_deepeval`: câu trả lời có bịa thông tin ngoài context không.
  - `contextual_relevancy`: context lấy được có liên quan tới câu hỏi không.
- **Ý nghĩa quan trọng:** `_get_metrics()` tạo **metric mới mỗi lần gọi**, không cache dùng chung giữa các thread — vì object metric DeepEval lưu `score`/`reason` làm instance state sau `.measure()`, nếu nhiều thread trong `ThreadPoolExecutor` cùng dùng 1 instance sẽ ghi đè chéo (race condition), gây lệch giữa số điểm và lý do giải thích. `judge_llm` (client gọi API) thì vẫn dùng chung vì không giữ state theo từng lần đo, khởi tạo qua double-checked locking để tránh 2 thread cùng tạo `GeminiJudge()`.

#### `metrics.py` — Metric cần ground truth

- **Hàm:**
  - `recall_at_k(retrieved_citations, gold_citations) -> float | None`: tỉ lệ citation vàng (gold) được tìm thấy trong kết quả retrieve.
  - `mrr(retrieved_citations, gold_citations) -> float`: Mean Reciprocal Rank — 1/rank của citation đúng đầu tiên tìm được.
  - `ndcg_at_k(retrieved_citations, gold_citations) -> float`: Normalized Discounted Cumulative Gain, đánh giá cả thứ hạng của các citation đúng.
  - `llm_judge(question, gold_answer, model_answer) -> dict`: chấm câu trả lời bằng LLM, thang 1–5 cho `correctness` (đúng/đủ nội dung pháp lý) và `faithfulness` (có bịa ngoài context không).
- **Input/Output:** các hàm recall/mrr/ndcg nhận 2 list citation (retrieved vs gold), trả về số thực. `llm_judge` nhận câu hỏi + đáp án chuẩn + câu trả lời mô hình, trả dict `{"correctness": int|None, "faithfulness": int|None}`.
- **Ý nghĩa:** Nhóm metric này **cần** `gold_citations`/`gold_answer` trong EVAL_SET (trả `None` nếu câu đó chưa có ground truth) — bổ sung cho nhóm DeepEval (không cần ground truth), đo trực tiếp độ chính xác retrieval (recall/mrr/ndcg) và chất lượng câu trả lời so với đáp án chuẩn.

#### `run_benchmark.py` — Entrypoint chạy benchmark

- **`MODES = ["bm25", "vector", "hybrid", "graphrag", "vector_graph"]`** — 5 mode mặc định.
- **Hàm:**
  - `_run_one_question(item, modes, include_reason=False, precomputed=None) -> list[dict]`: chạy tuần tự tất cả mode được chọn cho 1 câu hỏi. Với mỗi mode: nếu có sẵn kết quả trong `precomputed[mode][question]` thì dùng lại (không gọi `run_pipeline()`), ngược lại gọi `run_pipeline(question, mode, max_components=5, top_k=5, use_rerank=False)` — **các tham số này hiện bị hardcode trong hàm, không đọc từ `.env`/`config.py`**. Sau đó gọi `score_with_deepeval()`, trả 1 dòng/mode gồm điểm số + `context_len_chars`, `prompt_tokens`, `latency_sec` + các cột `_*_reason` (debug).
  - `run_benchmark(eval_set, modes=MODES, max_workers=4, include_reason=False, precomputed=None) -> pd.DataFrame`: chạy `_run_one_question` cho toàn bộ eval set, **song song hoá theo câu hỏi** (mỗi câu hỏi 1 thread, nhưng các mode bên trong 1 câu hỏi vẫn chạy tuần tự — tránh quá tải rate limit Vertex AI/Neo4j). Trả DataFrame 1 dòng/(câu hỏi, mode).
  - `summarize(df_results) -> pd.DataFrame`: trung bình các metric số theo `mode` — bảng chính so sánh các mode.
  - `summarize_by_category(df_results) -> pd.DataFrame | None`: trung bình theo `category x mode` (chỉ có nếu EVAL_SET có cột `category`) — giúp thấy GraphRAG/vector_graph nổi bật ở nhóm câu hỏi cần suy luận nhiều văn bản.
  - `load_precomputed_results(path) -> dict[str, dict]` và `_load_precomputed_result(path, question)`: đọc CSV kết quả `run_pipeline()` đã chạy sẵn (cột `question`, `result` — dict dạng chuỗi Python, parse bằng `ast.literal_eval`), dùng để tra cứu nhanh trong `_run_one_question` thay vì gọi lại retrieval; bỏ qua các dòng có cột `error` khác rỗng.
  - `main()`: parse CLI args (`--eval-set`, `--modes`, `--out-dir`, `--max-workers`, `--include-reason`, `--precomputed MODE=PATH` — có thể lặp lại nhiều mode), chạy benchmark, in bảng tổng hợp, lưu 2 file CSV (`benchmark_results.csv` chi tiết từng dòng, `benchmark_summary.csv` trung bình theo mode) vào thư mục `--out-dir` (mặc định `benchmark/results/`).
- **Input/Output tổng thể:** Input là 1 file jsonl EVAL_SET (đường dẫn qua `--eval-set`), tuỳ chọn thêm CSV kết quả có sẵn qua `--precomputed`. Output là 2 file CSV trong thư mục kết quả + bảng in ra console.
- **Ý nghĩa:** Đây là script so sánh khách quan 5 chiến lược retrieval trên cùng 1 bộ câu hỏi, dùng DeepEval (không cần ground truth) làm giám khảo tự động. `DEFAULT_MAX_WORKERS = 4` được chọn làm mức an toàn cho quota Vertex AI mặc định, tránh dính rate limit khi chạy song song quá nhiều.

#### `retry_failed.py` — Chạy lại benchmark cho các dòng bị lỗi

- **Usage:** `python -m benchmark.retry_failed --results-csv benchmark/results/benchmark_results.csv --eval-set data/questions.jsonl --max-workers 1`
- **Ý nghĩa:** Đọc lại 1 file `benchmark_results.csv` đã chạy trước đó, xác định dòng lỗi (metric nào đó là NaN, hoặc cột `_*_reason` chứa chuỗi `"lỗi:"` — do `deepeval_judge.py` ghi khi `metric.measure()` raise exception, ví dụ hết quota 429 giữa chừng), chỉ chạy lại benchmark cho các dòng đó rồi merge đè lên kết quả cũ — không cần chạy lại toàn bộ eval set.

#### `retrieval/retry_utils.py` — Retry tự động khi gọi LLM

- **Hàm:** `call_with_retry(fn, max_retries=4, initial_wait_sec=2.0) -> T`
- **Ý nghĩa:** Bọc quanh 1 lời gọi LLM (LangChain) để tự động thử lại khi gặp lỗi 429 (`ResourceExhausted`/hết quota) với exponential backoff, tránh benchmark dừng giữa chừng chỉ vì 1 lần rate-limit tạm thời.

### 2.3 Bảng tổng hợp input/output nhanh

| File                | Hàm chính                           | Input                              | Output                             |
| ------------------- | ----------------------------------- | ---------------------------------- | ---------------------------------- |
| `eval_set.py`       | `load_eval_set`                     | path jsonl                         | list câu hỏi (+ category)          |
| `deepeval_judge.py` | `score_with_deepeval`               | kết quả `run_pipeline()`           | dict 3 metric (+reason)            |
| `metrics.py`        | `recall_at_k`/`mrr`/`ndcg_at_k`     | retrieved & gold citations         | số thực đo chất lượng retrieval    |
| `metrics.py`        | `llm_judge`                         | câu hỏi, gold_answer, model_answer | điểm 1-5 correctness/faithfulness  |
| `run_benchmark.py`  | `run_benchmark`                     | eval_set, modes                    | DataFrame (1 dòng/câu hỏi×mode)    |
| `run_benchmark.py`  | `summarize`/`summarize_by_category` | DataFrame kết quả                  | bảng trung bình theo mode/category |

### 2.4 Cách chạy

```bash
python -m benchmark.run_benchmark --eval-set data/questions.jsonl --modes bm25 vector hybrid graphrag vector_graph --max-workers 4
```

Kết quả lưu tại `benchmark/results/benchmark_results.csv` (chi tiết) và `benchmark/results/benchmark_summary.csv` (trung bình theo mode).
