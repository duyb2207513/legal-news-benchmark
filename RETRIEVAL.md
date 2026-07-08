# Retrieval Pipeline

## Overview

Retrieval Pipeline chịu trách nhiệm tìm kiếm các điều luật liên quan và sinh câu trả lời dựa trên Legal Knowledge Graph.

Pipeline tổng quát:

```text
User Question
      │
      ▼
run_pipeline()
      │
      ├──────────────► Extract Keywords
      │
      ├──────────────► Retrieve Documents
      │                     │
      │                     ├── BM25 Search
      │                     ├── Vector Search
      │                     └── Hybrid Search (RRF)
      │
      ├──────────────► Build Context
      │
      ├──────────────► Build Prompt
      │
      ├──────────────► Gemini LLM
      │
      ▼
Return Answer
```

---

# Step 1. Receive User Question

## Purpose

Nhận câu hỏi từ người dùng và khởi tạo toàn bộ Retrieval Pipeline.

---

## Main Function

```python
run_pipeline(question, mode)
```

---

## Input

```python
question : str
mode : bm25 | vector | hybrid
```

---

## Output

```python
PipelineResult
```

---

## Call Tree

```text
run_pipeline()
│
├── extract_keywords()
├── retrieve()
├── build_context()
├── generate_answer()
└── return result
```

---

# Step 2. Keyword Extraction

## Purpose

Trích xuất các từ khóa pháp lý quan trọng từ câu hỏi.

Những từ khóa này được sử dụng để tăng chất lượng BM25 Search và Graph Retrieval.

---

## Main Function

```python
extract_keywords(question)
```

---

## Call Tree

```text
extract_keywords()
│
├── build_keyword_prompt()
├── Gemini.generate_content()
└── parse_keywords()
```

---

## Input

```text
Question
```

Ví dụ

```text
Doanh nghiệp có được tạm ngừng kinh doanh không?
```

---

## Output

```text
[
    "tạm ngừng kinh doanh",
    "doanh nghiệp",
    "thông báo"
]
```

---

# Step 3. Retrieve Documents

Sau khi có keyword, hệ thống thực hiện tìm kiếm tài liệu.

Tùy mode mà pipeline sẽ chạy khác nhau.

---

# Step 3.1 BM25 Retrieval

## Purpose

Tìm các TextUnit chứa từ khóa giống câu hỏi.

---

## Main Function

```python
bm25_search()
```

---

## Call Tree

```text
bm25_search()
│
├── preprocess_query()
├── QueryParser()
├── searcher.search()
└── format_results()
```

---

## Input

```python
keywords
```

---

## Output

```python
List[RetrievedDocument]
```

---

# Step 3.2 Vector Retrieval

## Purpose

Tìm các TextUnit có embedding gần với embedding của câu hỏi.

---

## Main Function

```python
vector_search()
```

---

## Call Tree

```text
vector_search()
│
├── embedding_model.embed()
├── Neo4j Vector Index
└── format_results()
```

---

## Output

```python
List[RetrievedDocument]
```

---

# Step 3.3 Hybrid Retrieval

## Purpose

Kết hợp kết quả từ BM25 và Vector Search.

---

## Main Function

```python
hybrid_search()
```

---

## Call Tree

```text
hybrid_search()
│
├── bm25_search()
├── vector_search()
├── reciprocal_rank_fusion()
└── rerank()
```

---

## Hybrid Algorithm

```text
BM25 Ranking

        +

Vector Ranking

        │

        ▼

Reciprocal Rank Fusion

        │

        ▼

Final Ranking
```

---

# Step 4. Build Context

## Purpose

Ghép các TextUnit được retrieve thành context để gửi cho LLM.

---

## Main Function

```python
build_context()
```

---

## Call Tree

```text
build_context()
│
├── sort_documents()
├── remove_duplicate()
├── concatenate()
└── return context
```

---

## Output

```text
Context
```

Ví dụ

```text
Điều ...

Khoản ...

Điểm ...
```

---

# Step 5. Prompt Construction

## Purpose

Xây dựng prompt cho Gemini.

---

## Main Function

```python
build_answer_prompt()
```

---

## Prompt Structure

```text
System Prompt

+

Retrieved Context

+

Question
```

---

# Step 6. Answer Generation

## Purpose

Sinh câu trả lời cuối cùng.

---

## Main Function

```python
generate_answer()
```

---

## Call Tree

```text
generate_answer()
│
├── ChatVertexAI()
├── llm.invoke()
└── parse_answer()
```

---

## Output

```python
answer
```

---

# Step 7. Return Pipeline Result

## Output Structure

```python
PipelineResult(
    answer,
    retrieved_documents,
    context,
    latency,
    prompt_tokens
)
```

---

# Complete Retrieval Pipeline

```text
                        User Question
                               │
                               ▼
                    run_pipeline(question)
                               │
                               ▼
                    extract_keywords()
                               │
                               ▼
                     Retrieve Documents
                     ┌─────────┼──────────┐
                     │         │          │
                     ▼         ▼          ▼
                  BM25     Vector     Hybrid
                     │         │          │
                     └─────────┴──────────┘
                               │
                               ▼
                    Reciprocal Rank Fusion
                               │
                               ▼
                        Build Context
                               │
                               ▼
                      Build Prompt
                               │
                               ▼
                        Gemini LLM
                               │
                               ▼
                         Final Answer
```

---

# Data Flow

```text
Question
    │
    ▼
Keyword Extraction
    │
    ▼
Retrieve Documents
    │
    ▼
Rank Documents
    │
    ▼
Context Builder
    │
    ▼
Prompt Builder
    │
    ▼
Gemini
    │
    ▼
Answer
```
