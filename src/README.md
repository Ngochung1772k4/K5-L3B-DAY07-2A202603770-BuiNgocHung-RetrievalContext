# Code Day 7 (`src/`)

Pipeline Agentic RAG và đánh giá nhiều phương pháp retrieval nối trực tiếp với kho dữ liệu & Chroma vector store từ Day 6.

## Cấu trúc module

- `src/config.py`: Cấu hình đường dẫn Day 6 (`DAY6_CHROMA_DIR`), embedding (`sentence-transformers/all-MiniLM-L6-v2`, 384d), Ollama (`qwen2.5:7b`).
- `src/prepare_day6.py`: Chuyển 176 chunks Strategy B từ Day 6 + 27 câu có đáp án (`20 dev / 7 test`) + 5 câu thiếu bằng chứng (`3 dev / 2 test`) sang chuẩn JSON Day 7 tại `data/`.
- `src/retrievers.py`: 5 bộ truy xuất kế thừa cùng interface:
  - `TfidfRetriever` (lexical baseline)
  - `Bm25Retriever` (Okapi BM25)
  - `ChromaDenseRetriever` (Day 6 Chroma PersistentClient `day6_b` + FastEmbed 384d; tự động dựng cache cục bộ từ `data/day6_chunks.json` nếu thiếu thư mục Day 6)
  - `HybridRrfRetriever` (Dense + BM25 hợp nhất bằng Reciprocal Rank Fusion)
  - `LlmRerankRetriever` (Hybrid RRF + rerank bằng `qwen2.5:7b` qua Ollama)
- `src/context_packing.py`: Đóng gói context (`pack_first_fit`, `pack_multi_turn` cân bằng các lượt search, `expand_with_neighbors` mở rộng đoạn liền kề sau primary hits).
- `src/eval_retrieval.py`: Benchmark retrieval và context packing trên `dev` hoặc `test`.
- `src/ollama_agent.py`: Vòng lặp Agent dùng tool calling thật của Ollama (`search_chunks`, `get_adjacent_chunk`), kiểm tra citation `[chunk-id]`, chống prompt injection và xử lý từ chối (`abstain`).
- `src/starter_adapters.py`: Adapter triển khai `Controller` và `Retriever` của `starter/agentic.py`.
- `src/run_agent_eval.py`: Chạy agent trên toàn bộ `dev` hoặc `test` và lưu trace.
- `src/generate_review_sheet.py`: In trace để đối chiếu nguồn và chấm thủ công `reviews.json`.
- `src/inspect_errors.py`: So sánh metrics và liệt kê lỗi theo từng câu hỏi.
- `src/adv_ingest.py`: Pipeline nạp đa phương thức cho **Thử thách nâng cao (+15đ)** trên 200 trang (`p.281–495`, Chapters 13–21 của *OpenStax Introduction to Sociology 2e*): trích xuất văn bản phân đoạn câu, bảng Markdown (`pdfplumber`), ảnh crop PNG (`PyMuPDF` `results/adv_crops/`), OCR tọa độ 2D (`RapidOCR`), VLM visual summary (`qwen3-vl:2b`), và lập chỉ mục Chroma (`adv_soc_ch13_21`, `1,077` chunks).
- `src/prepare_adv_benchmark.py`: Tạo bộ câu hỏi nâng cao tách biệt `dev` (`18` câu: `3 table, 3 image_diagram, 3 chart, 3 multihop, 3 text, 3 unanswerable`) và `test` (`30` câu: `6 table, 5 image_diagram, 5 chart, 4 multihop, 4 text, 6 unanswerable`) tại `data/adv_cases.json`.
- `src/adv_agent.py`: Multimodal Agentic RAG (`search_adv`, `get_page_assets`, `inspect_figure_vlm`, `get_adjacent_chunk`) kết hợp bước hậu kiểm chứng hai pha (`verify_and_format_answer`) đối chiếu trích dẫn nguyên văn và chặn bịa đặt.
- `src/adv_eval.py`: Chạy đánh giá phần nâng cao trên `dev` hoặc `test` và xuất `traces.json`.

## Lệnh chạy

```powershell
# 1. Cài môi trường
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. Chạy unit tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# 3. Chuẩn bị dữ liệu từ Day 6 (tùy chọn nếu đã có data/day6_*.json)
.\.venv\Scripts\python.exe -m src.prepare_day6

# 4. Đánh giá retrieval trên dev và test
.\.venv\Scripts\python.exe -m src.eval_retrieval --split dev
.\.venv\Scripts\python.exe -m src.eval_retrieval --split test --configs baseline_tfidf imp3_hybrid_filter_ctx2800

# 5. Chạy Agent cơ bản (cần Ollama đang chạy với model qwen2.5:7b)
.\.venv\Scripts\python.exe -m src.run_agent_eval --split dev --config baseline
.\.venv\Scripts\python.exe -m src.run_agent_eval --split dev --config selected
.\.venv\Scripts\python.exe -m src.run_agent_eval --split test --config baseline
.\.venv\Scripts\python.exe -m src.run_agent_eval --split test --config selected

# 6. Thử thách nâng cao (200 trang đa phương thức, cần qwen2.5:7b và qwen3-vl:2b trên Ollama)
# Tái tạo chỉ mục Chroma từ data/adv_chunks.json (hoặc chạy full ingest với --caption-vlm --index):
.\.venv\Scripts\python.exe -m src.adv_ingest --skip-extract --index
.\.venv\Scripts\python.exe -m src.prepare_adv_benchmark
.\.venv\Scripts\python.exe -u -m src.adv_eval --split dev
.\.venv\Scripts\python.exe -u -m src.adv_eval --split test

# 7. Mở giao diện Web UI trực quan (Duyệt trace + Ảnh crop gốc + Chạy thử Agent live)
.\.venv\Scripts\python.exe -m src.ui_app --port 8765
# Mở trình duyệt tại: http://127.0.0.1:8765
```
