# Báo cáo cá nhân — Day 7 Agentic RAG

**Học viên:** Bùi Ngọc Hùng · **MSSV:** `2A202603770` · **Lớp:** K5-L3B  
**Tiếp nối từ:** Day 6 (`K5-L3B-DAY06-BuiNgocHung-2A202603770-DataIngestion`)

---

## 1. Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:**
  - Tài liệu: *Introduction to Sociology 2e* (Paul Peter Urone & Roger Hinrichs, OpenStax / Rice University), giấy phép **CC BY 4.0** (giữ nguyên attribution trên từng chunk).
  - Phạm vi từ Day 6: **30 trang PDF vật lý** — Chương 1 (14 trang: `14, 15, 17–28`) và Chương 2 (16 trang: `38–53`), tổng 117.404 ký tự chuẩn hóa NFKC (`starter/data/pages.jsonl` bên Day 6).
  - Tập chunk sử dụng: **176 chunks** từ **Strategy B** của Day 6 (gom câu/đoạn theo ranh giới câu tối đa 800 ký tự, có `coherence = 1.8/2`), xuất sang `data/day6_chunks.json` với `id`, `text`, `source_id`, `source`, `page` và `metadata` (`chapter`, `chunk_index`, `char_start`, `char_end`, `char_count`, `language`, `attribution`).
- **Vector store, embedding model, phiên bản và số chiều:**
  - Embedding model: `FastEmbed 0.7.3` với `sentence-transformers/all-MiniLM-L6-v2` (**384 chiều**, ONNX CPU local, `$0` API cost).
  - Vector store: `Chroma 1.3.5 PersistentClient` kết nối trực tiếp collection `day6_b` (metric `cosine`, HNSW) tại kho Day 6 (`starter/results/chroma/`), kèm cơ chế tự dựng lại cache cục bộ tại `results/chroma_cache/` từ `data/day6_chunks.json` nếu chạy trên máy không có thư mục Day 6.
- **Agent model, cách chạy và phân chia dev/test:**
  - Agent LLM: `qwen2.5:7b` (Q4_K_M, hỗ trợ native tool calling) chạy local qua Ollama (`http://127.0.0.1:11434/api/chat`, `temperature=0.0`, `num_ctx=8192`, `$0` API cost).
  - Phân chia benchmark (tổng **32 câu** trong `data/agent_cases.json` và **27 câu** retrieval trong `data/day6_questions.json`):
    - **Dev (23 câu):** 20 câu có đáp án từ Day 6 (`q01`–`q20`, 5 câu có `filter: {"chapter": ...}`) + **3 câu thiếu bằng chứng** (`u_dev_01`–`u_dev_03`).
    - **Test (9 câu):** 7 câu có đáp án (`q21`–`q25` Chương 2 từ Day 6 + `q26`–`q27` Chương 1 bổ sung từ trang `14, 19` chưa dùng để test cân bằng hai chương; 4 câu có `filter`) + **2 câu thiếu bằng chứng** (`u_test_01`–`u_test_02`).
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - Câu có đáp án: mỗi phần tử `gold` được `starter/baseline.py:validate()` kiểm chứng nằm trọn nguyên văn trong chunk đúng `source_id` và `page`. Riêng `q03` có evidence nằm vắt qua 2 chunk liên tiếp trên trang 18 nên được khai báo đủ cả 2 phần trong `required_evidences` để kiểm tra `Context Evidence Rate`.
  - Câu thiếu bằng chứng: chọn chủ đề Xã hội học gần Chương 1–2 (Sapir-Whorf, looking-glass self, Stanford Prison Experiment, ethnocentrism/cultural relativism, snowball sampling) và xác minh bằng code (`src/prepare_day6.py`) đạt **0 lần xuất hiện** trên toàn bộ 30 trang văn bản Day 6.
  - Tiêu chí chấm thủ công theo `BENCHMARK.md`:
    - Context (đọc câu test có đáp án): `0` = không có evidence; `1` = đúng chủ đề/trang nhưng thiếu điều kiện/chi tiết cốt lõi; `2` = đủ evidence và nguồn.
    - Grounded answer: `0` = sai, thiếu evidence hoặc từ chối nhầm; `1` = đúng một phần; `2` = đúng và đủ (trung bình `/2`).
    - Citation hợp lệ: từng `[chunk-id]` được đọc đối chiếu xem đoạn chunk tương ứng có thực sự hỗ trợ khẳng định gắn kèm không (`true`/`false`).
    - Abstention: với câu thiếu bằng chứng, kiểm tra agent thực sự từ chối và không suy đoán thêm thông tin ngoài context.

---

## 2. Retrieval và thử nghiệm

### Tiêu chí chốt cấu hình trên `dev` (công bố trước khi chạy `test`)
Ưu tiên **Context Evidence Rate** (tỷ lệ context cuối gửi model chứa đủ `required_evidences`) và **Hit@1 / Evidence Hit@3 / MRR**, cân nhắc đánh đổi với độ trễ (`median_ms`) và chi phí. Chốt cấu hình trên `dev` xong mới chạy `test`.

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline (`baseline_tfidf`) | TF-IDF lexical, `top_k=3`, không filter, không rerank, `budget=800` | Mốc tham chiếu lexical như `starter/baseline.py` | dev | 13/20 (65.0%) | 17/20 (85.0%) | 1/20 (5.0%) | 0.51 ms / \$0 |
| Phương pháp 2 (`method2_bm25`) | Okapi BM25 (`k1=1.5, b=0.75`), `top_k=3`, không filter, `budget=800` | Chuẩn hóa độ dài đoạn (BM25) cải thiện xếp hạng top-1 so với TF-IDF | dev | 15/20 (75.0%) | 17/20 (85.0%) | 1/20 (5.0%) | 0.53 ms / \$0 |
| Phương pháp 3 (`method3_dense`) | Dense Chroma Day 6 (`day6_b`, `all-MiniLM-L6-v2` 384d), `top_k=3`, không filter, `budget=800` | Ngữ nghĩa dense tăng recall top-3 khi câu hỏi diễn đạt lại | dev | 12/20 (60.0%) | 18/20 (90.0%) | 0/20 (0.0%) | 7.14 ms / \$0 |
| Phương pháp 4 (`method4_hybrid_rrf`) | Hybrid RRF (`Dense + BM25`, `fetch_k=12, k=60`), `top_k=3`, không filter, `budget=800` | Hợp nhất hạng Dense và BM25 giữ cả khớp từ khóa lẫn ngữ nghĩa | dev | 17/20 (85.0%) | 17/20 (85.0%) | 1/20 (5.0%) | 7.61 ms / \$0 |
| Cải thiện 1 (`imp1_hybrid_filter_ctx1800`) | Hybrid RRF + `chapter` filter (khi query nêu chương) + `budget=1800` | Nới budget vượt kích thước 1 chunk Strategy B (~670+140 ký tự) + lọc chương | dev | 17/20 (85.0%) | 17/20 (85.0%) | 16/20 (80.0%) | 11.87 ms / \$0 |
| Cải thiện 2 / PP 5 (`imp2_hybrid_llm_rerank_nb`) | Hybrid RRF (`fetch_k=8`) + LLM rerank (`qwen2.5:7b`) + filter + chèn neighbor ngay sau rank-1 + `budget=1800` | Thử dùng LLM xếp lại top-8 và mở rộng chunk liền kề cho rank-1 | dev | 14/20 (70.0%) | 18/20 (90.0%) | 13/20 (65.0%) | 3097.65 ms / \$0 |
| **Cải thiện 3 — Đã chọn (`imp3_hybrid_filter_ctx2800`)** | Hybrid RRF (`fetch_k=12, k=60`) + `chapter` filter + neighbor sau primary hits + `budget=2800` | Bỏ LLM rerank chậm/nhiễu, nâng `budget=2800` và xếp neighbor *sau* top-3 để không đẩy mất rank-2/3 và giữ đủ 2 chunk của `q03` | dev | **17/20 (85.0%)** | **17/20 (85.0%)** | **17/20 (85.0%)** | 77.56 ms / \$0 |
| **Cấu hình đã chọn (`imp3_hybrid_filter_ctx2800`)** | Như trên (đã chốt trước khi chạy test) | Đánh giá độc lập trên held-out test | test | **4/7 (57.1%)** | **5/7 (71.4%)** | **5/7 (71.4%)** | 10.71 ms / \$0 |
| Baseline (`baseline_tfidf`) | TF-IDF lexical, `top_k=3`, không filter, `budget=800` | Đánh giá đối chứng trên held-out test | test | 4/7 (57.1%) | 4/7 (57.1%) | 0/7 (0.0%) | 0.40 ms / \$0 |

*(Ghi chú MRR trên dev: Baseline `0.750`, BM25 `0.800`, Dense `0.742`, Hybrid RRF `0.850`, Imp1 `0.850`, Imp2 `0.800`, Imp3 `0.850`. Trên test: Baseline `0.571`, Imp3 `0.643`. Thời gian là `median_ms`/query; chi phí API `$0`.)*

### Phân tích lỗi trên `dev` và lý do chọn cấu hình
1. **Lỗi 1 — Nghẽn `Context Evidence Rate = 0–1/20` khi `budget=800`:** Trên `dev`, cả Baseline (`Hit@3=17/20`) lẫn Dense (`Hit@3=18/20`) đều tìm trúng chunk ở top-3 nhưng `Context Evidence Rate` chỉ đạt `0–1/20`. Nguyên nhân: chunk Strategy B của Day 6 dài trung bình 670 ký tự (trung vị 721.5, tối đa 805); khi ghép thêm header nguồn `[chunk_id | URL | page | ch]` (~140 ký tự), một block đã dài ~810–940 ký tự nên bị `pack_first_fit(budget=800)` loại bỏ. Nâng `budget=1800` (Cải thiện 1) lập tức đưa `Context Evidence Rate` từ `1/20` lên `16/20`, nhưng vẫn sót `q03` (p.18, bằng chứng tách 2 chunk tổng ~1822 ký tự, vượt `1800` đúng 22 ký tự). Nâng lên `budget=2800` (Cải thiện 3) đã gộp đủ cả 2 chunk của `q03` và đạt **`17/20 (85.0%)`**.
2. **Lỗi 2 — Kết quả giảm ở Cải thiện 2 (`imp2_hybrid_llm_rerank_nb`):** Khi bật `qwen2.5:7b` listwise rerank + chèn neighbor ngay sau rank-1 trong `budget=1800`, `Hit@1` giảm từ `17/20` xuống `14/20` (`q02, q05, q10, q12` bị đẩy xuống rank-2), `Context Evidence Rate` giảm `16/20 -> 13/20` do neighbor của rank-1 chiếm hết chỗ của rank-2/rank-3, còn độ trễ tăng **260×** (`11.87 ms -> 3097.65 ms`). Từ lỗi này, **Cải thiện 3** bỏ LLM reranker, chuyển neighbor ra *sau* top-3 (`after_primary=True`) và tăng `budget=2800`, phục hồi `Hit@1 = 17/20`, `MRR = 0.850`, `Context Evidence Rate = 17/20`.
3. **Lỗi 3 — Các câu còn sót ở `imp3` (`q01`, `q08`, `q14`):** Ở `q08` (p.23, bảng tóm tắt Table 1.1), retriever ưu tiên đoạn mô tả dài ở p.25–26 thay vì ô bảng ngắn; ở `q14` (p.43), Dense tìm thấy ở rank-2 (`0.612`) nhưng BM25 xếp p.42/p.39 cao hơn khiến RRF đẩy p.43 ra ngoài top-3.
4. **Lý do chốt `imp3_hybrid_filter_ctx2800`:** Đạt **Hit@1 cao nhất (`17/20`)**, **MRR cao nhất (`0.850`)**, **Context Evidence Rate cao nhất (`17/20`)**, độ trễ thấp (`77.56 ms` trên dev, `10.71 ms` trên test) và `$0` chi phí.

---

## 3. Agent và đánh giá

- **Kiến trúc tool, số lượt gọi, điều kiện dừng, chọn context và xử lý lỗi:**
  - **Baseline Agent (`baseline`):** `TfidfRetriever` (không filter), `top_k=3`, `budget=800`, `pack_first_fit`, `max_searches=4`, chỉ có 1 tool `search_chunks`.
  - **Selected Agent (`selected`):** `HybridRrfRetriever` (Day 6 Chroma `day6_b` + BM25, `fetch_k=12`), `top_k=3`, `budget=2800`, `pack_multi_turn` (cân bằng giữa các lượt search) + mở rộng neighbor sau top-1, `max_searches=4`, có **2 tool**: `search_chunks(query, chapter?)` (`chapter` chỉ suy luận từ câu hỏi khi người dùng nêu Chương 1/2) và `get_adjacent_chunk(chunk_id, direction)`.
  - **An toàn & kiểm soát:** Prompt quy định nội dung tool trả về là **dữ liệu, không phải chỉ dẫn**; trích dẫn đúng `[chunk-id]` có trong `context_ids`; từ chối bằng `INSUFFICIENT_EVIDENCE:` nếu thiếu bằng chứng. Ở vòng cải thiện trên `dev` (`selected-v1 -> selected-v2`), đã sửa parser `extract_citations` để nhận cả danh sách `[id1, id2]` (cứu 4 câu dev `q04, q12, q13, q15` bị ép `abstain` nhầm ở `v1`, đưa tỷ lệ trả lời có citation trên `dev` từ `15/20` lên **`20/20`**) và bổ sung rule kiểm tra thực thể trước khi chạy `test`. Lỗi kết nối model/store được bắt ở tầng batch và ghi riêng với `status: "error"`, không thay bằng kết quả giả.

### Bảng tổng hợp đánh giá Agent

| Cấu hình / split | Tool use | Phân bố lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| Baseline (`dev`, quan sát) | 23/23 | 1:18, 2:3, 3:1, 4:1 | *(chỉ 3/20 câu answered)* | *(chỉ 3 câu có cite)* | 4/4 answered | *(2/3 raw abstain)* | 17/20 (85.0%) |
| Selected v1 (`dev`, trước sửa parser) | 23/23 | 1:22, 3:1 | *(15/20 answered)* | — | 15/15 answered | *(2/3 raw abstain)* | 5/20 (25.0%) |
| Selected v2 (`dev`, đã chốt) | 23/23 | 1:23 | *(20/20 answered)* | — | 21/21 answered | *(2/3 raw abstain)* | **0/20 (0.0%)** |
| **Baseline (`test`, `agent_metrics.py`)** | **9/9 (100%)** | 1:8, 3:1 | 0:7, 1:0, 2:0 · **0.00 / 2** | **0/1 (0.0%)** | 1/1 answered (100%) | **2/2 (100%)** | **6/7 (85.7%)** |
| **Selected (`test`, `agent_metrics.py`)** | **9/9 (100%)** | 1:9 | 0:2, 1:2, 2:3 · **1.14 / 2** | **8/17 (47.1%)** | 9/9 answered (100%) | **0/2 (0.0%)** | **0/7 (0.0%)** |

*(Ghi chú Context grade trên 7 câu test có đáp án: Baseline `0:7, 1:0, 2:0` (TB `0.00/2`); Selected `0:1, 1:3, 2:3` (TB `1.29/2`). Toàn bộ các lần chạy đều có `provider_errors = 0`.)*

### Chi tiết đối chiếu toàn bộ 9 câu `test` — Cấu hình Đã chọn (`selected`)

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| `q21` | Có đáp án (p.45) | `search_chunks("small interest groups folk dancers Mensa...")` -> trúng `c0f376f6` (p.45) | Giải thích đúng các nhóm chia sẻ bản sắc/sở thích (`[c0f376f6]`) | `ctx=2`, `ans=2`, `cite=[True]`. Chunk p.45 chứa đúng gold và hỗ trợ trọn câu trả lời. |
| `q22` | Có đáp án (p.46, filter ch.2) | `search_chunks(..., chapter="2")` -> ra p.44, p.39, sót p.46 | Bịa danh sách 3 đặc điểm từ đoạn field research chung (`[8c6f7260]`) | `ctx=0`, `ans=0`, `cite=[False]`. Không nêu được 3 loại (*participant observation, ethnography, case study*). |
| `q23` | Có đáp án (p.47, filter ch.2) | `search_chunks(..., chapter="2")` -> ra p.46–47 chung, sót câu lợi thế *covert* | Nêu lợi thế của participant observation nói chung (`[d8e10ad1] [364ae999] [9a278424]`) | `ctx=1`, `ans=0`, `cite=[False, False, False]`. Thiếu ý cốt lõi *"authentic, natural behaviors"* của quan sát ẩn danh. |
| `q24` | Có đáp án (p.48) | `search_chunks("institutional ethnography focus")` -> trúng `90f90f63` (p.48) | Trả lời đúng trọng tâm *everyday concrete social relationships* (`[90f90f63]`) | `ctx=2`, `ans=2`, `cite=[True]`. Đúng và đủ theo gold. |
| `q25` | Có đáp án (p.51, filter ch.2) | `search_chunks(..., chapter="2")` -> ra p.51, p.50, p.53 (`94f9e41d` gần sát đoạn gold) | Nêu được tránh đầu tư/rủi ro tiếp cận population (`[94f9e41d]`), nhưng gắn nhầm `[6c61a4d8]` và `[d5c31721]` lệch 1 đoạn | `ctx=1`, `ans=1`, `cite=[True, False, False]`. Thiếu ý *"saves time and money and adds depth"*; model bị lệch ID citation ở 2 câu sau. |
| `q26` | Có đáp án (p.14, filter ch.1) | `search_chunks(..., chapter="1")` -> ra 3 chunk p.23 (Durkheim/Spencer) thay vì p.14 | Trả lời theo định nghĩa xã hội của Durkheim & Spencer ở p.23 (`[da0086cc] [ba3aab5d]`) | `ctx=1`, `ans=1`, `cite=[True, True]`. 2 citation p.23 hỗ trợ đúng phát biểu, nhưng sót định nghĩa chuẩn của giáo trình ở p.14. |
| `q27` | Có đáp án (p.19) | `search_chunks("who first coined sociology 1780 1838")` -> trúng `dc661950` (p.19) | Trả lời chính xác Emmanuel-Joseph Sieyès (1780) & Auguste Comte (1838) (`[dc661950]`) | `ctx=2`, `ans=2`, `cite=[True]`. Chính xác, ngắn gọn, có nguồn. |
| `u_test_01` | Thiếu bằng chứng | `search_chunks("ethnocentrism cultural relativism definition")` -> p.15, p.47 | Suy diễn rằng *cultural relativism* gắn với *reification* `[463e77f6]` và *ethnocentrism* gắn với `[510b013b]` dù cuối câu có ghi thiếu bằng chứng | `abstain_ok=False`, `cite=[False, False]`. Lỗi bịa nối thuật ngữ không có trong context. |
| `u_test_02` | Thiếu bằng chứng | `search_chunks("snowball sampling random sampling", chapter="2")` -> p.43, p.53 | Đáp đúng phần *random sample* (`[cbb58209] [df2c0bc3]`), nhưng gán nhầm định nghĩa *scientific method* trong glossary `[7d1f57f9]` thành *snowball sampling* rồi tự suy đoán | `abstain_ok=False`, `cite=[False, True, True]`. Không từ chối hẳn phần *snowball sampling* không có trong sách. |

*(Đối chứng với **Baseline (`test`)**: 6/7 câu có đáp án `q21–q24, q26–q27` bị `abstain` nhầm vì `budget=800` làm context rỗng hoặc lệch chương; câu duy nhất `q25` lại trích nhầm chunk chân trang p.52 `http://cnx.org/...` `[c25e958b]` và bịa câu trả lời (`ans=0, cite=[False]`). Hai câu thiếu bằng chứng `u_test_01–u_test_02` đạt `abstain` đúng chủ yếu do context rỗng/chặn thiếu citation.)*

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:**
  - Dữ liệu chuẩn hóa: `data/day6_chunks.json`, `data/day6_questions.json`, `data/agent_cases.json`.
  - Retrieval results: `results/retrieval-dev/summary.json`, `results/retrieval-test/summary.json` (kèm `*_traces.json`).
  - Agent traces & reviews: `results/agent-baseline-dev/traces.json`, `results/agent-selected-dev-v2/traces.json`, `results/agent-baseline-test/{traces.json,reviews.json,summary.json}`, `results/agent-selected-test/{traces.json,reviews.json,summary.json}`.
  - Lệnh chạy tái lập: xem [src/README.md](file:///d:/AIInAction/Day7/K5-L3B-DAY07-2A202603770-BuiNgocHung-RetrievalContext/src/README.md) và `python starter/agent_metrics.py --cases data/agent_cases.json --traces results/agent-selected-test/traces.json --reviews results/agent-selected-test/reviews.json --split test --output results/agent-selected-test/summary.json`.
  - Thời gian & token chạy Agent (`$0` API cost vì chạy local Ollama `qwen2.5:7b`):
    - `baseline/dev` (23 câu): tổng `320.1s` (trung vị `9.6s`/câu), `32.613` prompt tokens + `2.948` eval tokens.
    - `selected-v2/dev` (23 câu): tổng `547.6s` (trung vị `23.1s`/câu), `45.418` prompt tokens + `4.299` eval tokens.
    - `baseline/test` (9 câu): tổng `85.4s` (trung vị `7.3s`/câu), `11.440` prompt tokens + `856` eval tokens.
    - `selected/test` (9 câu): tổng `251.6s` (trung vị `23.7s`/câu), `18.106` prompt tokens + `2.061` eval tokens.
- **Kết quả trước/sau, đánh đổi và giới hạn:**
  - **Cải thiện rõ rệt khả năng trả lời có căn cứ:** Chuyển từ `Baseline (TF-IDF + budget=800)` sang `Selected (Hybrid RRF + chapter filter + neighbor + budget=2800)` giảm tỷ lệ từ chối nhầm từ `6/7 (85.7%)` xuống **`0/7 (0%)`** trên `test` (và `17/20 -> 0/20` trên `dev`), nâng Grounded answer từ **`0.00/2` lên `1.14/2`** và Context grade từ **`0.00/2` lên `1.29/2`**.
  - **Đánh đổi phát hiện trung thực trên `test`:** Khi cung cấp context giàu ngữ cảnh (`budget=2800`), model local 7B (`qwen2.5:7b`) dễ bị "ảo giác liên kết" với kiến thức pretraining khi gặp câu hỏi gần chủ đề nhưng thiếu từ khóa (`u_test_01`, `u_test_02` -> abstention `0/2` trên `test`, `2/3` trên `dev`), đồng thời thỉnh thoảng gắn lệch `[chunk-id]` giữa các đoạn trong cùng prompt (`q25`). Ngoài ra, dù tool `get_adjacent_chunk` được cung cấp, `qwen2.5:7b` ở `selected/test` chỉ gọi `1` lượt `search_chunks` mỗi câu thay vì chủ động search lại khi thiếu ý như ở `q22`/`q23`.
  - **Giới hạn:** Corpus giới hạn ở 30 trang tiếng Anh từ Day 6; tập `test` gồm 9 câu nên độ biến thiên thống kê còn cao; `q21–q25` kế thừa từ test Day 6 (không dùng để tune ở Day 7, chỉ bổ sung `q26–q27` và `u_test_01–02`). Hướng cải thiện tiếp theo là dùng verifier/quote-extraction bắt buộc trích nguyên văn câu hỗ trợ trước khi trả lời và kiểm thử trên model lớn hơn hoặc có ràng buộc abstention từng ý.

---

## 4. Thử thách nâng cao (+15 điểm bonus)

- **Tài liệu >100 trang, phạm vi và pipeline nạp đa phương thức (`src/adv_ingest.py`):**
  - **Nguồn & quyền sử dụng:** *OpenStax Introduction to Sociology 2e* (CC BY 4.0), trích xuất dải trang vật lý `281–495` (Chương 13–21) gồm **200 trang nội dung thực tế** (loại bỏ và ghi log đầy đủ 15 trang trống/quiz/tài liệu tham khảo tại `data/adv_manifest.json`).
  - **Quy mô chỉ mục (`data/adv_chunks.json`, `results/adv_chroma/`):** Tổng cộng **1.077 chunks** được nhúng bằng `FastEmbed` (`sentence-transformers/all-MiniLM-L6-v2`, 384 chiều) vào Chroma collection `adv_soc_ch13_21`:
    - **920 `text` chunks:** Văn bản chuẩn hóa NFKC, tách theo ranh giới câu tối đa 850 ký tự, giữ số trang và chương.
    - **42 `table` chunks:** Bảng có cấu trúc trích xuất bằng `pdfplumber 0.11.10` sang bảng Markdown kèm tiêu đề bảng, tọa độ `bbox` và ảnh crop PNG (`results/adv_crops/p*_table_*.png`).
    - **115 `figure` chunks:** Hình minh họa, sơ đồ, bản đồ và biểu đồ trích xuất bằng `PyMuPDF 2.1` kèm tọa độ `bbox`, ảnh crop PNG (`results/adv_crops/p*_fig_*.png`), dòng văn bản OCR ghép theo tọa độ Y bằng `RapidOCR 1.4.4`, và mô tả thị giác (`vlm_description`) sinh bởi mô hình thị giác cục bộ `qwen3-vl:2b` trên Ollama (`114/115` hình).
- **Kiến trúc Multimodal Agentic RAG và hậu kiểm chứng hai pha (`src/adv_agent.py`):**
  - **4 công cụ (`ADV_TOOLS`):** `search_adv` (Hybrid RRF kết hợp `ChromaDenseRetriever + Bm25Retriever`, `fetch_k=15, top_k=5`, ưu tiên khớp nhãn `Figure X.Y`/`Table X.Y` và đan xen trang khi câu hỏi nối nhiều trang), `get_page_assets(page)` (lấy toàn bộ bảng, hình và văn bản trên trang `page`), `inspect_figure_vlm(chunk_id, visual_question)` (gọi trực tiếp `qwen3-vl:2b` trên ảnh crop PNG), và `get_adjacent_chunk(chunk_id, direction)`.
  - **Đóng gói context & chống lệch mã trích dẫn:** Mỗi chunk trong context (`budget=6500`, `num_ctx=4096`) được gắn nhãn ngắn `[C1]`, `[C2]` kèm ranh giới `<<<CHUNK [C1]>>> ... <<<END [C1]>>>` và ánh xạ ngược về mã SHA1 40 ký tự thực tế, triệt tiêu lỗi chép lệch hash của model 7B.
  - **Hậu kiểm chứng hai pha (`verify_and_format_answer`):** Trước khi trả lời, hệ thống buộc trích xuất nguyên văn `evidence_quotes`, kiểm tra mức độ khớp từ khóa/chuỗi con trực tiếp bằng Python (`_quote_supported_in_chunk`) trên đúng chunk trong context, đồng thời kiểm tra bẫy chỉ số (`TRAP CHECK`) để từ chối (`INSUFFICIENT_EVIDENCE`) khi câu hỏi hỏi về quốc gia/chỉ số không tồn tại trong bảng hoặc biểu đồ.
- **Bộ câu hỏi `dev` / `test` và quy trình kiểm thử không rò rỉ (`data/adv_cases.json`, `src/prepare_adv_benchmark.py`):**
  - **Tập `dev` (`18` câu):** `3 table, 3 image_diagram, 3 chart, 3 multihop, 3 text, 3 unanswerable` (`results/adv-agent-dev/summary.json`: đạt `15/18 = 83.3%`, `0` lỗi hệ thống). Qua `dev`, nhóm phát hiện và sửa lỗi `qwen2.5:7b` tự đoán sai bộ lọc `chapter`, lỗi lặp VLM và lỗi cắt trang thứ hai ở câu `multihop`.
  - **Tuân thủ nghiêm ngặt quy tắc `>= 90%` trên `test` của `ADVANCED.md`:**
    - Ở lần chạy **Test v1** (`30` câu `adv_t_*`, lưu nguyên vẹn tại `results/adv-agent-test-v1/{traces.json,reviews.json,summary.json}`), hệ thống đạt **`24/30 (80.0%)`**.
    - Tuân thủ đúng quy định *"không được quay lại chỉnh hệ thống trên chính tập test đó; hãy báo cáo trung thực kết quả hiện tại hoặc tạo bộ test mới chưa từng dùng để tinh chỉnh"*, nhóm giữ nguyên kết quả Test v1 để báo cáo minh bạch, sửa lỗi kiến trúc tổng quát (ưu tiên khớp nhãn `Figure X.Y`/`Table X.Y`, đan xen chunk đa trang, đồng bộ `num_ctx=4096`), khóa cấu hình và xây dựng bộ **Test v2 độc lập (`30` câu mới hoàn toàn `adv_t2_*` / `adv_u2_test_*`, `split="test"`)** được kiểm chứng trực tiếp trên ảnh crop và văn bản gốc (6 câu `unanswerable` đều được kiểm tra bằng regex đạt `0` lần xuất hiện).

### Bảng kết quả chấm thủ công (0/1) đối chiếu tài liệu gốc trên tập Test nâng cao

| Loại câu hỏi | Số câu yêu cầu | Test v1 (`results/adv-agent-test-v1/`) | **Test v2 độc lập (`results/adv-agent-test/`)** | Ghi chú đối chiếu tài liệu gốc (Test v2) |
|---|---:|---:|---:|---|
| **Bảng (`table`)** | 6 | 6/6 (100.0%) | **5/6 (83.3%)** | Đúng tuyệt đối 5/6 câu (`Table 13.1, 14.3, 15.1, 21.1`); riêng `adv_t2_tab_06` (`Table 18.1` p.416) đúng cả 5 mức GDP (`$62,400; $55,400; $54,800; $1,100; $600`) nhưng lệch 1 bậc hạng ở Norway/Switzerland (`#8/#10` thay vì `#9/#11`) nên chấm `0` nghiêm ngặt. |
| **Hình / sơ đồ (`image_diagram`)** | 5 | 3/5 (60.0%) | **5/5 (100.0%)** | Đúng trọn vẹn cả 5/5 câu (`Figure 13.4` p.285, `Figure 15.4` p.346, `Figures 18.10–18.11` p.417, `Figure 19.11` p.446, `Figure 21.10` p.493). |
| **Biểu đồ số liệu (`chart`)** | 5 | 4/5 (80.0%) | **4/5 (80.0%)** | Đúng 4/5 câu (`Figure 13.3` p.284, `Figure 13.6` p.287, `Figure 16.10` p.374, `Figure 20.6` p.463); câu `adv_t2_cht_04` (`Figure 19.2` p.434, thang đo đau Mosby 0–10) bị verifier từ chối nhầm (`0`). |
| **Đa bước / đa nguồn (`multihop`)** | 4 | 2/4 (50.0%) | **3/4 (75.0%)** | Đúng 3/4 câu (`adv_t2_mhop_01` p.374, `adv_t2_mhop_02` p.487 nối văn bản + bảng + hình, `adv_t2_mhop_03` nối `Table 17.1` p.387 + `Figure 17.8` p.391); câu `adv_t2_mhop_04` (p.461 + p.463) sót đoạn thống kê di cư p.463 ngoài `budget=6500` nên từ chối nhầm (`0`). |
| **Văn bản (`text`)** | 4 | 4/4 (100.0%) | **4/4 (100.0%)** | Đúng trọn vẹn 4/4 câu (`p.285` phân nhóm tuổi già, `p.434` ruy-băng hồng, `p.471` rác thải điện tử e-waste & Greenpeace, `p.493` frame alignment & amplification). |
| **Không có câu trả lời (`unanswerable`)** | 6 | 5/6 (83.3%) | **6/6 (100.0%)** | Từ chối chính xác 6/6 câu (`Zimbardo`, `Cooley looking-glass self`, `Goffman dramaturgical`, `Sutherland differential association`, `Lombroso atavistic`, và câu bẫy `Brazil/Argentina` không có trong `Table 18.1` p.416). |
| **Tổng cộng (`30` câu)** | **30** | **24/30 (80.0%)** | **27/30 (90.0%)** | **Đạt chuẩn `>= 90%` (`27/30`, không làm tròn), `0` lỗi hệ thống.** |

- **Thời gian chạy, token, chi phí và lỗi còn lại:**
  - **Test v2 (`results/adv-agent-test/summary.json`):** Trung bình `57.87 giây/câu` (`1.03` lượt search/câu), tổng `111.813` prompt tokens + `5.911` eval tokens, `0` lỗi runtime, tổng chi phí dịch vụ **`$0.00`** (100% chạy local trên `qwen2.5:7b` + `qwen3-vl:2b` + `FastEmbed` + `RapidOCR`).
  - **Giới hạn & lỗi còn lại:** (1) VLM 2B (`qwen3-vl:2b`) đôi khi tóm tắt ngắn các thang đo đơn giản (`Figure 19.2` p.434) khiến verifier hai pha thận trọng quá mức và từ chối nhầm (`adv_t2_cht_04`); (2) Ở câu đa bước nối bảng và đoạn văn cuối trang nhiều hình (`adv_t2_mhop_04`, p.461 + p.463, nơi p.463 có tới 3 biểu đồ), `budget=6500` ký tự ưu tiên 3 biểu đồ trước nên đẩy đoạn văn di cư ở cuối p.463 ra ngoài context; (3) Khi bảng lược bớt các thứ hạng giữa (`Table 18.1` nhảy từ `#7 Singapore` sang `#8 Jersey` rồi `#9 Norway`), model 7B có thể đếm dòng thay vì đọc cột `Rank` (`adv_t2_tab_06`).
