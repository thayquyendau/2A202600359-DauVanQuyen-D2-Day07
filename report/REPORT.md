# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Đậu Văn Quyền
**Nhóm:** D2
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity nghĩa là hai vector "chỉ" về cùng một hướng trong không gian đa chiều. Trong NLP, điều này ám chỉ hai đoạn văn bản có sự tương đồng cao về ngữ nghĩa (semantic meaning) hoặc từ vựng, dù độ dài có thể khác nhau.

**Ví dụ HIGH similarity:**
- Sentence A: "Hệ thống trí tuệ nhân tạo phải đảm bảo tính minh bạch."
- Sentence B: "Các mô hình AI cần được công khai rõ ràng cách thức hoạt động."
- Tại sao tương đồng: Cả hai đều nói về tính "transparency/minh bạch" của AI bằng các cách diễn đạt khác nhau.

**Ví dụ LOW similarity:**
- Sentence A: "Hệ thống trí tuệ nhân tạo phải đảm bảo tính minh bạch."
- Sentence B: "Hôm nay trời Hà Nội đổ mưa rất to."
- Tại sao khác: Hai câu thuộc hai chủ đề hoàn toàn khác nhau, không có sự liên quan về ngữ nghĩa.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Bởi vì Cosine Similarity tập trung vào "hướng" (meaning) thay vì "độ dài" (magnitude). Trong văn bản, một đoạn văn dài và một câu ngắn có thể cùng ý nghĩa, Euclidean sẽ thấy chúng xa nhau vì độ dài khác biệt, còn Cosine vẫn thấy chúng gần nhau.


### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Công thức: `num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))`
> Phép tính: `ceil((10,000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.11)`
> *Đáp án:* 23 chunks.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Khi overlap = 100, số chunk tăng lên thành `ceil(9900 / 400) = 25`. Overlap nhiều giúp tránh việc các thông tin quan trọng bị "chẻ đôi" ngay ranh giới giữa 2 chunk, giữ được context tốt hơn cho Retrieval.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Pháp luật Việt Nam (Dự thảo Luật Trí tuệ nhân tạo 2025).

**Tại sao nhóm chọn domain này?**
> Đây là một chủ đề cực hot và có cấu trúc văn bản rất chặt chẽ (Chương, Điều, Khoản). Việc thử nghiệm RAG trên văn bản luật giúp đánh giá chính xác khả năng truy xuất thông tin cụ thể và tính logic của Agent.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | Luật AI - Chương 1: Quy định chung | du-thao-Luat-AI.md | ~340 | {chapter: 1, topic: general} |
| 2 | Luật AI - Chương 2: Phân loại rủi ro | du-thao-Luat-AI.md | ~11,000 | {chapter: 2, topic: risk} |
| 3 | Luật AI - Chương 3: Hạ tầng & Chủ quyền | du-thao-Luat-AI.md | ~7,500 | {chapter: 3, topic: infra} |
| 4 | Luật AI - Chương 4: Hệ sinh thái | du-thao-Luat-AI.md | ~8,000 | {chapter: 4, topic: eco} |
| 5 | Luật AI - Chương 5: Đạo đức & Trách nhiệm | du-thao-Luat-AI.md | ~3,500 | {chapter: 5, topic: ethics} |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `doc_id` | string | `luat_ai_chuong_2` | Dùng để phân biệt các chapter khi delete hoặc update. |
| `chapter` | int | `2` | Giúp filter nhanh các quy định thuộc cùng một chương. |
| `source` | string | `data/file_name.md` | Truy xuất nguồn gốc chính xác của thông tin. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Tôi đã chạy `ChunkingStrategyComparator().compare()` trên Chương 2:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| Chương 2 | FixedSizeChunker (`fixed_size`) | 22 | 491.27 | Medium (bị cắt ngang câu) |
| Chương 2 | SentenceChunker (`by_sentences`) | 44 | 234.05 | High (giữ trọn câu) |
| Chương 2 | RecursiveChunker (`recursive`) | 171 | 59.56 | Very High (chia nhỏ cực chi tiết) |

### Strategy Của Tôi

**Loại:** `RecursiveChunker`

**Mô tả cách hoạt động:**
> Đây là strategy "thông minh" nhất. Nó bắt đầu split bằng separator thô nhất (`\n\n`), nếu đoạn đó vẫn quá dài so với `chunk_size` thì nó mới dùng đến các separator nhỏ hơn như `\n`, rồi đến `. `. Nó hoạt động theo cơ chế đệ quy (recursive) cho đến khi mọi chunk đều đạt kích thước mong muốn.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Văn bản luật có phân cấp rất rõ: Chương > Điều > Khoản. `RecursiveChunker` cho phép mình giữ nguyên các khối "Điều/Khoản" mà không bị xé lẻ vô lý như `FixedSize`, đồng thời đảm bảo không có chunk nào bị quá dài để Model xử lý.

**Code snippet (nếu custom):**
```python
# Sử dụng RecursiveChunker mặc định nhưng với separators tùy chỉnh cho Luật
separators = ["\nChương ", "\nĐiều ", "\n", ". ", " "]
chunker = RecursiveChunker(separators=separators, chunk_size=500)
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| Chương 2 | Sentence (baseline) | 44 | 234.05 | Good (Ok cho ý nhỏ) |
| Chương 2 | **Recursive** | 171 | 59.56 | Excellent (Rất hội tụ) |


### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Đậu Văn Quyền (tôi) | RecursiveChunker | 9/10 | Linh hoạt, chia nhỏ theo separator ưu tiên, hội tụ đúng ý | Chunk hơi nhỏ, tốn nhiều chunk (overhead) |
| Nguyễn Anh Đức | Document-Structure Hybrid | 9/10 | Giữ đúng tên Điều làm metadata/header ở mọi chunk, chất lượng xuất sắc | Cần custom parser riêng cho từng loại văn bản |
| Vũ Duy Linh | FixedSizeChunker | 8/10 | Đơn giản, dễ implement, chunk đều nhau (~497 ký tự) | Câu có thể bị cắt ngang ở phần biên nếu ráp nối không đều |
| Nguyễn Thành Đạt | Custom (Doc Parse + FixedSize) | 8.5/10 | Bảo toàn ngữ cảnh rất tốt, chunk vừa đủ lớn giữ trọn ý | Chunk khá dài (~682 ký tự), có thể chứa nhiều ý lẫn lộn |
| Hoàng Ngọc Anh | DocumentStructureChunker | 9/10 | Tách theo đơn vị điều/mục, giữ nội dung pháp lý đầy đủ | Phụ thuộc vào định dạng file, markdown khác có thể fail |
| Nguyễn Hoàng Việt | Semantic chunking | 9/10 | Linh hoạt, bảo toàn ngữ cảnh theo ý nghĩa | Chi phí cao, tốc độ chậm, chunk count rất lớn (359) |
**Strategy nào tốt nhất cho domain này? Tại sao?**
> Recursive hoặc Doc-structure là tốt nhất. Vì văn bản luật có cấu trúc phân tầng (hierarchy), việc tách theo Chương/Điều giúp Agent luôn biết mình đang đọc quy định nào, không bị chồng chéo dữ liệu.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Sử dụng **Regex Positive Lookbehind** `(?<=[.!?])\s+` để tách câu. Cách này đảm bảo dấu chấm/hỏi vẫn nằm trong câu cũ. Sau đó gom nhóm câu lại cho đến khi đủ số lượng yêu cầu.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Dùng một danh sách các separators theo độ ưu tiên giảm dần. Hàm đệ quy sẽ thử split và nếu phần tử nào vẫn to hơn `chunk_size`, nó sẽ gọi lại chính nó với separator tiếp theo trong danh sách.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Lưu trữ dưới dạng list các dictionaries. Khi search, hệ thống embed query rồi quét toàn bộ kho dữ liệu (exhaustive search), tính **Dot Product** để xếp hạng các đoạn văn bản tương đồng nhất.

**`search_with_filter` + `delete_document`** — approach:
> **Filter trước khi Search**: Lọc các record thỏa mãn điều kiện metadata trước, rồi mới chạy similarity trên tập thu gọn. `delete_document` sẽ tìm và xoá mọi chunk có `doc_id` tương ứng.

### KnowledgeBaseAgent

**`answer`** — approach:
> Sử dụng prompt mẫu chuẩn RAG. Context được lấy từ store -> nhét vào prompt cùng với câu hỏi của user. Giao cho LLM nhiệm vụ tổng hợp và trả lời dựa "strictly" vào context được cung cấp.

### Test Results

======================== 42 passed, 1 warning in 1.08s ========================

**Số tests pass:** 42 / 42


## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "Trí tuệ nhân tạo là hệ thống máy tính." | "AI là công nghệ dựa trên máy móc." | High | 0.85 | Đúng |
| 2 | "Cấm lừa dối người dùng qua AI." | "Hôm nay trời đẹp." | Low | 0.05 | Đúng |
| 3 | "Rủi ro cao gây hại sức khỏe." | "Hệ thống nguy hiểm tới tính mạng." | High | 0.78 | Đúng |
| 4 | "Nhà phát triển thiết kế AI." | "Người làm luật viết dự thảo." | Low | 0.15 | Đúng |
| 5 | "Công khai minh bạch dữ liệu." | "Dữ liệu cần được bảo mật." | High | 0.45 | Tương đối |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Pair 5 là bất ngờ nhất. Tôi nghĩ "công khai minh bạch" và "bảo mật" là đối lập nhau nên score phải rất thấp, nhưng thực tế lại được 0.45. Lý do có lẽ vì cả hai cùng xoay quanh chủ đề "dữ liệu" nên embedding vẫn bắt được sự liên quan ngữ nghĩa dù ý nghĩa hành động ngược nhau. Điều này cho thấy embeddings capture "topic" tốt hơn là "intent/hướng hành động".

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)


| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Định nghĩa "Hệ thống AI rủi ro cao"? | Gây thiệt hại đáng kể tính mạng, sức khỏe, quyền lợi, an ninh quốc gia... |
| 2 | Các hành vi bị nghiêm cấm? | Lợi dụng AI phạm pháp, lừa dối thao túng, gây hại nhóm dễ tổn thương, giả mạo quốc phòng... |
| 3 | Trách nhiệm khi có sự cố nghiêm trọng? | Nhà phát triển: khắc phục, tạm dừng; Bên triển khai: ghi nhận, thông báo, phối hợp. |
| 4 | Có được miễn trừ hoàn toàn quy định? | Không. Chỉ miễn một số nghĩa vụ, không miễn quyền con người, an ninh quốc gia. |
| 5 | Thời gian chuyển tiếp là bao lâu? | 12 tháng kể từ khi Luật có hiệu lực. |


### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | AI rủi ro cao... | Điều 8.1: Thiệt hại đáng kể tính mạng... | 0.3705 | YES | Hệ thống rủi ro cao gây thiệt hại... |
| 2 | Hành vi bị cấm... | Điều 6. Lợi dụng hệ thống AI... | 0.1782 | YES | Các hành vi cấm gồm phạm pháp, lừa dối... |
| 3 | Sự cố nghiêm trọng... | Điều 11. Trách nhiệm xử lý sự cố... | 0.1124 | YES | Nhà phát triển cần khắc phục kỹ thuật... |
| 4 | Miễn trừ thử nghiệm... | Điều 20. Miễn giảm nghĩa vụ một phần... | 0.2132 | YES | Không miễn hoàn toàn, trừ quyền con người... |
| 5 | Thời gian chuyển tiếp... | Điều 35. Thời gian 12 tháng... | 0.3460 | YES | Thời gian là 12 tháng, tiếp tục hoạt động... |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

### Đánh Giá Chất Lượng Retrieval (theo 5 góc nhìn)

**1. Retrieval Precision**
> Top-3 results của cả 5 queries đều chứa chunk liên quan trực tiếp đến câu hỏi (5/5). Tuy nhiên, score nhìn chung còn thấp (cao nhất là 0.37) do mình đang dùng `mock_embed` chứ chưa phải real embedder. Score có phân tách rõ giữa chunk đúng và chunk nhiễu — ví dụ Query 1 top-1 score 0.37 trong khi các chunk không liên quan chỉ khoảng 0.05-0.1.

**2. Chunk Coherence**
> Với `RecursiveChunker`, mỗi chunk giữ được ý khá trọn vẹn vì nó tách theo `\n\n` và `\n` — tức là theo đoạn văn tự nhiên trong văn bản luật. Nhìn chung chunk dễ đọc, không bị cắt ngang giữa câu. Tuy nhiên ở một số Điều dài (như Điều 13 với 7 khoản), chunk bị chia nhỏ quá nên mất đi cái nhìn tổng thể của cả Điều.

**3. Metadata Utility**
> Mình đã gán metadata `doc_id`, `chapter`, `source` cho mỗi chunk. Khi dùng `search_with_filter(metadata_filter={"chapter": 2})` thì kết quả chính xác hơn rõ rệt vì loại bỏ được nhiễu từ các chương không liên quan. Tuy nhiên filter theo `chapter` đôi khi quá chặt — ví dụ câu hỏi về "sự cố nghiêm trọng" liên quan cả Chương 2 (Điều 11) lẫn Chương 5 (trách nhiệm đạo đức), nếu filter chỉ 1 chương sẽ bỏ sót.

**4. Grounding Quality**
> Câu trả lời của Agent luôn dựa trên context được retrieve. Do mình thiết kế prompt với dòng "Hãy trả lời dựa TRỰC TIẾP vào context" nên Agent không bịa thêm. Có thể chỉ rõ chunk nào hỗ trợ câu trả lời — ví dụ Query 1 dựa hoàn toàn vào Chunk chứa Điều 8 Khoản 1. Điểm hạn chế là khi dùng `demo_llm` (mock), câu trả lời chưa thật sự "tổng hợp" được mà chỉ echo lại context.

**5. Data Strategy Impact**
> Bộ tài liệu nhóm chọn (Dự thảo Luật AI) rất phù hợp với benchmark queries vì các câu hỏi đều có câu trả lời nằm rõ ràng trong các Điều cụ thể. Strategy `RecursiveChunker` hợp với domain luật vì tách đúng theo cấu trúc tự nhiên (đoạn, khoản). Tuy nhiên, nếu thêm metadata `dieu_so` (số Điều) thì retrieval sẽ chính xác hơn nữa, thay vì chỉ filter theo `chapter`.

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Tôi học được từ Anh Đức cách kết hợp Document-Structure với logic chia nhỏ (Hybrid) để giữ được tên Điều làm metadata rất chuẩn. Từ Thành Đạt, tôi thấy cách kết hợp Doc Parse + FixedSize giúp chunk lớn vừa đủ (~682 ký tự), bảo toàn ngữ cảnh tốt. Còn Hoàng Việt cho thấy Semantic chunking nhóm câu theo ý nghĩa rất linh hoạt, dù chunk count lớn (359).

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Các nhóm khác dùng `sentence-transformers` bản local cho kết quả retrieval hội tụ hơn hẳn `mock_embed` mặc định, dù tốn tài nguyên hơn nhưng rất đáng giá. Score phân tách giữa chunk đúng và sai rõ ràng hơn nhiều so với mock.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ thêm bước **Metadata enrichment** — gắn thêm `dieu_so` và `tom_tat_y_chinh` cho mỗi chunk. Ngoài ra sẽ dùng `sentence-transformers` thay vì mock embedder để score phản ánh sát hơn chất lượng retrieval thực tế.

---

## Tự Đánh Giá

| Hạng mục | Loại | Điểm tối đa | Điểm tự đánh giá | Giải thích |
|----------|------|-----------|-------------------|-----------|
| Core Implementation (pytest) | Cá nhân | 30 | 30 | 42/42 tests passed |
| My Approach | Cá nhân | 10 | 8 | Giải thích đủ các phần, nhưng chưa đi sâu edge case |
| Competition Results | Cá nhân | 10 | 8 | 5/5 queries tìm đúng chunk, agent trả lời dựa trên context |
| Warm-up | Cá nhân | 5 | 4 | Trả lời đủ cosine + chunking math, ví dụ rõ ràng |
| Similarity Predictions | Cá nhân | 5 | 4 | 4/5 dự đoán đúng, Pair 5 bất ngờ nhưng có giải thích |
| Strategy Design | Nhóm | 15 | 13 | Giải thích strategy + so sánh với baseline và 5 thành viên |
| Document Set Quality | Nhóm | 10 | 8 | 9 file luật, metadata rõ, nhưng chưa có field `dieu_so` |
| Retrieval Quality | Nhóm | 10 | 8 | 5/5 top-3 đúng, nhưng dùng mock embedder nên score thấp |
| Demo | Nhóm | 5 | 4 | Trình bày strategy + so sánh, rút được bài học |
| **Tổng** | | **100** | **87 / 100** | |
