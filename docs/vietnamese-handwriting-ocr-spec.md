# Spec: Vietnamese Handwriting OCR Pipeline

## Mục tiêu
Xây dựng pipeline nhận dạng chữ viết tay tiếng Việt từ ảnh scan/chụp, độ chính xác cao hơn việc feed thẳng cả ảnh vào 1 VLM duy nhất, bằng cách kết hợp text detection + VLM recognition + dictionary-based correction.

## Kiến trúc tổng thể (Hybrid Pipeline)

```
Ảnh đầu vào
   │
   ▼
[1] Text Detection (DBNet / PP-OCR)
   │  → xuất ra danh sách polygon/bounding box cho từng dòng chữ
   ▼
[2] Crop từng dòng theo box, chuẩn hóa (deskew, resize nếu cần)
   │
   ▼
[3] Recognition (Qwen2.5-VL 7B) — chạy trên từng dòng đã crop, KHÔNG feed cả ảnh gốc
   │  → text thô cho mỗi dòng + optional confidence/candidate list
   ▼
[4] Post-processing: Dictionary-based correction
   │  → so khớp từng từ với từ điển tiếng Việt, sửa từ sai bằng edit-distance
   │  → ưu tiên sửa lỗi dấu thanh/dấu phụ (đặc thù lỗi OCR tiếng Việt)
   ▼
[5] Ghép lại thành văn bản hoàn chỉnh theo đúng thứ tự dòng
```

## Chi tiết từng bước

### Bước 1 — Text Detection
- **Công cụ**: DBNet (Differentiable Binarization) — có sẵn trong PaddleOCR (`PP-OCRv4` hoặc `PP-OCRv5` detection model)
- **Input**: ảnh gốc (đã qua tiền xử lý: binarization nhẹ, deskew nếu ảnh nghiêng nhiều)
- **Output**: list các polygon bao quanh từng dòng chữ viết tay
- **Lý do dùng bước này**: giảm nhiễu ngữ cảnh cho VLM ở bước sau — feed từng dòng nhỏ chính xác hơn nhiều so với feed cả trang giấy

### Bước 2 — Crop & chuẩn hóa
- Crop theo polygon, có thể warp về hình chữ nhật nếu dòng bị cong/nghiêng
- Resize về chiều cao chuẩn (ví dụ 64px hoặc theo yêu cầu input của Qwen-VL) để tăng tốc và ổn định chất lượng

### Bước 3 — Recognition bằng Qwen2.5-VL 7B
- Chạy qua Ollama (đã có sẵn trong stack hiện tại) nếu Ollama hỗ trợ multimodal cho model này, hoặc serve riêng qua vLLM/Transformers nếu cần throughput cao hơn
- **Prompt gợi ý**:
  > "Trích xuất chính xác toàn bộ văn bản viết tay trong ảnh này. Giữ nguyên dấu câu và dấu thanh tiếng Việt. Chỉ trả về văn bản, không thêm giải thích."
- Cân nhắc sinh **top-k candidate** cho các từ model không chắc chắn (nếu framework hỗ trợ) để bước 4 có nhiều lựa chọn hơn khi sửa lỗi
- Ghi chú giới hạn: bản 7B dễ sai dấu thanh với ảnh chất lượng thấp — nếu cần độ chính xác cao cho production, cân nhắc fine-tune LoRA trên dataset chữ viết tay tiếng Việt riêng

### Bước 4 — Dictionary-based Post-processing
- **Nguồn từ điển đề xuất**:
  - [underthesea Vietnamese Dictionary](https://github.com/undertheseanlp/underthesea) — ưu tiên dùng vì đầy đủ và có kèm tokenizer
  - `hunspell-vi` (.dic/.aff) — dùng qua `pyhunspell` cho spell-check nhanh
  - Bổ sung từ điển domain riêng: build từ chính các văn bản đã OCR đúng trước đó trong hệ thống, để cover thuật ngữ đặc thù (ví dụ văn bản hành chính)
- **Thuật toán sửa lỗi**:
  1. Tokenize câu OCR ra thành từng từ (dùng tokenizer tiếng Việt, ví dụ `underthesea` word segmentation, vì tiếng Việt có từ ghép)
  2. Với mỗi từ không có trong từ điển: tìm từ gần nhất bằng **Levenshtein distance**
  3. Ưu tiên khoảng cách tính theo **lỗi dấu thanh/dấu phụ** trước (vì đây là lỗi phổ biến nhất của Qwen-VL với tiếng Việt viết tay), sau đó mới đến lỗi phụ âm/nguyên âm
  4. Nếu có nhiều ứng viên gần bằng nhau, dùng thêm ngữ cảnh câu (n-gram LM đơn giản hoặc gọi lại LLM để chọn) để quyết định
- **Rủi ro cần xử lý trong code**: không "sửa nhầm" tên riêng, từ viết tắt, số liệu — nên có whitelist hoặc bỏ qua correction cho token là số/tên viết hoa toàn bộ

### Bước 5 — Ghép kết quả
- Ghép lại theo thứ tự dòng ban đầu (giữ metadata vị trí từ bước 1 để không bị đảo dòng)
- Output cuối: văn bản hoàn chỉnh + optionally kèm confidence score / vị trí các chỗ đã được sửa để review thủ công

## Stack đề xuất
| Thành phần | Công cụ |
|---|---|
| Detection | PaddleOCR (PP-OCR, dùng DBNet) |
| Recognition | Qwen2.5-VL 7B qua Ollama |
| Tokenizer tiếng Việt | underthesea |
| Từ điển | underthesea dictionary + hunspell-vi + domain dictionary tự build |
| Fuzzy matching | Levenshtein / python-Levenshtein hoặc rapidfuzz |

## Việc cần làm tiếp (để AI code triển khai)
1. Setup PaddleOCR detection-only mode (tắt recognition built-in, chỉ lấy box)
2. Viết hàm crop + chuẩn hóa ảnh theo polygon
3. Viết wrapper gọi Qwen2.5-VL 7B qua Ollama API cho từng ảnh dòng đã crop
4. Load từ điển tiếng Việt (underthesea + hunspell-vi), build thành set/tree để tra nhanh (ví dụ Trie hoặc BK-tree cho fuzzy search hiệu quả hơn linear scan)
5. Viết module correction: tokenize → check từ điển → fuzzy match → chọn ứng viên
6. Ghép pipeline end-to-end, benchmark trên tập ảnh chữ viết tay thật để đo CER/WER trước và sau khi có bước dictionary correction
