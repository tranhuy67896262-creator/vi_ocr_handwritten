# Hướng dẫn lấy Access Token Hugging Face

Để sử dụng các dịch vụ và API của Hugging Face, bạn cần cung cấp Access Token để xác thực. Dưới đây là các bước chi tiết để tạo và lấy token.

## Các bước thực hiện

1. **Đăng nhập/Đăng ký:** Truy cập [Hugging Face](https://huggingface.co/) và đăng nhập vào tài khoản của bạn.
2. **Truy cập Settings:** Vào trực tiếp trang quản lý Access Tokens qua đường dẫn sau:
   👉 [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. **Tạo Token mới:** Nhấn vào nút **New token**.
4. **Cấu hình Token:**
   - **Name:** Đặt tên gợi nhớ cho token (ví dụ: `my-project-api`).
   - **Role:** Chọn mức phân quyền:
     - `Read`: Cấp quyền tải xuống (download) models và datasets (khuyên dùng khi chỉ cần gọi API).
     - `Write`: Cấp quyền tải lên (push) và thay đổi models/datasets.
5. **Hoàn tất:** Nhấn **Generate a token**.
6. **Sao chép:** Nhấn vào biểu tượng Copy bên cạnh chuỗi token (thường bắt đầu bằng `hf_...`) để sử dụng.

> **⚠️ Lưu ý bảo mật:**
> - Tuyệt đối không chia sẻ token này cho người khác.
> - Không đẩy (commit) token trực tiếp lên các public repository như GitHub, GitLab.
> - Khuyến nghị lưu trữ token thông qua biến môi trường (ví dụ: file `.env`) hoặc trình quản lý Secret chuyên dụng.

---

# Pipeline Fine-tune Qwen2.5-VL-3B (chữ viết tay Việt)

Cách **thêm kiến thức mới mà không làm mất kiến thức gốc**: base model đóng băng hoàn toàn, chỉ train LoRA adapter rank thấp với lr nhỏ (`1e-5 → 2e-5`), 1-2 epoch, cosine schedule + KL-regularization. Kết quả là một adapter nhỏ (~vài trăm MB) lưu trong `models/qwen25vl-3b-vi-hwr-lora/`, không đụng tới weight gốc. Mặc định train **3B** (vừa T4 Colab); đổi sang 7B bằng `--model` hoặc dropdown trên UI.

## Cấu trúc

```
configs/configs.py        # toàn bộ config (dataset, LoRA, training) — nguồn sự thật duy nhất
src/datasets/dataset.py    # load + tự dò cột + format chat template Qwen (chuẩn hóa ảnh A4)
src/datasets/collator.py   # chỉ tính loss trên phần assistant
src/modeling/load.py       # load 4-bit + gắn LoRA
src/train/trainer.py      # Trainer + lưu adapter (+ training_metadata.json)
src/train/kl_trainer.py   # KL-regularization chống quên kiến thức gốc
src/infer/predict.py      # inference OCR: ảnh lẻ / PDF nhiều trang / Word .docx
src/utils/image.py        # chuẩn hóa ảnh khổ A4
scripts/train_qlora.py    # entry point train
scripts/eval_ocr.py       # OCR file + đánh giá CER/WER
scripts/export_merged.py  # merge LoRA vào base
scripts/export_gguf.sh    # convert GGUF (Linux/Colab)
scripts/ui.py             # UI Gradio 5 tab (Fine-tune/OCR/Eval/Export/Settings)
run_train.sh / run_train.bat  # wrapper setup + chạy (Linux / Windows)
```

## Cài đặt

```bash
pip install -r requirements.txt
```

> **Windows:** train thật nên chạy trong **WSL2** (hoặc GPU cloud như Colab) vì `bitsandbytes` + CUDA trên Windows hay lỗi vặt. Nhớ set `HF_TOKEN` trong `.env.dev`. VRAM: 3B khoảng ~8GB+ (T4 15GB ổn); 7B cần 16GB+.

## Chạy nhanh nhất: UI Gradio (mặc định)

```bash
run_train.bat   # Windows — mở UI ngay
./run_train.sh  # Linux / WSL2 / Colab — mở UI ngay
```

Mặc định wrapper chỉ cài tối thiểu (`gradio`, `python-dotenv`, `pymupdf`, `python-docx`) rồi mở UI — chưa tải torch/data. 5 tab: **Fine-tune** (dataset, model 3B/7B, cỡ data) | **OCR** (ảnh lẻ + PDF scan + Word .docx) | **Eval CER/WER** | **Export** (merge, GGUF, push Hub) | **Settings** (lưu HF_TOKEN, kiểm tra model đã tải chưa). Nút Fine-tune trong UI cần đủ deps — chạy wrapper với `--train` 1 lần để cài full.

Muốn train ngay từ lệnh (không qua UI): thêm cờ `--train`:

```bash
./run_train.sh --train --max-samples 100   # smoke test 100 ảnh
./run_train.sh --train                     # train đầy đủ
./run_train.sh --train --eval              # train xong eval 100 mẫu
```

**Trên Colab (khuyến nghị cho 3B):**
1. Upload toàn bộ project vào Colab (kéo-thả vào `/content/`).
2. Mở terminal (hoặc 1 cell) — truyền token luôn, script tự tạo `.env.dev`:
```bash
!chmod +x run_train.sh && ./run_train.sh hf_xxxxx --train --max-samples 100
```
Hoặc bỏ qua token nếu đã upload sẵn `.env.dev` / dán token ở tab Settings. Lấy token tại https://huggingface.co/settings/tokens.

Script tự cài dependencies + kiểm tra GPU + đọc `HF_TOKEN` từ `.env.dev`. Mọi flag train truyền thẳng qua được: `./run_train.sh --train --epochs 2 --lr 1e-5`.

## Chạy chi tiết (terminal)

```bash
# Train (thử nhanh với --max-samples trước khi train đủ)
python scripts/train_qlora.py --max-samples 100

# Train đầy đủ / đổi model
python scripts/train_qlora.py
python scripts/train_qlora.py --model Qwen/Qwen2.5-VL-7B-Instruct

# Run dài (Colab hay đứt): lưu checkpoint dày + resume khi chạy lại
python scripts/train_qlora.py --save-steps 100
python scripts/train_qlora.py --resume            # tiếp tục từ checkpoint mới nhất
python scripts/train_qlora.py --resume --save-steps 100

# Ghi đè config nhanh từ CLI
python scripts/train_qlora.py --epochs 2 --lr 1e-5 --lora-r 16 --lora-alpha 32

# OCR ảnh / PDF / Word bằng adapter (base phải cùng họ adapter)
python scripts/eval_ocr.py --image path/to/anh.jpg
python scripts/eval_ocr.py --image path/to/scan.pdf
python scripts/eval_ocr.py --image path/to/file.docx
python scripts/eval_ocr.py --image path/to/anh.jpg --adapter <owner>/qwen25vl-3b-vi-hwr-lora --model Qwen/Qwen2.5-VL-3B-Instruct

# Đánh giá CER/WER trên test split
python scripts/eval_ocr.py --num-test 200

# Export full model (merge LoRA vào base, chạy độc lập) — 3B cần ~8GB+ RAM/VRAM
python scripts/export_merged.py
#   --adapter <path-or-repo-id>   --model <base>   --output <dir>

# Export GGUF (llama.cpp/Ollama) — Linux/Colab, sau export_merged
./scripts/export_gguf.sh            # convert thang Q4_K_M (bo f16 trung gian) + convert mmproj + sinh Modelfile vào models/gguf/
#   QUANT=Q4_K_M ./scripts/export_gguf.sh   (chọn loại quantize)
#   LLAMA_CPP_DIR=... ./scripts/export_gguf.sh   (nếu llama.cpp chỗ khác)

# Chạy trên Ollama (cần CẢ 2 file: model + mmproj vision)
cd models/gguf   # đã có sẵn Modelfile (FROM text + ADAPTER mmproj) do export_gguf.sh sinh
ollama create qwen25vl-3b-vi-hwr -f Modelfile
ollama run qwen25vl-3b-vi-hwr "Đọc chữ trong ảnh" -- /path/to/anh.jpg
ollama push <owner>/qwen25vl-3b-vi-hwr
#   Thiếu mmproj → lỗi 500 "image input is not supported ... provide the mmproj".
#   Ollama ra chữ linh tinh nhưng llama.cpp đọc đúng → lỗi phía Ollama, test bằng:
#   llama-mtmd-cli -m model.gguf --mmproj mmproj.gguf --image anh.jpg -p "..." -n 256

# Push adapter lên Hub (repo tự tạo nếu chưa có)
python scripts/train_qlora.py --push --hub-repo <owner>/qwen25vl-3b-vi-hwr-lora
```
> Dataset mặc định: source gốc gated `5CD-AI/Viet-Handwriting-OCR-v2` (phải accept điều khoản trên HF). Đổi bằng `--dataset <owner>/<repo>`.

## Chạy trên A100 / H100 (Colab Pro)

```bash
!pip install -q flash-attn   # 1 lần mỗi runtime — code tự dùng khi có, fallback sdpa khi không
```

```bash
# A100 40GB: batch lớn hơn + model 7B
./run_train.sh --train --model Qwen/Qwen2.5-VL-7B-Instruct --batch-size 4 --max-samples 100

# H100: tương tự, có thể thử --batch-size 8
./run_train.sh --train --model Qwen/Qwen2.5-VL-7B-Instruct --batch-size 8 --max-samples 100
```

`ATTN_IMPLEMENTATION="auto"` trong config (flash_attention_2 nếu import được, sdpa nếu không) nên không cần sửa code. Batch size cứ tăng dần tới khi gần đầy VRAM rồi lùi 1 nấc.

## Ghi chú quan trọng

- Ảnh train/inference tự chuẩn hóa khổ A4 (pad trắng) và giới hạn pixel để vừa context 1024 token (`A4_STANDARDIZE`, `MAX_PIXELS` trong `configs/configs.py`).
- `models/`, `data/`, `.hf_cache/`, `.env.dev` không commit (git-ignored). Cache HF nằm trong project nên chạy lại không tải lại.
- UI offline (không tạo link public): `GRADIO_SHARE=0`.
- Cần torch CUDA thủ công: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128` (wrapper thường tự lo).