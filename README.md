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

Cách **thêm kiến thức mới mà không làm mất kiến thức gốc**: base model đóng băng hoàn toàn, chỉ train LoRA adapter rank thấp với lr nhỏ (`1e-5 → 2e-5`), 1-2 epoch, cosine schedule. Kết quả là một adapter nhỏ (~vài trăm MB) lưu trong `models/qwen25vl-3b-vi-hwr-lora/`, không đụng tới weight gốc.

## Cấu trúc

```
configs/configs.py        # toàn bộ config (dataset, LoRA, training)
src/data/dataset.py       # load + format dataset thành chat template Qwen
src/data/collator.py      # chỉ tính loss trên phần assistant
src/model/load.py         # load 4-bit + gắn LoRA
src/train/trainer.py      # Trainer + lưu adapter
src/infer/predict.py      # inference OCR
scripts/train_qlora.py    # entry point train
scripts/eval_ocr.py       # OCR 1 ảnh / đánh giá CER-WER
```

## Cài đặt

```bash
pip install -r requirements.txt
```

> **Windows:** nên chạy trong **WSL2** (hoặc GPU cloud như AutoDL/Colab Pro) vì `bitsandbytes` + CUDA trên Windows hay lỗi vặt, `flash_attn` không cài được. Nhớ set `HF_TOKEN` trong `.env.dev`. VRAM cần ~16 GB (RTX 4090 24GB là ổn).

## Chạy 1 lệnh

**Trên Colab (khuyến nghị cho 3B):**
1. Upload toàn bộ project vào Colab (kéo-thả vào `/content/`).
2. Mở terminal (hoặc 1 cell) — truyền token luôn, script tự tạo `.env.dev`:
```bash
!chmod +x run_train.sh && ./run_train.sh hf_xxxxx --max-samples 100   # smoke test
!./run_train.sh hf_xxxxx                                               # train đầy đủ
```
Hoặc bỏ qua token nếu đã upload sẵn `.env.dev`. Lấy token tại https://huggingface.co/settings/tokens.

**Trên Windows / Linux local:**
```bash
run_train.bat   # Windows (3B cần GPU ~8GB+; 7B thì cần 16GB+)
./run_train.sh  # Linux / WSL2
```

Script tự cài dependencies + kiểm tra GPU + đọc `HF_TOKEN` từ `.env.dev`, rồi gọi `train_qlora.py`. Mọi flag truyền thẳng qua được: `./run_train.sh --epochs 2 --lr 1e-5`.

## Chạy chi tiết

```bash
# Train (thử nhanh với --max-samples trước khi train đủ)
python scripts/train_qlora.py --max-samples 100

# Train đầy đủ trên dataset
python scripts/train_qlora.py

# Ghi đè config nhanh từ CLI
python scripts/train_qlora.py --epochs 2 --lr 1e-5 --lora-r 16 --lora-alpha 32

# OCR 1 ảnh bằng adapter vừa train
python scripts/eval_ocr.py --image path/to/anh.jpg

# OCR bằng adapter load trực tiếp từ Hub (không cần tải về)
python scripts/eval_ocr.py --image path/to/anh.jpg --adapter tranhuy67896262/qwen25vl-3b-vi-hwr-lora

# Đánh giá CER/WER trên test split
python scripts/eval_ocr.py --num-test 200

# Push adapter lên Hub (repo sẽ được tạo mới nếu chưa tồn tại)
python scripts/train_qlora.py --push --hub-repo <owner>/qwen25vl-3b-vi-hwr-lora

python scripts/train_qlora.py --push --hub-repo tranhuy67896262/qwen25vl-3b-vi-hwr-lora
```
> Dataset mặc định là `tranhuy67896262/Viet-Handwriting-OCR-v2-ds` (public). Nếu chưa có, script fallback sang dataset gốc gated `5CD-AI/Viet-Handwriting-OCR-v2` (phải đồng ý điều khoản trên HF). Có thể đổi bằng `--dataset <owner>/<repo>`.
> 
```angular2html
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```