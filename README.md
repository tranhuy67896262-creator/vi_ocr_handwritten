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

# Pipeline Fine-tune Qwen2.5-VL (chữ viết tay Việt)

Cách **thêm kiến thức mới mà không làm mất kiến thức gốc**: base model đóng băng hoàn toàn, chỉ train LoRA adapter rank thấp với lr nhỏ (`1e-5 → 2e-5`), 1-2 epoch, cosine schedule + KL-regularization. Adapter được lưu riêng, không đụng tới weight gốc.

- **3B:** phù hợp GPU Colab khoảng 15GB, nên bắt đầu với batch `1`.
- **7B:** nên chạy QLoRA trên A100 40GB trở lên hoặc H100, bắt đầu với batch `2`.
- Model mặc định trong config là 3B; truyền `--model` để đổi model.

## Cấu trúc

```
configs/configs.py        # toàn bộ config (dataset, LoRA, training) — nguồn sự thật duy nhất
src/datasets/dataset.py    # load + tự dò cột + format chat template Qwen (ảnh giữ nguyên gốc)
src/datasets/collator.py   # chỉ tính loss trên phần assistant
src/modeling/load.py       # load 4-bit + gắn LoRA
src/train/trainer.py      # Trainer + lưu adapter (+ training_metadata.json)
src/train/kl_trainer.py   # KL-regularization chống quên kiến thức gốc
src/infer/predict.py      # inference OCR: ảnh lẻ / PDF nhiều trang / Word .docx
src/utils/image.py        # split_strips: chẻ trang lớn thành lát ngang cho OCR PDF/DOCX
scripts/train.py    # entry point train (LoRA: QLoRA 4-bit hoặc bf16)
scripts/eval_ocr.py       # OCR file + đánh giá CER/WER
scripts/export_merged.py  # merge LoRA vào base
scripts/export_gguf.sh    # convert GGUF (Linux/Colab)
scripts/ui.py             # UI Gradio 5 tab (Fine-tune/OCR/Eval/Export/Settings)
run_train.sh / run_train.bat  # wrapper setup + chạy (Linux / Windows)
pipeline_smoke_10.sh      # train 10 ảnh + export + import/test Ollama
pipeline_train_20k.sh     # train tối đa 5k mặc định, chỉ chạy sau smoke test thành công
import_models_to_ollama.sh # import các GGUF đã export vào Ollama
```

## Cài đặt

```bash
pip install -r requirements.txt
```

> **Windows:** train thật nên chạy trong **WSL2** (hoặc GPU cloud như Colab) vì `bitsandbytes` + CUDA trên Windows hay lỗi vặt. Nhớ set `HF_TOKEN` trong `.env.dev`. QLoRA 4-bit: 3B phù hợp GPU 15GB; 7B nên dùng A100/H100. Nếu `bitsandbytes` lỗi, code fallback sang full-precision và có thể OOM.

## Chạy nhanh nhất: UI Gradio (mặc định)

```bash
run_train.bat   # Windows — mở UI ngay
./run_train.sh  # Linux / WSL2 / Colab — mở UI ngay
```

Mặc định wrapper chỉ cài tối thiểu (`gradio`, `python-dotenv`, `pymupdf`, `python-docx`) rồi mở UI — chưa tải torch/data. 5 tab: **Fine-tune** (dataset, model 3B/7B, cỡ data) | **OCR** (ảnh lẻ + PDF scan + Word .docx) | **Eval CER/WER** | **Export** (merge, GGUF, push Hub) | **Settings** (lưu HF_TOKEN, kiểm tra model đã tải chưa). Nút Fine-tune trong UI cần đủ deps — chạy wrapper với `--train` 1 lần để cài full.

Muốn train ngay từ lệnh (không qua UI): thêm cờ `--train`:

```bash
./run_train.sh --train --max-samples 100   # smoke test nhanh
./run_train.sh --train                     # train theo toàn bộ dataset
```

**Trên Colab với 3B và GPU 15GB:**
1. Upload toàn bộ project vào Colab (kéo-thả vào `/content/`).
2. Mở terminal (hoặc 1 cell) — truyền token luôn, script tự tạo `.env.dev`:
```bash
!chmod +x run_train.sh && ./run_train.sh hf_xxxxx --train --model Qwen/Qwen2.5-VL-3B-Instruct --batch-size 1 --max-samples 100
```
Hoặc bỏ qua token nếu đã upload sẵn `.env.dev` / dán token ở tab Settings. Lấy token tại https://huggingface.co/settings/tokens.

Script tự cài dependencies + kiểm tra GPU + đọc `HF_TOKEN` từ `.env.dev`. Mọi flag train truyền thẳng qua được: `./run_train.sh --train --epochs 2 --lr 1e-5`.

**Pipeline kiểm tra đầy đủ cho 7B:** các script ở thư mục gốc được thiết kế cho `Qwen/Qwen2.5-VL-7B-Instruct` và sẽ dừng nếu thiếu adapter, GGUF, `mmproj`, Ollama hoặc output OCR.

```bash
# A100 40GB: bắt đầu batch 2, có thể tăng lên 4 nếu còn VRAM
BATCH_SIZE=2 bash pipeline_smoke_10.sh hf_xxxxx

# Chỉ chạy sau khi smoke test tạo marker thành công
BATCH_SIZE=2 bash pipeline_train_20k.sh hf_xxxxx

# Import lại các model GGUF đã tạo (smoke và 20k nếu tồn tại)
bash import_models_to_ollama.sh
```

Smoke pipeline thực hiện: train 10 ảnh → OCR bằng adapter HF → merge model → convert GGUF + `mmproj` → import Ollama → chạy OCR ảnh test. Pipeline 20k chỉ chạy khi smoke test thành công.

## Quy trình chạy an toàn: bắt lỗi sớm (đúc kết từ vụ loss 0.0)

Collator từng đo `prompt_len` bằng tokenizer text-only, sót token ảnh của prompt vào labels → loss `0.0` suốt 800 step mà không báo lỗi. Đã sửa (`_mask_prompt` đo trong không gian multimodal + log batch đầu). Từ nay mỗi lần train đi theo 4 bước:

**Bước 0 — Check mask tĩnh (2 phút, trước khi train):**
```bash
cat > /tmp/diag_labels.py <<'EOF'
import sys; sys.path.insert(0, ".")
from transformers import AutoProcessor
from configs.configs import Configs
from src.datasets.dataset import build_train_eval_datasets
from src.datasets.collator import DataCollatorForQwenVL
cfg = Configs()
proc = AutoProcessor.from_pretrained(cfg.MODEL_NAME, trust_remote_code=True)
train_ds, _ = build_train_eval_datasets(cfg, max_samples=8)
col = DataCollatorForQwenVL(proc, max_length=cfg.MAX_SEQ_LEN, min_pixels=cfg.MIN_PIXELS, max_pixels=cfg.MAX_PIXELS)
batch = col([train_ds[i] for i in range(4)])
for i in range(4):
    total = len(batch["input_ids"][i]); valid = int((batch["labels"][i] != -100).sum())
    print(f"sample {i}: seq_len={total} valid_labels={valid} ({100*valid/total:.1f}%)")
print("valid decode:", proc.tokenizer.decode(batch["labels"][0][batch['labels'][0] != -100][:60]))
EOF
python /tmp/diag_labels.py
```
- Đạt: `valid decode` ra chữ Việt, valid ~5–30%.
- Hỏng: ra `<|image_pad|>` hoặc valid 0%/>80% → đừng train.

**Bước 1 — Smoke 100 mẫu**, nhìn dòng đầu tiên sau `Map: 100%`:
```bash
./run_train.sh --train --max-samples 100
```
```
[collator] batch đầu: seq_len=386 valid=45 (11.7%) | labels decode: <chữ đáp án...>
```
Decode ra chữ Việt mới cho chạy tiếp. Dòng này còn nằm trong `models/training.log` (`grep collator models/training.log`).

**Bước 2 — Check loss ở step 50–100:**
```bash
python -c "import json,glob; p=sorted(glob.glob('models/checkpoints/<run>/checkpoint-*/trainer_state.json'))[-1]; d=json.load(open(p)); [print(e) for e in d['log_history'][-3:]]"
```
- Đạt: loss dương giảm dần, `grad_norm` 1–30.
- Hỏng (`loss: 0.0` tuyệt đối): Ctrl+C ngay, khỏi tốn GPU.

**Bước 2b — Eval loss tự động mỗi 100 step:** train đủ data (có eval split nội bộ 119 mẫu) thì Trainer tự chạy eval loss cùng nhịp checkpoint — tìm trong log dòng `{'eval_loss': ..., 'epoch': ...}`. `eval_loss` giảm dần = đi đúng hướng; tăng dần hoặc `nan` = dừng xem lại.

**Check CER giữa chừng (máy khác, không đụng GPU train):** nhờ `--push-every-save`, snapshot mới nhất luôn nằm trên Hub — mở máy khác eval CER ngay khi train đang chạy:
```bash
python scripts/eval_ocr.py --model Qwen/Qwen2.5-VL-7B-Instruct \
  --adapter <owner>/qwen25vl-7b-vi-hwr-lora --adapter-revision stage-full \
  --num-test 100 --run-name mid-500
```
So trend CER theo mốc: base → step ~500 → ~1500 → ~3000 → xong. CER giảm dần = đúng hướng; đứng yên hoặc tệ hơn base tới step ~1000–1500 = dừng, sai hướng.

**Bước 3 — Check chống mất ở step 100:** `ls models/checkpoints/<run>/` có `checkpoint-100` + nhánh Hub có commit mới (`--push-every-save` đang làm việc).

**Full 7B trên Modal L40S (48GB):**
```bash
./run_train.sh --train \
  --model Qwen/Qwen2.5-VL-7B-Instruct --no-4bit \
  --batch-size 4 --gradient-accumulation-steps 4 --lr 2e-5 --epochs 1 \
  --save-steps 100 --push --push-every-save \
  --hub-repo <owner>/qwen25vl-7b-vi-hwr-lora \
  --hub-revision stage-full --run-name full2
```
- Giữ batch 4 (đang dùng ~35GB; batch 8 OOM). ~3700 step, ~8–9h.
- bf16 (`--no-4bit`) nhanh hơn QLoRA từng step (không dequantize); mặc định không cờ là QLoRA (`USE_4BIT=True`).

**Thang đo chất lượng: smoke-100 → 5k → full:**
- smoke-100 (vài phút): check kỹ thuật (mask đúng, loss ≠ 0, push chạy).
- 5k (~45 phút, ~312 step): check chất lượng — đủ rẻ để thử, đủ lớn để CER có ý nghĩa. CER tốt hơn base rõ rệt thì full mới đáng tiền; còn tệ thì dừng xem lại trước khi đốt 9h.
```bash
./run_train.sh --train \
  --model Qwen/Qwen2.5-VL-7B-Instruct --no-4bit \
  --max-samples 5000 --batch-size 4 --gradient-accumulation-steps 4 \
  --lr 2e-5 --epochs 1 --save-steps 100 \
  --push --push-every-save \
  --hub-repo <owner>/qwen25vl-7b-vi-hwr-lora \
  --hub-revision stage-5k --run-name run-5k
```
So base vs adapter 5k trên cùng 200 mẫu test:
```bash
python scripts/eval_ocr.py --model Qwen/Qwen2.5-VL-7B-Instruct --num-test 200
python scripts/eval_ocr.py --model Qwen/Qwen2.5-VL-7B-Instruct --adapter <owner>/qwen25vl-7b-vi-hwr-lora --adapter-revision stage-5k --num-test 200
```
Lưu ý: `--max-samples 5000` lấy 5000 mẫu đầu (không shuffle) — đủ để validate, chốt cuối cùng vẫn bằng bản full + eval test split.

> ⚠️ Adapter train trước fix collator (triệu chứng loss `0.0`) là rác — đừng eval chúng để kết luận chất lượng. Chạy lại dùng `--run-name` mới để khỏi lẫn checkpoint cũ.

## Chạy chi tiết (terminal)

```bash
# Train (thử nhanh với --max-samples trước khi train đủ)
python scripts/train.py --max-samples 100

# Train đầy đủ / đổi model
python scripts/train.py
python scripts/train.py --model Qwen/Qwen2.5-VL-7B-Instruct

# Run dài (Colab hay đứt): lưu checkpoint dày + resume khi chạy lại
python scripts/train.py --save-steps 100
python scripts/train.py --resume            # tiếp tục từ checkpoint mới nhất (local)
python scripts/train.py --resume --save-steps 100

# Ghi đè config nhanh từ CLI
python scripts/train.py --epochs 2 --lr 1e-5 --lora-r 16 --lora-alpha 32

# OCR ảnh / PDF / Word bằng adapter (base phải cùng họ adapter)
python scripts/eval_ocr.py --image path/to/anh.jpg
python scripts/eval_ocr.py --image path/to/scan.pdf
python scripts/eval_ocr.py --image path/to/file.docx
python scripts/eval_ocr.py --image path/to/anh.jpg --adapter <owner>/qwen25vl-3b-vi-hwr-lora --model Qwen/Qwen2.5-VL-3B-Instruct

# Đánh giá CER/WER trên test split
python scripts/eval_ocr.py --num-test 200
python scripts/eval_ocr.py --model Qwen/Qwen2.5-VL-7B-Instruct \
  --adapter models/qwen25vl-7b-vi-hwr-lora --num-test 200

# Export full model → GGUF → Ollama (Linux/Modal, merge 7B cần GPU ~28GB)
# Bước 1 — merge LoRA vào base:
python scripts/export_merged.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --adapter <path-adapter-local-hoac-repo-id> \
  --output models/qwen25vl-7b-vi-hwr-lora-merged
# Bước 2 — convert GGUF + mmproj + Modelfile (tự clone/build llama.cpp nếu chưa có):
./scripts/export_gguf.sh models/qwen25vl-7b-vi-hwr-lora-merged models/gguf
#   ra models/gguf/: *-Q6_K.gguf (text) + mmproj-*.gguf (vision, BẮT BUỘC) + Modelfile
#   QUANT=Q4_K_M ./scripts/export_gguf.sh ...   (đổi quantize, mặc định Q6_K)
#   OLLAMA_SYSTEM="..." ...                     (đổi system prompt OCR)
#   LLAMA_CPP_DIR=... ./scripts/export_gguf.sh  (nếu llama.cpp chỗ khác)
# Bước 3 — import vào Ollama (cần CẢ 2 file: model + mmproj vision):
cd models/gguf   # đã có sẵn Modelfile (FROM text + ADAPTER mmproj + SYSTEM OCR) do export_gguf.sh sinh
ollama create qwen25vl-7b-vi-hwr -f Modelfile
#   hoặc bấm nút import trong tab Export của UI (tự chạy lệnh trên, cần ollama serve đang chạy)
ollama run qwen25vl-7b-vi-hwr "Đọc chữ trong ảnh" -- /path/to/anh.jpg
ollama push <owner>/qwen25vl-7b-vi-hwr
#   Thiếu mmproj → lỗi 500 "image input is not supported ... provide the mmproj".
#   Ollama ra chữ linh tinh nhưng llama.cpp đọc đúng → lỗi phía Ollama, test bằng:
#   llama-mtmd-cli -m model.gguf --mmproj mmproj.gguf --image anh.jpg -p "..." -n 256

# Push adapter lên Hub (repo tự tạo nếu chưa có)
python scripts/train.py --push --hub-repo <owner>/qwen25vl-3b-vi-hwr-lora
```

## Train tiếp tục trên máy khác (Modal → Colab) — không phải từ đầu

`--resume` chỉ dùng được khi checkpoint còn nằm **local** (Modal hết phiên là mất). Để an toàn, push snapshot lên Hub **định kỳ** mỗi `save_steps`, rồi máy khác kéo về train tiếp:

```bash
# 1) Trên Modal — train + push snapshot mỗi 100 step lên nhánh 'running'
bash run_train.sh hf_xxx --train \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --max-samples 1000 --save-steps 100 \
  --push --push-every-save \
  --hub-repo <owner>/qwen25vl-3b-vi-hwr-lora \
  --hub-revision stage-10k-running \
  --run-name stage-10k

# 2) Modal đứt → sang Colab, kéo snapshot cuối về train tiếp
bash run_train.sh hf_xxx --train \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --max-samples 1000 --save-steps 100 \
  --init-adapter <owner>/qwen25vl-3b-vi-hwr-lora@stage-10k-running \
  --push --push-every-save \
  --hub-repo <owner>/qwen25vl-3b-vi-hwr-lora \
  --hub-revision stage-10k-running \
  --run-name stage-10k

# 3) Mốc chạy xong → push bản sạch sang nhánh chính thức (bỏ --push-every-save)
bash run_train.sh hf_xxx --train \
  --init-adapter <owner>/qwen25vl-3b-vi-hwr-lora@stage-10k-running \
  --push --hub-repo <owner>/qwen25vl-3b-vi-hwr-lora \
  --hub-revision stage-10k
```

- `--push-every-save` + `--save-steps 100` → mỗi 100 step có 1 snapshot trên Hub, mất tối đa 100 step khi đứt.
- `--init-adapter <repo>@<revision>` nạp **trọng số** adapter cũ rồi train tiếp (reset LR/optimizer — khác `--resume` là giữ cả optimizer).
- Base model phải **cùng họ** với adapter (3B adapter + 3B base).
- Mỗi mốc dùng 1 nhánh riêng để không đè nhau; nhánh `main` để trống làm nơi đặt bản production cuối cùng.

> Dataset mặc định: source gốc gated `5CD-AI/Viet-Handwriting-OCR-v2` (phải accept điều khoản trên HF). Đổi bằng `--dataset <owner>/<repo>`.

## Chạy trên A100 / H100 (Colab Pro)

```bash
!pip install -q flash-attn   # 1 lần mỗi runtime — code tự dùng khi có, fallback sdpa khi không
```

```bash
# A100 40GB: bắt đầu batch 2, thử batch 4 nếu còn VRAM
./run_train.sh --train --model Qwen/Qwen2.5-VL-7B-Instruct --batch-size 2 --max-samples 100

# H100 80GB: bắt đầu batch 4, có thể thử batch 8
./run_train.sh --train --model Qwen/Qwen2.5-VL-7B-Instruct --batch-size 4 --max-samples 100
```

`ATTN_IMPLEMENTATION="auto"` trong config (flash_attention_2 nếu import được, sdpa nếu không) nên không cần sửa code. Batch size cứ tăng dần tới khi gần đầy VRAM rồi lùi 1 nấc. `gradient_accumulation_steps=8`, nên batch 2 có effective batch 16 và batch 4 có effective batch 32.

## Ghi chú quan trọng

- Ảnh train/inference giữ nguyên gốc (dataset là ảnh crop dòng) — processor tự resize về lưới 28x28 và giới hạn theo `MIN_PIXELS`/`MAX_PIXELS` trong `configs/configs.py`.
- `models/`, `data/`, `.hf_cache/`, `.env.dev` không commit (git-ignored). Cache HF nằm trong project nên chạy lại không tải lại.
- UI offline (không tạo link public): `GRADIO_SHARE=0`.
- Cần torch CUDA thủ công: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128` (wrapper thường tự lo).
