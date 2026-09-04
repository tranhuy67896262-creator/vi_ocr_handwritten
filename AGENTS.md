# AGENTS.md

Dự án fine-tune **Qwen2.5-VL-3B-Instruct** bằng **QLoRA** trên chữ viết tay tiếng Việt (dataset `hf://buckets/tranhuy67896262/Viet-Handwriting-OCR-v2` — Storage Bucket public, đọc được qua `load_dataset` bằng prefix `hf://buckets/...` — / fallback gốc gated `5CD-AI/Viet-Handwriting-OCR-v2`). Mục tiêu thiết kế: **chỉ thêm kiến thức, không làm mất kiến thức gốc** → base model luôn đóng băng, chỉ train LoRA adapter (lr 1e-5–2e-5, 1–2 epoch). Đừng "tối ưu" lr lên ~2e-4 hay kéo dài epoch — sẽ phá mục tiêu này. KL-regularization (mặc định bật) chống catastrophic forgetting; chỉ tắt khi VRAM hẹp.

## Chạy & verify

- Entry points: `python scripts/train_qlora.py` (train), `python scripts/eval_ocr.py --image <path>` (OCR 1 ảnh) hoặc `--num-test N` (CER/WER trên test split), `python scripts/export_merged.py` (merge LoRA + base -> full model, cần GPU), `python scripts/ui.py` (UI Gradio — cần `pip install gradio`). `main.py` chỉ là stub in config.
- Scripts tự `sys.path.insert(0, project_root)` — chạy trực tiếp, đừng import như module.
- `run_train.sh` (Linux/Colab; nhận token làm arg đầu và tự ghi `.env.dev`) / `run_train.bat` (Windows) = auto-cài uv + deps + torch-CUDA rồi gọi `train_qlora.py`. Mọi CLI flag truyền thẳng qua. Cả 2 đều ưu tiên dùng venv `.venv` nếu có (tạo bằng `uv venv --python 3.13 .venv`), không thì dùng python hệ thống.
- Không có test/lint/CI. Verify nhanh: `python -m compileall configs src scripts main.py` (PowerShell **không** expand `src/**/*.py` — đừng dùng glob đó).
- Git branch `main` đã có commit. Không bao giờ commit `models/` / `data/`.

## Cấu trúc & config

- `configs/configs.py` là nguồn sự thật duy nhất: dataset, model, LoRA, training, prompt hệ thống, cờ `KL_REGULARIZATION`/`KL_COEFFICIENT`. Scripts ghi đè field bằng CLI flag (`--dataset`, `--epochs`, `--lr`, `--lora-r`, `--lora-alpha`, `--no-kl`, ...).
- `src/data/dataset.py`: auto-dò cột ảnh/văn bản (`detect_columns`) rồi format chat template Qwen; fallback sang dataset gốc nếu fork không load được.
- `src/data/collator.py`: tokenize batch bằng `processor(..., add_special_tokens=False)` rồi mask `labels` = -100 cho phần prompt **và padding**. **Giữ `add_special_tokens=False` ở cả 2 chỗ** — bỏ ra sẽ lệch label 1 token.
- `src/model/load.py`: `load_model_and_processor` trả **3-tuple `(model, processor, use_4bit)`** với `use_4bit` là giá trị thực sau fallback (thiếu `bitsandbytes` → LoRA full-precision). Trainer phải nhận `use_4bit` này để chọn optimizer `paged_adamw_8bit` vs `adamw_torch` — đừng lấy `config.USE_4BIT` (luôn True) trong trainer.
- `src/train/kl_trainer.py` (`KLLoRATrainer`, dùng khi `KL_REGULARIZATION=True`): mỗi step forward 2 lần — có LoRA (train) + `disable_adapter()` (no_grad/eval) làm model gốc tham chiếu → `loss = CE + KL_COEFFICIENT * KL(ref || active)`. Tốn thêm 1 forward no_grad/step, không cần copy model thứ 2.
- `src/train/trainer.py`: lưu adapter + `training_metadata.json` vào `models/qwen25vl-3b-vi-hwr-lora/`. `src/infer/predict.py`: load adapter bằng `PeftModel` từ đường dẫn local hoặc repo id `owner/repo`.

## Quirk môi trường

- Python 3.14 / Windows. `bitsandbytes`+CUDA và `flash_attn` không ổn trên Windows → `ATTN_IMPLEMENTATION="sdpa"`. Train thật nên chạy trên **WSL2 hoặc GPU cloud** (~16 GB VRAM, RTX 4090 ổn; KL-reg thêm ~1 forward no_grad/step).
- `HF_TOKEN` đọc từ `.env.dev` (git-ignored) qua `python-dotenv` trong `configs/configs.py`. Mọi access model/dataset cần token; fork **private**, gốc **gated** (phải đồng ý điều khoản HF).
- `models/` và `data/` bị git-ignore (trừ `.gitkeep`) — không bao giờ commit adapter/model/data.

## Quy ước

- Comment/docstring bằng tiếng Việt.
- Đừng hard-code tên cột dataset — luôn dùng `detect_columns()` (đã áp dụng ở cả train lẫn eval).