# AGENTS.md

Dự án fine-tune **Qwen2.5-VL-7B-Instruct** bằng **QLoRA** trên chữ viết tay tiếng Việt (dataset `5CD-AI/Viet-Handwriting-OCR-v2` / fork `tranhuy67896262/Viet-Handwriting-OCR-v2-bucket`). Mục tiêu thiết kế: **chỉ thêm kiến thức, không làm mất kiến thức gốc** → base model luôn đóng băng, chỉ train LoRA adapter (lr 1e-5–2e-5, 1–2 epoch). Đừng "tối ưu" lr lên ~2e-4 hay kéo dài epoch — sẽ phá mục tiêu này.

## Chạy & verify

- Entry points: `python scripts/train_qlora.py` (train), `python scripts/eval_ocr.py --image <path>` (OCR 1 ảnh) hoặc `--num-test N` (CER/WER trên test split). `main.py` chỉ là stub in config.
- Scripts tự `sys.path.insert(0, project_root)` — chạy trực tiếp, đừng import như module.
- Không có test/lint/CI. Verify nhanh: `python -m py_compile configs/configs.py src/**/*.py scripts/*.py`.
- Git repo **chưa có commit nào** (branch `master` trống).

## Cấu trúc & config

- `configs/configs.py` là nguồn sự thật duy nhất: dataset, model, LoRA, training, prompt hệ thống. Scripts ghi đè field bằng CLI flag (vd `--epochs`, `--lr`, `--lora-r`, `--lora-alpha`, `--dataset`).
- `src/data/dataset.py`: auto-detect cột ảnh/văn bản (`detect_columns`) rồi format chat template Qwen; fallback sang dataset gốc nếu fork không load được.
- `src/data/collator.py`: tokenize batch bằng `processor(..., add_special_tokens=False)` rồi mask `labels` (phần không-phải-assistant = -100) theo `prompt_len`. **Giữ `add_special_tokens=False` ở cả 2 chỗ** — bỏ ra sẽ lệch label 1 token.
- `src/model/load.py`: load 4-bit QLoRA (`USE_4BIT`), tự fallback sang full-precision LoRA nếu thiếu `bitsandbytes`. `src/train/trainer.py`: `Trainer` (transformers), lưu adapter vào `models/qwen25vl-7b-vi-hwr-lora/`. `src/infer/predict.py`: inference, load adapter bằng `PeftModel`.

## Quirk môi trường

- Chạy trên **Python 3.14 / Windows**. `bitsandbytes`+CUDA và `flash_attn` không ổn trên Windows → `ATTN_IMPLEMENTATION="sdpa"`. Train thật nên chạy trên **WSL2 hoặc GPU cloud** (~16 GB VRAM, RTX 4090 ổn).
- `HF_TOKEN` đọc từ `.env.dev` (git-ignored) qua `python-dotenv` trong `configs/configs.py`. Mọi access model/dataset cần token; dataset fork là **private**, dataset gốc là **gated** (phải đồng ý điều khoản trên HF).
- `models/` và `data/` bị git-ignore (trừ `.gitkeep`) — không bao giờ commit adapter/model/data.

## Quy ước

- Comment/docstring bằng tiếng Việt.
- Dataset thường có cột `image` + `text`, nhưng `detect_columns()` tự dò nên đừng hard-code tên cột.