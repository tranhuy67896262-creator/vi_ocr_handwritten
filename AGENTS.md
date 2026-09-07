# AGENTS.md

Fine-tune **Qwen2.5-VL-7B-Instruct** bằng **QLoRA** cho chữ viết tay tiếng Việt. Mục tiêu thiết kế: **chỉ thêm kiến thức, không mất kiến thức gốc** → base model đóng băng, chỉ train LoRA adapter (lr 1e-5–2e-5, 1–2 epoch). Đừng nâng lr (~2e-4) hay kéo dài epoch. KL-regularization (mặc định bật) chống catastrophic forgetting; chỉ tắt (`--no-kl`) khi VRAM hẹp.

## Chạy & verify

- Entry points: `python scripts/train_qlora.py` (train), `python scripts/eval_ocr.py --image <path>` (OCR 1 ảnh) hoặc `--num-test N` (CER/WER, mặc định 100), `python scripts/export_merged.py` (merge LoRA→full model, cần GPU), `python scripts/ui.py` (Gradio — cài riêng `gradio`), `scripts/export_gguf.sh` (Linux/Colab, sau export_merged). `main.py` chỉ là stub in config.
- Scripts tự `sys.path.insert(0, project_root)` — chạy trực tiếp, đừng import như module.
- `run_train.sh` (Linux/Colab) / `run_train.bat` (Windows): auto-cài uv + deps + torch-CUDA rồi gọi `train_qlora.py`. Token HF làm arg đầu (tự ghi `.env.dev`), mọi flag còn lại truyền thẳng qua. Ưu tiên `.venv` nếu có. **Cờ riêng (không forward)**: `--ui` = chỉ cài env + mở Gradio; `--eval[=N]` (chỉ `.sh`) = sau train chạy eval N mẫu (mặc định 100).
- `requirements.txt` **không** chứa torch — PyPI mặc định là bản CPU; wrapper tự cài bản CUDA (`cu128`). Đừng thêm torch vào requirements.
- Smoke test trước train đủ: `python scripts/train_qlora.py --max-samples 100`.
- Không có test/lint/CI. Verify: `python -m compileall configs src scripts main.py` (PowerShell không expand `src/**/*.py` — đừng dùng glob đó).
- Không bao giờ commit `models/` / `data/` / `.hf_cache/` / `.env.dev` (đã git-ignore).

## Cấu trúc & config

- `configs/configs.py` là nguồn sự thật duy nhất. `ADAPTER_DIR` tự suy từ `MODEL_NAME` (hiện tại `models/qwen25vl-7b-vi-hwr-lora/`) — đừng hard-code đường dẫn `*-3b-*` cũ còn sót trong README. CLI ghi đè field (`--dataset/--model/--epochs/--lr/--batch-size/--max-seq-len/--lora-r/--lora-alpha/--no-kl/--push/--hub-repo`).
- `src/data/dataset.py`: `detect_columns()` auto-dò cột ảnh/văn bản; `load_dataset_with_fallback` thử `config.DATASET_NAME` (`hf://buckets/...`) rồi fallback gated `5CD-AI/Viet-Handwriting-OCR-v2`. Đừng hard-code tên cột (áp dụng cả train lẫn eval).
- `src/data/collator.py`: mask `labels = -100` cho prompt **và** padding. **Giữ `add_special_tokens=False` ở cả processor-batch lẫn tokenizer-prompt** (collator) và `predict.py` — bỏ ra lệch label/decode 1 token.
- `src/model/load.py`: `load_model_and_processor` trả 3-tuple `(model, processor, use_4bit)` — `use_4bit` là giá trị thực sau fallback (không CUDA / thiếu `bitsandbytes` → LoRA full-precision). `src/train/trainer.py` dùng `use_4bit` này để chọn optimizer `paged_adamw_8bit` vs `adamw_torch` — đừng đọc `config.USE_4BIT` (luôn True). Trainer còn set `remove_unused_columns=False` và `gradient_checkpointing_kwargs={"use_reentrant": False}` — giữ nguyên.
- `src/train/kl_trainer.py` (`KLLoRATrainer`, khi `KL_REGULARIZATION=True`): mỗi step forward 2 lần — có LoRA + `disable_adapter()` (no_grad) làm ref → `loss = CE + KL_COEFFICIENT * KL(ref || active)`. Không cần copy model thứ 2.
- `src/train/trainer.py`: lưu adapter + `training_metadata.json` vào `ADAPTER_DIR`; push Hub cần `--hub-repo owner/repo`. `src/infer/predict.py`: `adapter_dir` là path local **hoặc** repo id `owner/repo` (không phải tên trần).

## Quirk môi trường

- Python 3.13 / Windows. `bitsandbytes`+CUDA và `flash_attn` hỏng trên Windows → `ATTN_IMPLEMENTATION="sdpa"`. Train thật chạy **WSL2/GPU cloud** (7B cần ~16 GB+; export merged cần gấp ~2x).
- `HF_TOKEN` đọc từ `.env.dev` qua `python-dotenv` trong `configs.py`. Dataset gốc gated — phải accept điều khoản trên HF.
- Comment/docstring bằng tiếng Việt.
