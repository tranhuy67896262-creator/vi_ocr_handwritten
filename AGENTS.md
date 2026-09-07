# AGENTS.md

Fine-tune **Qwen2.5-VL-3B-Instruct** bằng **QLoRA** cho chữ viết tay tiếng Việt. Mục tiêu thiết kế: **chỉ thêm kiến thức, không mất kiến thức gốc** → base model đóng băng, chỉ train LoRA adapter (lr 1e-5–2e-5, 1–2 epoch). Đừng nâng lr (~2e-4) hay kéo dài epoch. KL-regularization (mặc định bật) chống catastrophic forgetting; chỉ tắt (`--no-kl`) khi VRAM hẹp.

## Chạy & verify

- Entry points: `python scripts/train_qlora.py` (train), `python scripts/eval_ocr.py --image <path>` (OCR ảnh/PDF/DOCX) hoặc `--num-test N` (CER/WER, mặc định 100), `python scripts/export_merged.py` (merge LoRA→full model, cần GPU), `python scripts/ui.py` (Gradio), `scripts/export_gguf.sh` (Linux/Colab, sau export_merged). `main.py` chỉ là stub in config.
- Scripts tự `sys.path.insert(0, project_root)` — chạy trực tiếp, đừng import như module.
- `run_train.sh` / `run_train.bat`: auto-cài uv + deps + torch-CUDA rồi gọi `train_qlora.py`. Token HF làm arg đầu (tự ghi `.env.dev`), mọi flag còn lại truyền thẳng qua. Ưu tiên `.venv` nếu có. Cả 2 set `UV_LINK_MODE=copy` và cache HF trong project (`.hf_cache`, tôn trợ `HF_HOME` có sẵn). **Mặc định (không cờ)**: chỉ cài `gradio`+`dotenv`+`pymupdf`+`python-docx` rồi mở UI ngay — `--ui` mode còn cài thêm `bitsandbytes`+`transformers`+`peft` best-effort để nút Fine-tune trong UI không OOM (thiếu → fallback full-precision → OOM 14.56 GB). **Cờ riêng**: `--train` (install full + train thật), `--eval[=N]` (eval sau train, mặc định 100). UI offline: `GRADIO_SHARE=0`.
- Batch quirk (đã verify): `=` là delimiter của cmd → `--eval=50` thành 2 arg (wrapper nuốt số sau `--eval`); đừng parse `--eval*` bằng regex findstr (`*`/`$` hỏng sau `[...]`); không ngoặc đơn trong echo/REM nằm trong block `if (...)`.
- `.sh` quirk: giữ `HAS_ARGS` thay vì `[ ${#NEW_ARGS[@]} ... ]` (mảng rỗng + `set -u` crash bash cũ).
- `requirements.txt` **không** chứa torch (PyPI mặc định bản CPU; wrapper cài CUDA `cu128`). Ghim `torchao>=0.16` trừ Windows (`sys_platform != "win32"`): peft `ImportError` khi gắn adapter nếu dính torchao 0.10.0 của Colab; chế độ UI cài best-effort.
- Smoke test: `python scripts/train_qlora.py --max-samples 100`. Không test/lint/CI — verify bằng `python -m compileall configs src scripts main.py` (PowerShell không expand `src/**/*.py`).
- Không bao giờ commit `models/` / `data/` / `.hf_cache/` / `.env.dev` (đã git-ignore).

## Cấu trúc & config

- `configs/configs.py` là nguồn sự thất duy nhất. `ADAPTER_DIR` tự suy từ `MODEL_NAME` (`models/qwen25vl-3b-vi-hwr-lora/`). CLI: `--dataset/--model/--epochs/--lr/--batch-size/--max-seq-len/--lora-r/--lora-alpha/--no-kl/--push/--hub-repo/--resume/--save-steps`. `--model` tự tính lại `ADAPTER_DIR` (class attribute tính 1 lần — đổi model mà không làm vậy sẽ ghi đè nhầm adapter).
- `src/datasets/`: `detect_columns()` auto-dò cột (đừng hard-code tên cột); ảnh chuẩn hóa A4 (`standardize_a4` — pad trắng, canvas bội 28 trong budget `MAX_PIXELS`), gọi ở cả train lẫn `predict_image` theo `A4_STANDARDIZE`.
- `src/datasets/collator.py`: mask `labels=-100` cho prompt **và** padding; giữ `add_special_tokens=False` ở cả 3 chỗ (batch, prompt, `predict.py`) — bỏ ra lệch 1 token.
- `src/modeling/load.py`: `load_model_and_processor` trả `(model, processor, use_4bit)` thực sau fallback — trainer dùng nó (không phải `config.USE_4BIT`) để chọn optimizer. Giữ `remove_unused_columns=False`, `gradient_checkpointing_kwargs={"use_reentrant": False}`. Attention qua `resolve_attn_implementation` (flash_attention_2 nếu có, không thì sdpa) — đừng đọc config trực tiếp.
- `src/train/kl_trainer.py`: `loss = CE + KL_COEFFICIENT * KL(ref || active)`, ref = forward với `disable_adapter()` (no-grad), không cần copy model.
- `src/train/trainer.py`: lưu adapter + `training_metadata.json` vào `ADAPTER_DIR`; push cần `--hub-repo owner/repo`.
- `src/infer/predict.py`: `adapter_dir` là path local **hoặc** repo id `owner/repo`; base phải cùng họ adapter. Có `ocr_pdf()` (render theo `PDF_DPI`) và `ocr_docx()` (đúng thứ tự đọc, cả ảnh trong bảng). CLI eval/export nhận `--model`.
- UI (`scripts/ui.py`): dropdown Base model riêng mỗi tab, đổi model tự trỏ adapter về default (`sync_adapter`); nút nặng gọi `_free_gpu()` trước subprocess; Export 2 nút (thư mục + GGUF full-chain, GGUF yield 3-tuple, chỉ Linux, skip merge nếu fresh); nút push adapter/GGUF lên Hub (cần HF_TOKEN Write + repo `owner/repo`); 403 = sai owner hoặc token Read. Tab Settings: lưu token (hiện tên tài khoản), kiểm tra cache model (`check_model_ui`).

## Quirk môi trường

- Python 3.13 / Windows: train thật chạy **WSL2/GPU cloud**. 3B full-precision (bf16, không QLoRA vì `bitsandbytes` chưa có trên Windows) chiếm ~14 GB — GPU tiểu dùng 14.56 GB (như T4 16GB consumer) **sẽ OOM** nếu để KL regularization bật. `train_qlora.py` tự set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (chống fragmentation, không tăng capacity). Chạy smoke test nhanh trên Windows được: `python scripts/train_qlora.py --max-samples 100 --no-kl --batch-size 1 --max-seq-len 512`. 7B cần ~16GB+; export merged cần gấp ~2x.
- `run_train.sh` auto-cài `flash-attn` (best-effort, Linux only) rồi tăng `--batch-size` dần cho A100/H100.
- `HF_TOKEN` đọc env trước, fallback `.env.dev` mỗi lần khởi tạo `Configs()` (tránh stale khi lưu token giữa session). Dataset gốc gated — accept điều khoản trên HF.
- Comment/docstring tiếng Việt.
