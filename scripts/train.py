"""Fine-tune QLoRA Qwen2.5-VL cho chữ viết tay tiếng Việt."""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.configs import Configs, adapter_dir
from src.datasets.dataset import build_train_eval_datasets
from src.modeling.load import build_lora_model, load_model_and_processor
from src.train.trainer import train
from src.utils.logging import log_and_exit, setup_file_logging


def main():
    """Parse CLI và chạy train."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen2.5-VL-3B bằng LoRA/QLoRA trên chữ viết tay Việt"
    )
    parser.add_argument("--dataset", type=str, default=None, help="Tên dataset HF (mặc định: từ config)")
    parser.add_argument("--model", type=str, default=None, help="Tên model HF (mặc định: từ config)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None,
                        help="Số bước tích lũy gradient trước mỗi lần cập nhật")
    parser.add_argument("--max-seq-len", type=int, default=None,
                        help="Giới hạn token/text mỗi mẫu (giảm để tiết kiệm VRAM)")
    parser.add_argument("--max-pixels", type=int, default=None,
                        help="Giới hạn ảnh theo số tile 28x28 (vd 768 -> 768*28*28 px). "
                             "Giảm để bớt image-token, tránh crash 'Mismatch in image token count'")
    parser.add_argument("--min-pixels", type=int, default=None,
                        help="Pixel tối thiểu theo số tile 28x28; ảnh nhỏ hơn bị phóng to")
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Giới hạn mẫu để test nhanh")
    parser.add_argument("--start-samples", type=int, default=0,
                        help="Bỏ qua N mẫu đầu, train lát data mới "
                             "(vd 5k đầu xong, chạy tiếp 5k sau: --start-samples 5000 --max-samples 5000)")
    parser.add_argument("--resume", action="store_true",
                        help="Tiếp tục từ checkpoint mới nhất (đứt giữa chừng train tiếp, không mất công)")
    parser.add_argument("--save-steps", type=int, default=None,
                        help="Lưu checkpoint mỗi N steps (mặc định: từ config; run dài nên để ~100)")
    parser.add_argument("--no-kl", action="store_true",
                        help="Tắt KL-regularization (chống mất kiến thức) để tiết kiệm VRAM")
    parser.add_argument("--kl-coef", type=float, default=None,
                        help="Hệ số KL (mặc định 0.5 từ config). Model lớn underfit "
                             "(vd 7B thua 3B cùng data) thì hạ về 0.1-0.2 để học nhanh hơn")
    parser.add_argument("--no-4bit", action="store_true",
                        help="Train LoRA full-precision bf16 (bỏ 4-bit QLoRA; chất lượng nhỉnh hơn nhưng tốn VRAM)")
    parser.add_argument("--push", action="store_true", help="Push adapter lên Hugging Face Hub")
    parser.add_argument("--push-every-save", action="store_true", help=(
        "Push snapshot lên Hub mỗi save_steps (chống mất khi đứt giữa chừng, "
        "vd Modal hết tiền). Cần --hub-repo."))
    parser.add_argument("--hub-repo", type=str, default=None, help=(
        "Tên repo Hub đích khi push, vd: owner/qwen25vl-3b-vi-hwr-lora"))
    parser.add_argument("--hub-revision", type=str, default=None, help=(
        "Nhánh (revision) đích trên Hub khi push, vd stage-10k (mặc định: main)"))
    parser.add_argument("--init-adapter", type=str, default=None, help=(
        "Tiếp tục train từ adapter cũ: path local hoặc repo id 'owner/repo[@revision]'. "
        "Nạp trọng số adapter, reset LR/step (khác --resume là khôi phục cả optimizer)."))
    parser.add_argument("--adapter-revision", type=str, default=None, help=(
        "Revision của adapter trên Hub khi dùng --init-adapter (thay @revision)"))
    parser.add_argument("--run-name", type=str, default=None, help=(
        "Tên run/mốc — tách thư mục checkpoint + nhãn trong registry (vd stage-10k)"))
    parser.add_argument("--auto-progress", action="store_true", help=(
        "Tự đọc stage_progress.json trên nhánh --init-adapter và train tiếp phần còn lại "
        "(vd lát 15000 mẫu đứt ở 4800 → start=9800, max=10200). Gần đúng khi mất máy hẳn; "
        "máy còn checkpoint local thì --resume chính xác hơn và được ưu tiên."))
    args = parser.parse_args()

    config = Configs()
    setup_file_logging(config.MODELS_DIR / "training.log")
    if args.dataset:
        config.DATASET_NAME = args.dataset
    if args.model:
        config.MODEL_NAME = args.model
    if args.epochs is not None:
        config.NUM_EPOCHS = args.epochs
    if args.lr is not None:
        config.LEARNING_RATE = args.lr
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.gradient_accumulation_steps is not None:
        config.GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation_steps
    if args.max_seq_len is not None:
        config.MAX_SEQ_LEN = args.max_seq_len
    if args.max_pixels is not None:
        config.MAX_PIXELS = args.max_pixels * 28 * 28
    if args.min_pixels is not None:
        config.MIN_PIXELS = args.min_pixels * 28 * 28
    if args.lora_r is not None:
        config.LORA_R = args.lora_r
    if args.lora_alpha is not None:
        config.LORA_ALPHA = args.lora_alpha
    if args.no_kl:
        config.KL_REGULARIZATION = False
    if args.kl_coef is not None:
        if args.kl_coef < 0:
            parser.error("--kl-coef phải >= 0 (0 = chỉ tính CE, tắt hẳn thì dùng --no-kl).")
        config.KL_COEFFICIENT = args.kl_coef
    if args.no_4bit:
        config.USE_4BIT = False
    if args.hub_repo:
        config.HUB_ADAPTER_ID = args.hub_repo

    # ADAPTER_DIR = class attribute tính sẵn lúc import — tính lại sau khi biết model
    # + precision, nếu không adapter 3B/7B hoặc qlora/bf16 sẽ ghi đè lẫn nhau.
    config.ADAPTER_DIR = adapter_dir(config.MODELS_DIR, config.MODEL_NAME, config.USE_4BIT)
    config.ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {config.DATASET_NAME}")
    print(f"Model:   {config.MODEL_NAME}")
    precision = "4-bit QLoRA" if config.USE_4BIT else "bf16 LoRA (full-precision)"
    print(f"Precision: {precision} | R={config.LORA_R} alpha={config.LORA_ALPHA}"
          f" | lr={config.LEARNING_RATE} | epochs={config.NUM_EPOCHS}")

    try:
        if args.auto_progress and not args.resume:
            args.start_samples, args.max_samples = _apply_progress(config, args)
        train_ds, eval_ds = build_train_eval_datasets(
            config, max_samples=args.max_samples, start_samples=args.start_samples)
    except Exception as exc:
        log_and_exit(
            exc, stage="DATASET",
            extra_hint="Kiểm tra HF_TOKEN (.env.dev) có quyền truy cập dataset; tên dataset đã tồn tại trên Hub chưa.",
        )

    try:
        model, processor, use_4bit = load_model_and_processor(config)
        init_adapter, adapter_revision = _split_adapter_ref(
            args.init_adapter, args.adapter_revision
        )
        model = build_lora_model(
            config, model, init_adapter=init_adapter, adapter_revision=adapter_revision
        )
        _print_actual_lora_rank(model, config)
    except Exception as exc:
        log_and_exit(exc, stage="MODEL", extra_hint="Kiểm tra kết nối mạng, HF_TOKEN, tên model.")

    try:
        train(config, model, processor, train_ds, eval_ds,
              push=args.push or config.PUSH_TO_HUB, hub_repo_id=config.HUB_ADAPTER_ID,
              use_4bit=use_4bit, resume=args.resume, save_steps=args.save_steps,
              hub_revision=args.hub_revision, run_name=args.run_name,
              init_adapter=init_adapter, push_every_save=args.push_every_save,
              start_samples=args.start_samples)
    except Exception as exc:
        log_and_exit(exc, stage="TRAIN")


def _print_actual_lora_rank(model, config):
    """In R/alpha THỰC của adapter đang train.

    Nối chain cũ (``--init-adapter``) thì r/alpha lấy theo adapter cũ,
    ``--lora-r``/config bị bỏ qua — in ra để log không gây hiểu nhầm.
    """
    try:
        adapter_cfg = next(iter(model.peft_config.values()))
        print(f"Adapter thực: R={adapter_cfg.r} alpha={adapter_cfg.lora_alpha}")
    except (AttributeError, StopIteration):
        print(f"Adapter theo config: R={config.LORA_R} alpha={config.LORA_ALPHA}")


def _split_adapter_ref(init_adapter, adapter_revision):
    """Tách 'owner/repo@revision' thành (adapter_id, revision)."""
    if not init_adapter or init_adapter.strip().lower() == "none":
        return None, None
    if adapter_revision is not None and adapter_revision.strip().lower() == "none":
        adapter_revision = None
    if "@" in init_adapter and adapter_revision is None:
        init_adapter, _, adapter_revision = init_adapter.rpartition("@")
    return init_adapter, adapter_revision


def _apply_progress(config, args):
    """Đọc stage_progress.json trên nhánh init-adapter, trả (start, max) còn lại.

    Chỉ áp khi file khớp đúng lát đang xin (cùng start + cùng count) và còn dư.
    Lệch là bỏ qua, train đủ lát như thường. Gần đúng khi mất máy (dataloader
    shuffle) — máy còn checkpoint local thì dùng --resume thay vì cờ này.
    """
    start, count = args.start_samples, args.max_samples
    repo_id, _, revision = (args.init_adapter or "").rpartition("@")
    if not repo_id or "/" not in repo_id or not count:
        return start, count
    try:
        from huggingface_hub import hf_hub_download

        import json

        local = hf_hub_download(
            repo_id=repo_id, filename="stage_progress.json",
            revision=revision or None, token=config.HF_TOKEN)
        with open(local, encoding="utf-8") as handle:
            progress = json.load(handle)
        # File tiến độ của dataset khác (đổi data giữa chừng) thì không áp —
        # file cũ (chưa có trường dataset) vẫn áp như trước để tương thích.
        progress_dataset = progress.get("dataset") or ""
        if progress_dataset and progress_dataset != config.DATASET_NAME:
            print(f"[auto-progress] nhánh train trên '{progress_dataset}', "
                  f"khác dataset đang xin '{config.DATASET_NAME}' — train đủ lát.")
            return start, count
        done = int(progress.get("trained_samples", 0))
        if (int(progress.get("start_samples", -1)) == start
                and int(progress.get("slice_total", -1)) == count
                and 0 < done < count):
            start, count = start + done, count - done
            print(f"[auto-progress] {repo_id}@{revision or 'main'} đã xong {done}/{done + count} "
                  f"mẫu → train tiếp start={start} max={count}.")
        else:
            print("[auto-progress] file tiến độ không khớp lát đang xin — train đủ lát.")
    except Exception as exc:
        print(f"[auto-progress] bỏ qua (không đọc được tiến độ): {type(exc).__name__}: {exc}")
    return start, count


if __name__ == "__main__":
    main()
