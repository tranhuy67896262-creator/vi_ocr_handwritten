"""Fine-tune QLoRA Qwen2.5-VL cho chữ viết tay tiếng Việt."""
import argparse
import os
import sys
from pathlib import Path

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
    except Exception as exc:
        log_and_exit(exc, stage="MODEL", extra_hint="Kiểm tra kết nối mạng, HF_TOKEN, tên model.")

    try:
        train(config, model, processor, train_ds, eval_ds,
              push=args.push or config.PUSH_TO_HUB, hub_repo_id=config.HUB_ADAPTER_ID,
              use_4bit=use_4bit, resume=args.resume, save_steps=args.save_steps,
              hub_revision=args.hub_revision, run_name=args.run_name,
              init_adapter=init_adapter, push_every_save=args.push_every_save)
    except Exception as exc:
        log_and_exit(exc, stage="TRAIN")


def _split_adapter_ref(init_adapter, adapter_revision):
    """Tách 'owner/repo@revision' thành (adapter_id, revision)."""
    if not init_adapter:
        return None, None
    if "@" in init_adapter and adapter_revision is None:
        init_adapter, _, adapter_revision = init_adapter.rpartition("@")
    return init_adapter, adapter_revision


if __name__ == "__main__":
    main()
