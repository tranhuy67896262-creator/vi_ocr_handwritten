import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.configs import Configs
from src.data.dataset import build_train_eval_datasets
from src.model.load import build_lora_model, load_model_and_processor
from src.train.trainer import train
from src.utils.logging import setup_file_logging


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen2.5-VL-7B bằng LoRA/QLoRA trên chữ viết tay Việt"
    )
    parser.add_argument("--dataset", type=str, default=None, help="Tên dataset HF (mặc định: từ config)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Giới hạn mẫu để test nhanh")
    parser.add_argument("--no-kl", action="store_true",
                        help="Tắt KL-regularization (chống mất kiến thức) để tiết kiệm VRAM")
    parser.add_argument("--push", action="store_true", help="Push adapter lên Hugging Face Hub")
    parser.add_argument("--hub-repo", type=str, default=None, help="Tên repo Hub đích khi push, vd: owner/qwen25vl-7b-vi-hwr-lora")
    args = parser.parse_args()

    config = Configs()
    setup_file_logging(config.MODELS_DIR / "training.log")
    if args.dataset:
        config.DATASET_NAME = args.dataset
    if args.epochs is not None:
        config.NUM_EPOCHS = args.epochs
    if args.lr is not None:
        config.LEARNING_RATE = args.lr
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.lora_r is not None:
        config.LORA_R = args.lora_r
    if args.lora_alpha is not None:
        config.LORA_ALPHA = args.lora_alpha
    if args.no_kl:
        config.KL_REGULARIZATION = False
    if args.hub_repo:
        config.HUB_ADAPTER_ID = args.hub_repo

    print(f"Dataset: {config.DATASET_NAME}")
    print(f"Model:   {config.MODEL_NAME}")
    print(f"QLoRA R={config.LORA_R} alpha={config.LORA_ALPHA} | lr={config.LEARNING_RATE} | epochs={config.NUM_EPOCHS}")

    train_ds, eval_ds = build_train_eval_datasets(config, max_samples=args.max_samples)

    model, processor, use_4bit = load_model_and_processor(config)
    model = build_lora_model(config, model)

    train(config, model, processor, train_ds, eval_ds,
          push=args.push or config.PUSH_TO_HUB, hub_repo_id=config.HUB_ADAPTER_ID,
          use_4bit=use_4bit)


if __name__ == "__main__":
    main()