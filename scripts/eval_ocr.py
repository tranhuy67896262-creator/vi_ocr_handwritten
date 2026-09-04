import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset
from jiwer import cer, wer
from PIL import Image

from configs.configs import Configs
from src.data.dataset import detect_columns
from src.infer.predict import load_ocr_model, predict_image
from src.utils.logging import log_and_exit, setup_file_logging


def main():
    parser = argparse.ArgumentParser(
        description="OCR 1 ảnh hoặc đánh giá CER/WER trên test split"
    )
    parser.add_argument("--image", type=str, default=None, help="Đường dẫn ảnh cần OCR")
    parser.add_argument("--adapter", type=str, default=None,
                        help="Thư mục LoRA adapter hoặc repo id trên Hub (vd owner/repo); mặc định: từ config")
    parser.add_argument("--num-test", type=int, default=100, help="Số mẫu đánh giá trên test split")
    args = parser.parse_args()

    config = Configs()
    setup_file_logging(config.MODELS_DIR / "eval.log")
    adapter_dir = args.adapter or str(config.ADAPTER_DIR)

    try:
        model, processor = load_ocr_model(config, adapter_dir)
    except Exception as exc:
        log_and_exit(
            exc, stage="MODEL",
            extra_hint=f"Kiểm tra adapter tại: {adapter_dir} (tồn tại local hoặc là repo id 'owner/repo').",
        )

    if args.image:
        try:
            img = Image.open(args.image).convert("RGB")
            print(predict_image(config, model, processor, img))
        except Exception as exc:
            log_and_exit(exc, stage="OCR", extra_hint=f"Kiểm tra đường dẫn ảnh: {args.image}")
        return

    try:
        ds = load_dataset(config.DATASET_NAME, split=config.TEST_SPLIT, token=config.HF_TOKEN or None)
    except Exception:
        ds = load_dataset("5CD-AI/Viet-Handwriting-OCR-v2", split=config.TEST_SPLIT, token=config.HF_TOKEN or None)

    ds = ds.select(range(min(args.num_test, len(ds))))
    image_col, text_col = detect_columns(ds)
    texts, preds = [], []
    for row in ds:
        try:
            img = row[image_col].convert("RGB")
            pred = predict_image(config, model, processor, img)
        except Exception as exc:
            log_and_exit(exc, stage="OCR",
                         extra_hint="Lỗi khi OCR 1 mẫu (thường do ảnh hỏng hoặc VRAM).")
        texts.append(row[text_col])
        preds.append(pred)
        print(f"GT : {row[text_col]}")
        print(f"PR : {pred}")
        print("-" * 60)

    print(f"CER: {cer(texts, preds):.4f}")
    print(f"WER: {wer(texts, preds):.4f}")


if __name__ == "__main__":
    main()