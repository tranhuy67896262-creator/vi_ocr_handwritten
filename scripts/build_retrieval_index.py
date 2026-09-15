"""Xây index retrieval (CLIP embeddings) cho few-shot OCR."""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.configs import Configs
from src.datasets.dataset import detect_columns, load_dataset_with_fallback
from src.infer.retrieval import CLIP_NAME, ClipEmbedder, build_index
from src.utils.logging import log_and_exit, setup_file_logging


def _default_output(config, seed, pool):
    """Đường dẫn index mặc định: models/retrieval/<dataset>-<split>-s<seed>-p<pool>.npz."""
    tag = config.DATASET_NAME.rsplit("/", 1)[-1].replace(" ", "_")
    return config.MODELS_DIR / "retrieval" / f"{tag}-{config.TRAIN_SPLIT}-s{seed}-p{pool}.npz"


def main():
    """Parse CLI, nhúng pool ảnh và lưu index."""
    parser = argparse.ArgumentParser(
        description="Xây index retrieval few-shot: nhúng CLIP 1 pool ảnh -> .npz"
    )
    parser.add_argument("--dataset", type=str, default=None, help="Tên dataset HF (mặc định: config)")
    parser.add_argument("--split", type=str, default=None, help="Split làm pool (mặc định: train)")
    parser.add_argument("--pool", type=int, default=5000, help="Số ảnh vào index (0 = toàn bộ)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip", type=str, default=CLIP_NAME, help="Model CLIP để nhúng")
    parser.add_argument("--output", type=str, default=None, help="File .npz đầu ra")
    args = parser.parse_args()

    config = Configs()
    setup_file_logging(config.MODELS_DIR / "retrieval.log")
    if args.dataset:
        config.DATASET_NAME = args.dataset
    if args.split:
        config.TRAIN_SPLIT = args.split
    output = Path(args.output) if args.output else _default_output(config, args.seed, args.pool)

    try:
        ds = load_dataset_with_fallback(config)
    except Exception as exc:  # noqa: BLE001
        log_and_exit(exc, stage="DATASET",
                     extra_hint="Kiểm tra HF_TOKEN và tên dataset (pool lấy từ split này).")
    image_col, text_col = detect_columns(ds)
    print(f"Pool: {config.DATASET_NAME} [{config.TRAIN_SPLIT}] | ảnh='{image_col}' text='{text_col}'")

    try:
        embedder = ClipEmbedder(args.clip)
        print(f"Embedder: {args.clip} | device={embedder.device}")
        index = build_index(ds, image_col, embedder, pool=args.pool, seed=args.seed)
    except Exception as exc:  # noqa: BLE001
        log_and_exit(exc, stage="INDEX",
                     extra_hint="Cần mạng để tải CLIP; hoặc ảnh trong pool bị lỗi.")

    index.meta.update({
        "dataset": config.DATASET_NAME,
        "split": config.TRAIN_SPLIT,
        "image_col": image_col,
        "text_col": text_col,
    })
    index.save(output)
    print(f"Đã lưu index ({len(index.refs)} ảnh) vào: {output}")


if __name__ == "__main__":
    main()
