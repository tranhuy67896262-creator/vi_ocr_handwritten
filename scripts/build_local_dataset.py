"""Gộp ảnh crop (dòng/word) + nhãn CSV thành DatasetDict local để train — KHÔNG cần HF.

Đọc ``assets/labels.csv`` (cột ``image,text,split``) hoặc CSV tùy chỉnh, tạo dataset
HuggingFace lưu local (``save_to_disk``) để dùng trực tiếp:
    python scripts/train.py --dataset data/local
"""
import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from configs.configs import Configs
from src.utils.console import force_utf8_console
from src.utils.logging import log_and_exit, setup_file_logging


def _read_rows(labels_path):
    """Đọc CSV nhãn -> list dict (bỏ qua dòng trống)."""
    with open(labels_path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _build_records(rows, images_dir):
    """Tạo list {image, text} từ ảnh tồn tại + nhãn không rỗng."""
    records, skipped = [], 0
    for row in rows:
        image_ref = (row.get("image") or "").strip()
        text = (row.get("text") or "").strip()
        if not image_ref or not text:
            skipped += 1
            continue
        path = images_dir / image_ref
        if not path.is_file():
            print(f"  [WARN] Không thấy ảnh: {path}")
            skipped += 1
            continue
        try:
            image = Image.open(path).convert("RGB")
        except OSError as exc:
            print(f"  [WARN] Ảnh lỗi {path}: {exc}")
            skipped += 1
            continue
        records.append({"image": image, "text": text})
    return records, skipped


def main():
    """Parse CLI, dựng DatasetDict và lưu local."""
    force_utf8_console()
    parser = argparse.ArgumentParser(
        description="Gộp ảnh crop + nhãn CSV -> DatasetDict local (không cần HF)"
    )
    parser.add_argument("--images-dir", type=Path, default=Configs.PROJECT_ROOT / "assets")
    parser.add_argument("--labels", type=Path, default=Configs.PROJECT_ROOT / "assets" / "labels.csv")
    parser.add_argument("--output", type=Path, default=Configs.PROJECT_ROOT / "data" / "local")
    parser.add_argument("--test-ratio", type=float, default=0.0,
                        help="Tỉ lệ tách test (0 = toàn bộ vào train)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    Configs()  # tạo thư mục cần thiết + đọc token
    setup_file_logging(Configs.PROJECT_ROOT / "models" / "build_dataset.log")

    if not args.labels.is_file():
        log_and_exit(FileNotFoundError(f"Không thấy file nhãn: {args.labels}"),
                     stage="LABELS",
                     extra_hint="Cần CSV cột image,text (vd assets/labels.csv).")
    rows = _read_rows(args.labels)
    records, skipped = _build_records(rows, args.images_dir)
    if not records:
        log_and_exit(ValueError("Không có mẫu hợp lệ nào (ảnh + nhãn)."),
                     stage="BUILD", extra_hint="Kiểm tra cột image/text và đường dẫn ảnh.")

    try:
        from datasets import Dataset, DatasetDict
    except ImportError as exc:  # pragma: no cover
        log_and_exit(exc, stage="DATASETS", extra_hint="pip install datasets")

    dataset = Dataset.from_list(records)
    if args.test_ratio and 0 < args.test_ratio < 1:
        split = dataset.train_test_split(test_size=args.test_ratio, seed=args.seed)
        bundle = DatasetDict({"train": split["train"], "test": split["test"]})
    else:
        bundle = DatasetDict({"train": dataset})

    args.output.mkdir(parents=True, exist_ok=True)
    bundle.save_to_disk(str(args.output))
    print(f"Đã lưu dataset local: {args.output}")
    for name, part in bundle.items():
        print(f"  {name}: {len(part)} rows")
    if skipped:
        print(f"  [WARN] Bỏ qua {skipped} dòng (thiếu nhãn/ảnh).")
    print("Train: python scripts/train.py --dataset", args.output)


if __name__ == "__main__":
    main()
