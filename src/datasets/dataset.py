"""Load + format dataset chữ viết tay thành chat template Qwen."""
from pathlib import Path

from datasets import load_dataset, load_from_disk


def detect_columns(ds):
    """Tự động tìm cột ảnh (Image) và cột văn bản (text) trong dataset."""
    features = ds.features
    image_col = None
    for name, feat in features.items():
        if "Image" in str(feat):
            image_col = name
            break
    if image_col is None:
        image_col = next((n for n in ("image", "images", "img") if n in features), None)

    text_col = next(
        (n for n in ("text", "transcription", "label", "ground_truth") if n in features),
        None,
    )
    if text_col is None:
        text_col = next(
            (n for n, f in features.items()
             if str(f).startswith("Value") and n != image_col),
            None,
        )

    if image_col is None or text_col is None:
        raise ValueError(
            f"Không tìm thấy cột ảnh/văn bản. Các cột có sẵn: {list(features.keys())}"
        )
    return image_col, text_col


def convert_to_chat(example, image_col, text_col, system_prompt):
    """Chuyển 1 mẫu (ảnh + text) thành chat template của Qwen2.5-VL.

    Ảnh giữ nguyên gốc (không pad/crop) — processor Qwen2.5-VL tự resize về lưới
    28x28 và giới hạn theo MIN_PIXELS/MAX_PIXELS.
    """
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": system_prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": example[text_col]}],
            },
        ],
        "images": [example[image_col]],
    }


def load_dataset_with_fallback(config):
    """Load dataset theo config (mặc định mirror private tranhuy67896262/Viet-Handwriting-OCR-v2-local).

    Nếu ``DATASET_NAME`` là thư mục local thì đọc trực tiếp từ đĩa:
    ``save_to_disk`` (vd ``data/combined`` do ``scripts.labeling merge`` tạo ra)
    hoặc thư mục chứa file parquet (vd copy từ bucket: ``data/v2-bucket`` với
    ``data/*.parquet`` đặt tên theo split ``train-*``/``test-*``).
    """
    candidates = list(dict.fromkeys([config.DATASET_NAME, "tranhuy67896262/Viet-Handwriting-OCR-v2-local"]))
    last_err = None
    for name in candidates:
        try:
            print(f"Đang load dataset: {name}")
            if Path(name).is_dir():
                return _load_local_dataset(name, config.TRAIN_SPLIT)
            return load_dataset(name, split=config.TRAIN_SPLIT, token=config.HF_TOKEN or None)
        except Exception as err:  # noqa: BLE001
            last_err = err
            print(f"  Không load được {name}: {err}")
    raise RuntimeError(f"Không load được dataset nào. Lỗi cuối: {last_err}")


def _has_parquet(folder):
    """True nếu thư mục chứa ít nhất 1 file parquet."""
    try:
        return any(Path(folder).glob("*.parquet"))
    except OSError:
        return False


def _open_local_bundle(path):
    """Mở dataset local: ưu tiên save_to_disk, fallback thư mục parquet.

    Thư mục parquet (vd copy từ bucket) đặt file theo tên split
    (``train-*.parquet``/``test-*.parquet``) ở root hoặc trong ``data/`` —
    builder parquet tự suy split từ tiền tố tên file.
    """
    root = Path(path)
    if (root / "dataset_dict.json").exists():
        return load_from_disk(str(root))
    data_dir = root if _has_parquet(root) else root / "data"
    files = sorted(data_dir.glob("*.parquet")) if data_dir.is_dir() else []
    if not files:
        return load_from_disk(str(root))  # không phải dataset local: ném lỗi gốc
    print(f"  Đọc {len(files)} file parquet từ {data_dir}")
    return load_dataset("parquet", data_dir=str(data_dir))


def _load_local_dataset(path, split):
    """Đọc DatasetDict đã lưu local và chọn split train."""
    bundle = _open_local_bundle(path)
    if hasattr(bundle, "keys"):
        if split in bundle:
            return bundle[split]
        first = next(iter(bundle.keys()))
        print(f"  Không có split '{split}', dùng '{first}'")
        return bundle[first]
    return bundle


def build_train_eval_datasets(config, max_samples=None, start_samples=0):
    """Load + format dataset, tách 1 phần nhỏ làm eval để theo dõi loss.

    ``start_samples``: bỏ qua N mẫu đầu — train lát data mới (vd 5k đầu xong,
    chạy tiếp 5k sau bằng ``--start-samples 5000 --max-samples 5000``).
    """
    ds = load_dataset_with_fallback(config)
    if max_samples is not None or start_samples:
        start = max(start_samples, 0)
        end = len(ds) if max_samples is None else min(start + max_samples, len(ds))
        ds = ds.select(range(start, end))

    image_col, text_col = detect_columns(ds)
    print(f"Cột ảnh: {image_col} | Cột văn bản: {text_col}")

    ds = ds.map(
        lambda ex: convert_to_chat(ex, image_col, text_col, config.SYSTEM_PROMPT),
        remove_columns=ds.column_names,
    )

    if max_samples is None and len(ds) > 100:
        split = ds.train_test_split(test_size=config.VAL_RATIO, seed=config.SEED)
        print(f"Train: {len(split['train'])} | Eval: {len(split['test'])}")
        return split["train"], split["test"]

    print(f"Train: {len(ds)} | Eval: None"
          + (f" (từ mẫu {start_samples})" if start_samples else ""))
    return ds, None
