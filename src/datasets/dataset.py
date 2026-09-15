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
    """Chuyển 1 mẫu (ảnh + text) thành chat template của Qwen2.5-VL."""
    img = example[image_col]
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
        "images": [img],
    }


def load_dataset_with_fallback(config, split=None):
    """Load dataset theo config (mặc định source gốc 5CD-AI/Viet-Handwriting-OCR-v2).

    ``split`` mặc định là ``config.TRAIN_SPLIT``. Nếu ``DATASET_NAME`` là thư mục đã
    lưu bằng ``save_to_disk`` (vd ``data/combined`` do ``scripts.labeling merge`` tạo)
    thì đọc trực tiếp từ đĩa.
    """
    split = split or config.TRAIN_SPLIT
    candidates = list(dict.fromkeys([config.DATASET_NAME, "5CD-AI/Viet-Handwriting-OCR-v2"]))
    last_err = None
    for name in candidates:
        try:
            print(f"Đang load dataset: {name} [{split}]")
            if Path(name).is_dir():
                return _load_local_dataset(name, split)
            return load_dataset(name, split=split, token=config.HF_TOKEN or None)
        except Exception as err:  # noqa: BLE001
            last_err = err
            print(f"  Không load được {name}: {err}")
    raise RuntimeError(f"Không load được dataset nào. Lỗi cuối: {last_err}")


def _load_local_dataset(path, split):
    """Đọc DatasetDict đã lưu local và chọn split."""
    bundle = load_from_disk(path)
    if hasattr(bundle, "keys"):
        if split in bundle:
            return bundle[split]
        first = next(iter(bundle.keys()))
        print(f"  Không có split '{split}', dùng '{first}'")
        return bundle[first]
    return bundle


def _format(ds, image_col, text_col, config):
    """Map 1 split sang chat template Qwen (bỏ các cột gốc)."""
    return ds.map(
        lambda ex: convert_to_chat(ex, image_col, text_col, config.SYSTEM_PROMPT),
        remove_columns=ds.column_names,
    )


def _fixed_eval_dataset(config):
    """Eval CỐ ĐỊNH từ ``TEST_SPLIT`` để mọi mốc staged training đo cùng một thước.

    Không lấy từ train: nếu lấy từ train thì mỗi mốc (5k/10k/...) sẽ có tập eval
    khác nhau -> CER giữa các mốc không so sánh được.
    """
    want = getattr(config, "EVAL_SAMPLES", 0) or 0
    if want <= 0:
        return None
    try:
        ds = load_dataset_with_fallback(config, config.TEST_SPLIT)
    except Exception as err:  # noqa: BLE001
        print(f"  Không lấy được eval từ split '{config.TEST_SPLIT}': {err}")
        return None
    seed = getattr(config, "EVAL_SHUFFLE_SEED", config.SEED)
    ds = ds.shuffle(seed=seed).select(range(min(want, len(ds))))
    image_col, text_col = detect_columns(ds)
    print(f"Eval: {len(ds)} mẫu từ '{config.TEST_SPLIT}' (shuffle seed={seed})")
    return _format(ds, image_col, text_col, config)


def build_train_eval_datasets(config, max_samples=None):
    """Load + format dataset train (shuffle rồi cắt N mẫu) và eval cố định.

    - Train: ``shuffle(DATA_SHUFFLE_SEED)`` rồi ``select(range(N))`` -> các mốc staged
      training (5k/10k/20k/40k/59k) vẫn LỒNG NHAU nhưng không lệch theo thứ tự parquet.
    - Eval: ``EVAL_SAMPLES`` dòng từ ``TEST_SPLIT`` (xem ``_fixed_eval_dataset``); nếu
      không bật thì fallback ``train_test_split(VAL_RATIO)`` như trước.
    """
    train_ds = load_dataset_with_fallback(config, config.TRAIN_SPLIT)
    seed = getattr(config, "DATA_SHUFFLE_SEED", config.SEED)
    train_ds = train_ds.shuffle(seed=seed)
    if max_samples is not None:
        train_ds = train_ds.select(range(min(max_samples, len(train_ds))))

    image_col, text_col = detect_columns(train_ds)
    print(f"Cột ảnh: {image_col} | Cột văn bản: {text_col}")
    train_ds = _format(train_ds, image_col, text_col, config)

    eval_ds = _fixed_eval_dataset(config)
    if eval_ds is None and max_samples is None and len(train_ds) > 100:
        split = train_ds.train_test_split(test_size=config.VAL_RATIO, seed=config.SEED)
        print(f"Train: {len(split['train'])} | Eval: {len(split['test'])}")
        return split["train"], split["test"]

    print(f"Train: {len(train_ds)} | Eval: {len(eval_ds) if eval_ds is not None else 0}")
    return train_ds, eval_ds
