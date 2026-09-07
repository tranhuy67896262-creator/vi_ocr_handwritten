from datasets import load_dataset


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
    """Load dataset fork; nếu không load được thì fallback sang dataset gốc."""
    candidates = [config.DATASET_NAME, "5CD-AI/Viet-Handwriting-OCR-v2"]
    last_err = None
    for name in candidates:
        try:
            print(f"Đang load dataset: {name}")
            return load_dataset(name, split=config.TRAIN_SPLIT, token=config.HF_TOKEN or None)
        except Exception as err:  # noqa: BLE001
            last_err = err
            print(f"  Không load được {name}: {err}")
    raise RuntimeError(f"Không load được dataset nào. Lỗi cuối: {last_err}")


def build_train_eval_datasets(config, max_samples=None):
    """Load + format dataset, tách 1 phần nhỏ làm eval để theo dõi loss."""
    ds = load_dataset_with_fallback(config)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

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

    print(f"Train: {len(ds)} | Eval: None")
    return ds, None