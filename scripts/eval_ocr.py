"""OCR 1 file / đánh giá CER-WER trên test split (zero-shot / few-shot / adapter)."""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset
from jiwer import cer, wer
from PIL import Image

from configs.configs import Configs, _adapter_tag
from src.datasets.dataset import detect_columns, load_dataset_with_fallback
from src.infer.predict import load_ocr_model, predict_image, predict_image_fewshot
from src.utils.image import standardize_a4
from src.utils.logging import log_and_exit, setup_file_logging


def _load_static_exemplars(config, k, seed):
    """Lấy ``k`` mẫu (ảnh, text) cố định từ split train (shuffle theo seed)."""
    ds = load_dataset_with_fallback(config)
    image_col, text_col = detect_columns(ds)
    k = min(k, len(ds))
    ds = ds.shuffle(seed=seed).select(range(k))
    print(f"Few-shot: {k} mẫu cố định từ {config.DATASET_NAME} "
          f"[{config.TRAIN_SPLIT}] (seed={seed})")
    return [(row[image_col].convert("RGB"), row[text_col]) for row in ds]


def _build_retriever(config, index_path, clip_name):
    """Nạp index + embedder + pool dataset để truy hồi mẫu theo ảnh query."""
    from src.infer.retrieval import CLIP_NAME, ClipEmbedder, ImageIndex
    index = ImageIndex.load(index_path)
    embedder = ClipEmbedder(clip_name or CLIP_NAME)
    # refs là chỉ số dòng của pool lúc build -> phải nạp ĐÚNG dataset/split đó,
    # nếu không sẽ lấy nhầm ảnh/nhãn (HF full vs data/combined có thể lệch).
    # Chỉ dùng tạm rồi khôi phục để test split vẫn theo --dataset của người dùng.
    meta = index.meta or {}
    orig_dataset, orig_split = config.DATASET_NAME, config.TRAIN_SPLIT
    if meta.get("dataset"):
        config.DATASET_NAME = meta["dataset"]
    if meta.get("split"):
        config.TRAIN_SPLIT = meta["split"]
    try:
        ds = load_dataset_with_fallback(config)
    finally:
        config.DATASET_NAME, config.TRAIN_SPLIT = orig_dataset, orig_split
    image_col, text_col = detect_columns(ds)
    if index.refs.size and int(index.refs.max()) >= len(ds):
        log_and_exit(
            ValueError("Index refs vuot so dong dataset hien tai."),
            stage="RETRIEVAL",
            extra_hint=(f"Index build tren '{meta.get('dataset')}' ({len(index.refs)} anh) "
                        f"nhung dataset hien tai chi {len(ds)} dong. Build lai index dung dataset."),
        )
    print(f"Retrieval: index '{index_path}' ({len(index.refs)} ảnh) | "
          f"pool={meta.get('dataset')}[{meta.get('split')}] | embedder={embedder.model_name}")
    return index, embedder, ds, image_col, text_col


def _retrieve_exemplars(config, retriever, query_img, k):
    """Truy hồi ``k`` mẫu gần nhất với ảnh query từ index (theo style CLIP)."""
    index, embedder, ds, image_col, text_col = retriever
    q = query_img.convert("RGB")
    if config.A4_STANDARDIZE:
        q = standardize_a4(q, max_pixels=config.MAX_PIXELS)
    refs = index.search(embedder.embed([q])[0], k)
    exemplars = []
    for ref in refs:
        row = ds[ref]
        exemplars.append((row[image_col].convert("RGB"), row[text_col]))
    return exemplars


def _predict(config, model, processor, img, k, static_exemplars, retriever):
    """OCR 1 ảnh: zero-shot (k=0) / few-shot cố định / few-shot retrieval."""
    if not k or k <= 0:
        return predict_image(config, model, processor, img)
    exemplars = (_retrieve_exemplars(config, retriever, img, k)
                 if retriever is not None else static_exemplars)
    return predict_image_fewshot(config, model, processor, img, exemplars)


def main():
    """Parse CLI và chạy OCR/eval."""
    parser = argparse.ArgumentParser(
        description="OCR 1 ảnh hoặc đánh giá CER/WER trên test split"
    )
    parser.add_argument("--image", type=str, default=None, help="Đường dẫn ảnh cần OCR")
    parser.add_argument("--model", type=str, default=None,
                        help="Base model (mặc định: từ config) — phải cùng họ với adapter")
    parser.add_argument("--adapter", type=str, default=None,
                        help="Thư mục LoRA adapter hoặc repo id trên Hub (vd owner/repo); mặc định: từ config")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Tên dataset HF để đánh giá (mặc định: từ config)")
    parser.add_argument("--num-test", type=int, default=100, help="Số mẫu đánh giá trên test split")
    parser.add_argument("--no-adapter", action="store_true",
                        help="Zero-shot: chạy base model KHÔNG nạp LoRA adapter")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Prompt hệ thống tùy biến cho OCR (mặc định: config.SYSTEM_PROMPT)")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Số token tối đa sinh ra (mặc định: config.MAX_NEW_TOKENS)")
    parser.add_argument("--few-shot", type=int, default=0,
                        help="Số mẫu few-shot in-context (0 = tắt). Áp cho ảnh lẻ và eval.")
    parser.add_argument("--few-shot-seed", type=int, default=42,
                        help="Seed chọn mẫu few-shot cố định từ train split")
    parser.add_argument("--retrieval-index", type=str, default=None,
                        help="File .npz (build_retrieval_index.py) -> few-shot theo retrieval")
    parser.add_argument("--clip", type=str, default=None,
                        help="Model CLIP để nhúng query khi retrieval (mặc định: theo retrieval.CLIP_NAME)")
    args = parser.parse_args()

    config = Configs()
    setup_file_logging(config.MODELS_DIR / "eval.log")
    if args.model:
        config.MODEL_NAME = args.model
        config.ADAPTER_DIR = config.MODELS_DIR / f"qwen25vl-{_adapter_tag(args.model)}-vi-hwr-lora"
    if args.dataset:
        config.DATASET_NAME = args.dataset
    if args.max_new_tokens is not None:
        config.MAX_NEW_TOKENS = args.max_new_tokens
    if args.prompt:
        # predict_image/ocr_pdf/ocr_docx mặc định dùng config.SYSTEM_PROMPT.
        config.SYSTEM_PROMPT = args.prompt
    if args.no_adapter:
        adapter_dir = None
        print("Zero-shot: base model, KHÔNG nạp adapter.")
    else:
        adapter_dir = args.adapter or str(config.ADAPTER_DIR)

    try:
        model, processor = load_ocr_model(config, adapter_dir, args.model)
    except Exception as exc:
        hint = ("Kiểm tra kết nối mạng/HF_TOKEN và tên base model."
                if adapter_dir is None else
                f"Kiểm tra adapter tại: {adapter_dir} (tồn tại local hoặc là repo id 'owner/repo').")
        log_and_exit(exc, stage="MODEL", extra_hint=hint)

    k = args.few_shot or 0
    static_exemplars = []
    retriever = None
    if k > 0:
        try:
            if args.retrieval_index:
                retriever = _build_retriever(config, args.retrieval_index, args.clip)
            else:
                static_exemplars = _load_static_exemplars(config, k, args.few_shot_seed)
        except Exception as exc:  # noqa: BLE001
            log_and_exit(exc, stage="FEWSHOT",
                         extra_hint="Kiểm tra HF_TOKEN/dataset, hoặc file --retrieval-index.")

    if args.image:
        try:
            suf = args.image.lower()
            if suf.endswith(".pdf"):
                from src.infer.predict import ocr_pdf
                print(ocr_pdf(config, model, processor, args.image))
            elif suf.endswith(".docx"):
                from src.infer.predict import ocr_docx
                print(ocr_docx(config, model, processor, args.image))
            else:
                img = Image.open(args.image).convert("RGB")
                print(_predict(config, model, processor, img, k, static_exemplars, retriever))
        except Exception as exc:
            log_and_exit(exc, stage="OCR", extra_hint=f"Kiểm tra đường dẫn ảnh: {args.image}")
        return

    try:
        ds = load_dataset(config.DATASET_NAME, split=config.TEST_SPLIT, token=config.HF_TOKEN or None)
    except Exception:
        ds = load_dataset("5CD-AI/Viet-Handwriting-OCR-v2", split=config.TEST_SPLIT, token=config.HF_TOKEN or None)

    ds = ds.select(range(min(args.num_test, len(ds))))
    image_col, text_col = detect_columns(ds)
    mode = "zero-shot" if k <= 0 else (f"retrieval-{k}shot" if retriever else f"few-{k}shot")
    print(f"Eval {len(ds)} mẫu | chế độ: {mode}")
    texts, preds = [], []
    for row in ds:
        try:
            img = row[image_col].convert("RGB")
            pred = _predict(config, model, processor, img, k, static_exemplars, retriever)
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
