"""Inference OCR: ảnh lẻ, PDF nhiều trang, Word .docx."""
from pathlib import Path

import torch
from peft import PeftModel
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration

from src.modeling.load import load_processor, resolve_attn_implementation
from src.utils.image import standardize_a4


def load_ocr_model(config, adapter_dir=None, model_name=None):
    """Load model gốc + LoRA adapter để OCR.

    adapter_dir có thể là đường dẫn local hoặc repo id trên HF Hub (vd 'owner/repo').
    model_name là base model (mặc định config.MODEL_NAME) — PHẢI cùng họ với adapter
    (adapter 3B + base 7B sẽ lỗi shape).
    """
    base = model_name or config.MODEL_NAME
    processor = load_processor(config, base)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        attn_implementation=resolve_attn_implementation(config),
        token=config.HF_TOKEN or None,
    )
    if adapter_dir:
        is_local = Path(adapter_dir).exists()
        if not is_local and "/" not in str(adapter_dir):
            raise ValueError(
                f"Không thấy adapter '{adapter_dir}'. Truyền đường dẫn local hoặc repo id dạng 'owner/repo'."
            )
        print(f"Load adapter từ: {adapter_dir}")
        model = PeftModel.from_pretrained(model, adapter_dir, token=config.HF_TOKEN or None)
    model.eval()
    return model, processor


def predict_image(config, model, processor, image, system_prompt=None):
    """OCR 1 ảnh PIL -> trả về chuỗi chữ viết tay đọc được."""
    image = image.convert("RGB")
    if config.A4_STANDARDIZE:
        image = standardize_a4(image, max_pixels=config.MAX_PIXELS)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": system_prompt or config.SYSTEM_PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text], images=[image], return_tensors="pt", add_special_tokens=False,
        min_pixels=config.MIN_PIXELS, max_pixels=config.MAX_PIXELS,
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=config.MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
        )
    output_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def render_pdf_pages(pdf_path, dpi=200):
    """Render từng trang PDF scan thành list ảnh PIL (RGB).

    Lazy import pymupdf để môi trường chỉ-train không cần cài.
    """
    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz
        except ImportError:
            raise ImportError(
                "Thiếu pymupdf — chạy `pip install pymupdf` để OCR file PDF."
            ) from None
    pages = []
    doc = fitz.open(pdf_path)
    try:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            pages.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
    finally:
        doc.close()
    return pages


def _ocr_tiled(config, model, processor, img, tiles=None):
    """OCR 1 ảnh trang scan: chẻ lát TRƯỚC khi thu nhỏ để giữ chi tiết chữ nhỏ.

    Ảnh lớn (cạnh dài > 1200px, vd A4 scan 200dpi) chẻ thành `tiles` dải ngang
    chồng nhau, mỗi dải chuẩn hóa + OCR riêng rồi gộp. Ảnh nhỏ OCR 1 phát.
    """
    from src.utils.image import split_strips
    n = tiles or config.PAGE_TILES
    strips = split_strips(img, n) if max(img.size) > 1200 else [img]
    texts = [predict_image(config, model, processor, s) for s in strips]
    return "\n".join(texts)


def ocr_pdf(config, model, processor, pdf_path, dpi=None, tiles=None):
    """OCR file PDF scan nhiều trang: OCR từng trang (có chẻ lát) rồi gộp."""
    pages = render_pdf_pages(pdf_path, dpi=dpi or config.PDF_DPI)
    parts = []
    for i, img in enumerate(pages):
        txt = _ocr_tiled(config, model, processor, img, tiles=tiles)
        parts.append(f"--- Trang {i + 1}/{len(pages)} ---\n{txt}")
    return "\n\n".join(parts)


def extract_docx_images(docx_path):
    """Trích ảnh nhúng trong file Word (.docx) theo đúng thứ tự đọc
    (đoạn văn + bảng). Trả về list ảnh PIL (RGB)."""
    try:
        from docx import Document
        from docx.opc.constants import RELATIONSHIP_TYPE as RT
        from docx.oxml.ns import qn
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except ImportError:
        raise ImportError(
            "Thiếu python-docx — chạy `pip install python-docx` để OCR file Word."
        ) from None

    doc = Document(docx_path)
    blobs = {}
    for rel in doc.part.rels.values():
        if rel.reltype == RT.IMAGE:
            blobs[rel.rId] = rel.target_part.blob

    def _paras(block_parent):
        for child in block_parent._element.body.iterchildren():
            if child.tag == qn("w:p"):
                yield Paragraph(child, block_parent)
            elif child.tag == qn("w:tbl"):
                for row in Table(child, block_parent).rows:
                    for cell in row.cells:
                        yield from cell.paragraphs

    images = []
    for para in _paras(doc):
        for run in para.runs:
            for blip in run._element.findall(
                ".//a:blip",
                namespaces={"a": "http://schemas.openxmlformats.org/drawingml/2006/main"},
            ):
                rid = blip.get(
                    "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
                )
                if rid in blobs:
                    from io import BytesIO
                    try:
                        images.append(Image.open(BytesIO(blobs[rid])).convert("RGB"))
                    except Exception:
                        continue
    return images


def ocr_docx(config, model, processor, docx_path, tiles=None):
    """OCR file Word (.docx): OCR từng ảnh nhúng (có chẻ lát) rồi gộp theo thứ tự."""
    images = extract_docx_images(docx_path)
    if not images:
        return "File Word không chứa ảnh nào để OCR."
    parts = []
    for i, img in enumerate(images):
        txt = _ocr_tiled(config, model, processor, img, tiles=tiles)
        parts.append(f"--- Ảnh {i + 1}/{len(images)} ---\n{txt}")
    return "\n\n".join(parts)
