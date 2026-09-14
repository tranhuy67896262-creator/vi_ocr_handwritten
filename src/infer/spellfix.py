"""Hậu xử lý tiếng Việt cho OCR: chuẩn hóa (underthesea) + model spell-correction.

Pipeline 2 tầng (giống nghiên cứu Underthesea + BARTpho):
  1. Chuẩn hóa chuỗi/dấu cũ bằng ``underthesea.text_normalize`` (nếu có cài).
  2. Sửa dấu/lỗi ký tự/lỗi OCR bằng model HF ``nrl-ai/vn-spell-correction-small``
     (BARTpho-syllable 115M; bắt cả o<->0, l<->1, m<->rn).
Chạy trên ``transformers`` (đã có sẵn). Xử lý theo từng dòng để giữ cấu trúc dòng.
"""
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MAX_TOKENS = 256


def _device():
    """CUDA nếu có, ngược lại CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def _normalize_vietnamese(text):
    """Chuẩn hóa NFC/old-diacritics/Đ-Ð bằng underthesea (bỏ qua nếu chưa cài)."""
    try:
        from underthesea import text_normalize
    except ImportError:
        return text
    try:
        return text_normalize(text)
    except Exception:  # noqa: BLE001  # pylint: disable=broad-except
        return text


class VNSpellCorrector:
    """Sửa text OCR tiếng Việt bằng seq2seq (model HF có sẵn)."""

    def __init__(self, model_id, device=None, hf_token=None, max_length=MAX_TOKENS):
        self.model_id = model_id
        self.device = device or _device()
        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token or None)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id, token=hf_token or None
        ).to(self.device).eval()

    def _fix_chunk(self, text):
        enc = self.tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=self.max_length)
        enc = {key: val.to(self.device) for key, val in enc.items()}
        with torch.no_grad():
            out = self.model.generate(**enc, max_length=self.max_length, num_beams=1)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def _fix_line(self, line):
        """Chuẩn hóa + cắt dòng dài theo từ để không mất chữ, rồi sửa từng khúc."""
        if not line.strip():
            return line
        line = _normalize_vietnamese(line)
        chunks, cur, cur_len = [], [], 0
        for word in line.split():
            cost = len(self.tokenizer.tokenize(word)) + 1
            if cur and cur_len + cost > self.max_length - 8:
                chunks.append(" ".join(cur))
                cur, cur_len = [], 0
            cur.append(word)
            cur_len += cost
        if cur:
            chunks.append(" ".join(cur))
        return " ".join(self._fix_chunk(chunk) for chunk in chunks)

    def correct(self, text):
        """Sửa toàn bộ text, giữ nguyên số dòng."""
        if not text:
            return text
        return "\n".join(self._fix_line(line) for line in text.split("\n"))

    def correct_batch(self, texts):
        """Sửa list text (tuần tự)."""
        return [self.correct(text) for text in texts]
