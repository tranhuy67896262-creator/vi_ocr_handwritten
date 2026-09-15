"""Chẻ lát ảnh trang scan cho OCR."""


def split_strips(img, n, overlap=0.12):
    """Chẻ ảnh thành n dải ngang chồng nhau (đọc từ trên xuống).

    Dùng cho trang scan lớn: chẻ TRƯỚC khi processor thu nhỏ để mỗi dải
    giữ nguyên chi tiết, rồi mới OCR từng dải trong budget token.
    overlap (0-1) giúp dòng chữ ở mối nối không bị cắt đôi.
    n <= 1 trả về [img] nguyên.
    """
    if n <= 1:
        return [img]
    w, h = img.size
    step = h / n
    pad = step * overlap
    parts = []
    for i in range(n):
        y0 = int(max(0, round(i * step - (pad if i > 0 else 0))))
        y1 = int(min(h, round((i + 1) * step + (pad if i < n - 1 else 0))))
        parts.append(img.crop((0, y0, w, y1)))
    return parts
