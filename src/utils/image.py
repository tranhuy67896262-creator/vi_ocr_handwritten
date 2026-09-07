import math

from PIL import Image

A4_RATIO = 210 / 297  # rộng / cao của khổ A4 dọc
_TILE = 28  # patch của Qwen2.5-VL
_FALLBACK_MAX_PIXELS = 768 * _TILE * _TILE  # = config.MAX_PIXELS khi không truyền


def _a4_canvas(r, max_pixels):
    """Canvas bội số 28, tỉ lệ gần A4 nhất mà diện tích vẫn <= max_pixels.

    Ưu tiên tỉ lệ trước, rồi chọn canvas lớn nhất trong nhóm tỉ lệ tốt
    để giữ chi tiết chữ viết (sai số tỉ lệ thường < 0.5%).
    """
    tiles = max(1, max_pixels // (_TILE * _TILE))
    cands = []
    for gh in range(1, int(math.ceil(math.sqrt(tiles / r))) + 2):
        gw = max(1, round(gh * r))
        if gw * gh <= tiles:
            cands.append((abs(gw / gh - r), gw * gh, gw, gh))
    best_err = min(c[0] for c in cands)
    _, _, gw, gh = max((c for c in cands if c[0] <= best_err + 0.003),
                       key=lambda c: c[1])
    return gw * _TILE, gh * _TILE


def standardize_a4(img, max_pixels=None):
    """Pad ảnh về đúng tỉ lệ khổ A4 (giữ nguyên nội dung, nền trắng).

    - Ảnh dọc/vuông -> A4 dọc (210:297), ảnh ngang -> A4 xoay ngang (297:210).
    - Không crop/kéo méo: chỉ thu nhỏ cho vừa canvas rồi pad nền trắng.
    - Canvas là bội số 28 (patch Qwen2.5-VL), diện tích <= max_pixels
      -> số image-token bị chặn trên, không tràn MAX_SEQ_LEN gây crash
      truncation (đã gặp với ảnh chụp 3000x4000).
    """
    img = img.convert("RGB")
    w, h = img.size
    if w <= 0 or h <= 0:
        return img
    if max_pixels is None:
        max_pixels = _FALLBACK_MAX_PIXELS

    r = 1 / A4_RATIO if w > h else A4_RATIO
    cw, ch = _a4_canvas(r, max_pixels)

    s = min(1.0, cw / w, ch / h)  # chỉ thu nhỏ, không phóng to
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    canvas = Image.new("RGB", (cw, ch), (255, 255, 255))
    canvas.paste(img.resize((nw, nh), Image.LANCZOS), ((cw - nw) // 2, (ch - nh) // 2))
    return canvas
