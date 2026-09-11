"""Module gắn nhãn dữ liệu OCR: local (ảnh cá nhân) ghép nguồn Hugging Face.

Kiến trúc SOLID, tách tầng:
    domain -> interfaces -> repositories/sources -> services -> web.
Chạy web trên cổng 9000: ``python -m scripts.labeling`` (xem ``cli.py``).
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
