"""Tiện ích console: buộc stdout/stderr UTF-8 (tránh lỗi trên Windows cp1252)."""
import sys


def force_utf8_console():
    """Tránh UnicodeEncodeError khi in tiếng Việt trên console Windows cp1252."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            continue
