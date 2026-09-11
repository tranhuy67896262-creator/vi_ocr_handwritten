"""Khởi động HTTP server đa luồng cho label editor."""

from __future__ import annotations

from http.server import ThreadingHTTPServer
from pathlib import Path

from .handler import create_handler
from .service import LabelService

DEFAULT_TEMPLATE = Path(__file__).resolve().parent / "templates" / "index.html"


def serve(
    service: LabelService,
    host: str = "127.0.0.1",
    port: int = 9000,
    template_path: Path = DEFAULT_TEMPLATE,
) -> None:
    """Chạy server cho tới khi Ctrl+C."""
    handler = create_handler(service, template_path)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"OCR Label Editor: http://{host}:{port}/")
    print("Nhan Ctrl+C de dung server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDa dung server.")
    finally:
        server.server_close()
