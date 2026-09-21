"""Nạp gói GGUF vào Ollama (`ollama create`)."""
from pathlib import Path

from .proc import stream_command


def default_ollama_name(adapter, revision):
    """Tên model Ollama mặc định từ adapter (+ revision nếu có)."""
    base = Path(adapter).name.strip().lower().replace("_", "-")
    if revision and revision.strip():
        base += "-" + revision.strip().lower().replace("_", "-")
    return base or "qwen25vl-vi-hwr"


class OllamaImporter:
    """Deploy 1 bundle GGUF vào Ollama server đang chạy."""

    def __init__(self, runner=stream_command):
        self._run = runner

    def import_bundle(self, gguf_dir, name, emit):
        """Chạy `ollama create` với cwd là thư mục gguf. Trả True nếu xong."""
        code = self._run(["ollama", "create", name, "-f", "Modelfile"],
                         emit, cwd=gguf_dir)
        return code == 0

    def push_model(self, hub_name, emit):
        """Đẩy model local lên Ollama Hub (`ollama push owner/model`).

        Cần `ollama signin` 1 lần trên máy này trước. Trả True nếu xong.
        """
        code = self._run(["ollama", "push", hub_name], emit)
        return code == 0
