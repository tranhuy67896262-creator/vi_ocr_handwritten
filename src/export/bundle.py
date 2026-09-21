"""Build gói GGUF: merge adapter + convert + verify đủ mmproj/Modelfile."""
from pathlib import Path

from .proc import stream_command


def _newest_mtime(path):
    """Mốc file mới nhất trong cây thư mục (0 nếu rỗng)."""
    files = [f for f in Path(path).rglob("*") if f.is_file()]
    return max((f.stat().st_mtime for f in files), default=0)


def merged_is_fresh(merge_dir, adapter):
    """Merged còn dùng được nếu có safetensors và mới hơn adapter local.

    Adapter là repo Hub (không có local) thì luôn merge lại cho chắc.
    """
    merged = Path(merge_dir)
    if not merged.exists() or not list(merged.glob("*.safetensors")):
        return False
    src = Path(adapter)
    if not src.exists():
        return False
    return _newest_mtime(merged) >= _newest_mtime(src)


def split_gguf_files(gguf_dir):
    """Tách (main_text, mmproj) trong thư mục GGUF; thiếu thì None tương ứng."""
    files = sorted(Path(gguf_dir).glob("*.gguf"))
    mmproj = next((str(f) for f in files if "mmproj" in f.name), None)
    main = next((str(f) for f in files if "mmproj" not in f.name), None)
    modelfile = Path(gguf_dir, "Modelfile")
    if not mmproj or not main or not modelfile.exists():
        return None, None
    return main, mmproj


class GgufBundle:
    """Build 1 gói GGUF hoàn chỉnh từ adapter; không biết gì về Gradio/Ollama."""

    def __init__(self, runner=stream_command):
        self._run = runner

    def build(self, *, adapter, model, revision, models_dir,
              python_exe, scripts_dir, emit):
        """Merge + convert, trả đường dẫn thư mục gguf nếu OK, None nếu lỗi."""
        tag = f"{Path(adapter).name}-{revision}" if revision else Path(adapter).name
        merge_dir = str(Path(models_dir) / f"{tag}-merged")
        gguf_dir = str(Path(models_dir) / f"gguf-{tag}")
        if merged_is_fresh(merge_dir, adapter):
            emit(f"Dùng merged có sẵn (mới hơn adapter): {merge_dir}")
        else:
            cmd = [python_exe, str(Path(scripts_dir) / "export_merged.py"),
                   "--adapter", adapter, "--model", model, "--output", merge_dir]
            if revision:
                cmd += ["--adapter-revision", revision]
            if self._run(cmd, emit) is None:
                return None
            if not list(Path(merge_dir).glob("*.safetensors")):
                emit("[ERR] Merge thất bại (không ra safetensors).")
                return None
        Path(gguf_dir).mkdir(parents=True, exist_ok=True)
        self._run(["bash", str(Path(scripts_dir) / "export_gguf.sh"),
                   merge_dir, gguf_dir], emit)
        main, mmproj = split_gguf_files(gguf_dir)
        if not main:
            emit("[ERR] Thiếu gguf/mmproj/Modelfile — xem log convert.")
            return None
        emit(f"GGUF: {main}\nmmproj: {mmproj}")
        return gguf_dir
