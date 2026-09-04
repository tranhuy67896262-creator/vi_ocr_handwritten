import logging
import sys
import traceback
from pathlib import Path

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def log_and_exit(exc, stage="", extra_hint=""):
    """Phân loại lỗi, ghi log file + stderr rồi exit(1).

    - CUDA OOM: báo đúng loại + gợi ý giảm VRAM.
    - Lỗi khác: ghi full traceback kèm stage để biết bug nằm ở đâu.
    """
    exc_type = type(exc).__name__
    msg = str(exc) or exc_type

    hint = extra_hint
    if exc_type == "OutOfMemoryError" or "CUDA out of memory" in msg:
        hint = (
            "GPU hết VRAM. Giảm --batch-size 1, thêm --no-kl, giảm --max-seq-len "
            "(vd 512), hoặc đổi model nhỏ hơn (Qwen/Qwen2.5-VL-3B-Instruct)."
        )
    elif exc_type in ("DatasetNotFoundError",) or "doesn't exist on the Hub" in msg:
        hint = "Tên dataset không tồn tại hoặc không truy cập được (private/gated/bucket). Kiểm tra token HF."
    elif exc_type == "ConnectionError" or "getaddrinfo" in msg:
        hint = "Lỗi mạng khi tải từ Hugging Face. Kiểm tra kết nối internet."

    full = "".join(traceback.format_exception(exc))
    log = logging.getLogger("error")
    log.error("[%s] Loai loi: %s | %s\n%s", stage or "UNKNOWN", exc_type, msg, full)

    print(f"[ERROR] Stage: {stage or 'UNKNOWN'} | Loại lỗi: {exc_type}", file=sys.stderr)
    print(f"  {msg}", file=sys.stderr)
    if hint:
        print(f"  Gợi ý: {hint}", file=sys.stderr)
    sys.exit(1)


def setup_file_logging(log_path):
    """Ghi log console + traceback lỗi vào file .log.

    - Mọi log của transformers/accelerate/huggingface_hub sẽ được append vào file.
    - Exception không bắt được cũng được ghi kèm traceback vào file.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    for handler in root.handlers:
        if getattr(handler, "baseFilename", None) == str(log_path):
            break
    else:
        handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        handler.setLevel(logging.INFO)
        root.addHandler(handler)

    def _excepthook(exc_type, exc, tb):
        msg = "".join(traceback.format_exception(exc_type, exc, tb))
        logging.getLogger("unhandled").error("Lỗi không bắt được:\n%s", msg)
        print(msg, file=sys.stderr)

    sys.excepthook = _excepthook
    return log_path