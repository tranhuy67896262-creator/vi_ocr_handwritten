import logging
import sys
import traceback
from pathlib import Path

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


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