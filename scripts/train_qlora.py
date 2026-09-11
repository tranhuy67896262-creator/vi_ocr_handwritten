"""Entry cũ (deprecated): chuyển sang ``scripts/train.py``.

Giữ lại để các lệnh/tài liệu cũ không gãy. Nên dùng ``python scripts/train.py``
(trainer LoRA tổng quát, hỗ trợ cả 4-bit QLoRA lẫn bf16).
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import main  # noqa: E402  (phải chèn sys.path trước)


if __name__ == "__main__":
    main()
