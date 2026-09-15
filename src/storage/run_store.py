"""Registry lịch sử train/eval (JSON, stdlib-only).

1 file JSON = 1 list các run. ``RunStore`` đóng gói đọc/ghi; nguồn chân lý để
*tiếp tục* train vẫn là adapter trên HF Hub — registry chỉ là tiện tra cứu cục bộ.
"""
import json
import os
from datetime import datetime
from pathlib import Path

RUN_FIELDS = [
    "run_name", "created_at", "model", "dataset", "adapter_dir", "init_adapter",
    "hub_repo", "hub_revision", "train_samples", "eval_samples", "train_steps",
    "num_epochs", "batch_size", "gradient_accumulation_steps", "learning_rate",
    "max_seq_len", "lora_r", "lora_alpha", "use_4bit", "kl_regularization",
    "cer", "wer",
]


class RunStore:
    """Quản lý lịch sử train/eval trong 1 file JSON (list các run)."""

    def __init__(self, path):
        self.path = Path(path)

    def _load(self):
        """Đọc list run ([] nếu chưa có / file lỗi)."""
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def _save(self, runs):
        """Ghi list run (atomic: temp rồi replace, chống hỏng khi đứt giữa chừng)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(runs, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def add(self, record):
        """Thêm 1 run, trả về ``id`` (1-based) của nó."""
        record = dict(record)
        record.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
        runs = self._load()
        record["id"] = max((r.get("id", 0) for r in runs), default=0) + 1
        runs.append(record)
        self._save(runs)
        return record["id"]

    def update_metrics(self, run_name, cer=None, wer=None):
        """Ghi CER/WER vào run gần nhất có ``run_name``. Trả về id hoặc None."""
        runs = self._load()
        for run in reversed(runs):
            if run.get("run_name") == run_name:
                run["cer"] = cer
                run["wer"] = wer
                self._save(runs)
                return run.get("id")
        return None

    def list(self, limit=20):
        """Trả các run gần nhất (mới trước), mỗi run đủ field (None nếu thiếu)."""
        runs = self._load()
        recent = list(reversed(runs[-limit:]))
        return [{"id": r.get("id"), **{f: r.get(f) for f in RUN_FIELDS}} for r in recent]
