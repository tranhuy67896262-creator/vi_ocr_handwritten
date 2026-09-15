"""Registry lịch sử train/eval (SQLite, stdlib-only).

Mỗi lần train (hay eval) ghi 1 dòng vào ``models/runs.db`` để tra cứu sau:
run nào đã train với config gì, adapter nằm đâu (local + repo/revision trên Hub),
CER/WER ra sao. Nguồn chân lý để *tiếp tục* train vẫn là adapter trên HF Hub —
registry chỉ là tiện tra cứu cục bộ, không bắt buộc sync qua máy khác.
"""
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_name TEXT,
    created_at TEXT,
    model TEXT,
    dataset TEXT,
    adapter_dir TEXT,
    init_adapter TEXT,
    hub_repo TEXT,
    hub_revision TEXT,
    train_samples INTEGER,
    eval_samples INTEGER,
    train_steps INTEGER,
    num_epochs INTEGER,
    batch_size INTEGER,
    gradient_accumulation_steps INTEGER,
    learning_rate REAL,
    max_seq_len INTEGER,
    lora_r INTEGER,
    lora_alpha INTEGER,
    use_4bit INTEGER,
    kl_regularization INTEGER,
    cer REAL,
    wer REAL
);
"""

RUN_FIELDS = [
    "run_name", "created_at", "model", "dataset", "adapter_dir", "init_adapter",
    "hub_repo", "hub_revision", "train_samples", "eval_samples", "train_steps",
    "num_epochs", "batch_size", "gradient_accumulation_steps", "learning_rate",
    "max_seq_len", "lora_r", "lora_alpha", "use_4bit", "kl_regularization",
    "cer", "wer",
]


@contextmanager
def _db(db_path):
    """Mở kết nối, commit khi thoát, luôn đóng."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_db(db_path):
    """Tạo bảng ``runs`` nếu chưa tồn tại."""
    with _db(db_path) as conn:
        conn.execute(_SCHEMA)


def add_run(db_path, record):
    """Thêm 1 dòng lịch sử. ``record`` là dict, key khớp ``RUN_FIELDS``."""
    record = dict(record)
    record.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
    cols = [f for f in RUN_FIELDS if f in record]
    placeholders = ", ".join("?" for _ in cols)
    sql = f"INSERT INTO runs ({', '.join(cols)}) VALUES ({placeholders})"
    with _db(db_path) as conn:
        cur = conn.execute(sql, [record[c] for c in cols])
        return cur.lastrowid


def update_metrics(db_path, run_name, cer=None, wer=None):
    """Ghi CER/WER vào run gần nhất có ``run_name`` (eval chạy sau train)."""
    init_db(db_path)
    with _db(db_path) as conn:
        row = conn.execute(
            "SELECT id FROM runs WHERE run_name = ? ORDER BY id DESC LIMIT 1",
            (run_name,),
        ).fetchone()
        if row is None:
            return None
        conn.execute(
            "UPDATE runs SET cer = ?, wer = ? WHERE id = ?", (cer, wer, row["id"])
        )
        return row["id"]


def list_runs(db_path, limit=20):
    """Trả các run gần nhất (mới trước) dưới dạng list dict."""
    init_db(db_path)
    with _db(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
