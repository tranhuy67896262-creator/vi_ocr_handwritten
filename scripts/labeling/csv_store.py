"""Repository CSV dùng chung: đọc/ghi nguyên tử kèm backup tự động.

SRP: chỉ lo việc lưu trữ bảng CSV, không biết gì về nghiệp vụ gắn nhãn.
"""

from __future__ import annotations

import csv
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path


class CsvTableRepository:
    """Bọc một file CSV (backup bản cũ rồi thay thế nguyên tử khi ghi)."""

    def __init__(self, path: Path, fieldnames: list[str]) -> None:
        """Khởi tạo repository cho file CSV với các cột cho trước."""
        self.path = Path(path)
        self.fieldnames = list(fieldnames)

    def exists(self) -> bool:
        """True nếu file CSV đã tồn tại."""
        return self.path.is_file()

    def read(self) -> list[dict[str, str]]:
        """Đọc toàn bộ dòng; trả ``[]`` nếu file chưa tồn tại."""
        if not self.path.is_file():
            return []
        try:
            with self.path.open(newline="", encoding="utf-8-sig") as handle:
                return [dict(row) for row in csv.DictReader(handle)]
        except (OSError, csv.Error, UnicodeDecodeError) as error:
            raise ValueError(f"Không đọc được CSV {self.path}: {error}") from error

    def write(self, rows: list[dict[str, str]]) -> str | None:
        """Ghi đè CSV; trả tên file backup (hoặc None nếu chưa từng có file)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        backup = self._backup()
        fd, temp_name = tempfile.mkstemp(prefix=f"{self.path.name}.", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
            os.replace(temp_name, self.path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        return backup.name if backup else None

    def _backup(self) -> Path | None:
        if not self.path.is_file():
            return None
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = self.path.with_name(f"{self.path.name}.{stamp}.bak")
        shutil.copy2(self.path, backup)
        return backup
