"""Nguồn dữ liệu local: thư mục ảnh + ``labels.csv`` (đọc/ghi đầy đủ)."""

from __future__ import annotations

import mimetypes
import re
from pathlib import Path

from .csv_store import CsvTableRepository
from .domain import VALID_SPLITS, LabelRecord, RecordNotFound, ValidationError
from .interfaces import MutableSource

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
FIELDNAMES = ["image", "text", "split"]


class LocalImageSource(MutableSource):
    """Quản lý ảnh cá nhân + ``labels.csv``; mỗi ảnh luôn có đúng một dòng."""

    def __init__(
        self,
        images_dir: Path,
        csv_path: Path,
        source_name: str = "local",
        label: str = "Ảnh cá nhân",
    ) -> None:
        """Khởi tạo nguồn local; tạo thư mục ảnh nếu cần."""
        self.images_dir = Path(images_dir).resolve()
        self.csv_path = Path(csv_path).resolve()
        self._name = source_name
        self._label = label
        self._repo = CsvTableRepository(self.csv_path, FIELDNAMES)
        self.images_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        """Định danh nguồn."""
        return self._name

    @property
    def label(self) -> str:
        """Tên hiển thị."""
        return self._label

    def is_available(self) -> bool:
        """Nguồn local luôn sẵn sàng."""
        return True

    def availability_error(self) -> str:
        """Local không bao giờ lỗi sẵn sàng."""
        return ""

    def count(self, query: str = "", split: str = "") -> int:
        """Số dòng khớp bộ lọc."""
        query_l = query.strip().lower()
        total = 0
        for row in self._sync_rows():
            if not self._match(row, query_l, split):
                continue
            total += 1
        return total

    def list_records(
        self,
        query: str = "",
        split: str = "",
        offset: int = 0,
        limit: int | None = None,
    ) -> list[LabelRecord]:
        """Danh sách bản ghi local khớp bộ lọc."""
        query_l = query.strip().lower()
        matched = [row for row in self._sync_rows() if self._match(row, query_l, split)]
        window = matched[max(offset, 0):]
        if limit is not None:
            window = window[:limit]
        return [self._to_record(row) for row in window]

    def image(self, image_ref: str) -> tuple[bytes, str]:
        """Đọc bytes ảnh theo đường dẫn tương đối."""
        path = self._resolve(image_ref)
        if not path.is_file():
            raise RecordNotFound(f"Không tìm thấy ảnh: {image_ref}")
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return path.read_bytes(), content_type

    def update_text(self, image_ref: str, text: str) -> None:
        """Cập nhật text cho dòng tương ứng."""
        rows = self._sync_rows()
        index = self._find(rows, image_ref)
        if index is None:
            raise RecordNotFound(f"Không có dòng cho ảnh: {image_ref}")
        rows[index]["text"] = text if text is not None else ""
        self._repo.write(rows)

    def update_split(self, image_ref: str, split: str) -> None:
        """Cập nhật split cho dòng tương ứng."""
        if split not in VALID_SPLITS:
            raise ValidationError("split chỉ được là train hoặc test")
        rows = self._sync_rows()
        index = self._find(rows, image_ref)
        if index is None:
            raise RecordNotFound(f"Không có dòng cho ảnh: {image_ref}")
        rows[index]["split"] = split
        self._repo.write(rows)

    def add_record(self, filename: str, content: bytes, text: str, split: str) -> LabelRecord:
        """Lưu ảnh upload xuống thư mục rồi thêm dòng nhãn."""
        if split not in VALID_SPLITS:
            raise ValidationError("split chỉ được là train hoặc test")
        if not content:
            raise ValidationError("File ảnh rỗng")
        if len(content) > MAX_UPLOAD_BYTES:
            raise ValidationError("File ảnh quá lớn (tối đa 20MB)")
        target = self._unique_target(filename)
        target.write_bytes(content)
        try:
            rows = self._sync_rows()
            image_ref = target.relative_to(self.images_dir).as_posix()
            rows.append({"image": image_ref, "text": text or "", "split": split})
            self._repo.write(rows)
        except Exception:
            target.unlink(missing_ok=True)
            raise
        return self._to_record({"image": image_ref, "text": text or "", "split": split})

    def delete_record(self, image_ref: str) -> bool:
        """Xóa dòng khỏi CSV (và file ảnh nếu không còn dòng nào trỏ tới)."""
        rows = self._sync_rows()
        index = self._find(rows, image_ref)
        if index is None:
            raise RecordNotFound(f"Không có dòng cho ảnh: {image_ref}")
        removed = rows.pop(index)
        self._repo.write(rows)
        ref = Path(removed.get("image", "")).as_posix()
        if not ref or any(Path(row.get("image", "")).as_posix() == ref for row in rows):
            return False
        candidate = self._resolve(ref)
        if candidate.is_file():
            candidate.unlink()
            return True
        return False

    # ------------------------------------------------------------------ helpers
    def _disk_images(self) -> list[str]:
        return sorted(
            path.relative_to(self.images_dir).as_posix()
            for path in self.images_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    def _sync_rows(self) -> list[dict[str, str]]:
        rows = self._repo.read()
        known = {Path(row.get("image", "")).as_posix() for row in rows}
        added = False
        for image_ref in self._disk_images():
            if image_ref not in known:
                rows.append({"image": image_ref, "text": "", "split": "train"})
                added = True
        if added or not self._repo.exists():
            self._repo.write(rows)
        return rows

    @staticmethod
    def _match(row: dict[str, str], query_l: str, split: str) -> bool:
        if split and (row.get("split") or "train") != split:
            return False
        if query_l and query_l not in f"{row.get('image', '')} {row.get('text', '')}".lower():
            return False
        return True

    @staticmethod
    def _find(rows: list[dict[str, str]], image_ref: str) -> int | None:
        normalized = Path(image_ref).as_posix()
        for index, row in enumerate(rows):
            if Path(row.get("image", "")).as_posix() == normalized:
                return index
        return None

    def _to_record(self, row: dict[str, str]) -> LabelRecord:
        image_ref = Path(row.get("image", "")).as_posix()
        return LabelRecord(
            uid=LabelRecord.build_uid(self._name, image_ref),
            source=self._name,
            image_ref=image_ref,
            text=row.get("text", ""),
            split=row.get("split") or "train",
            can_edit_text=True,
            can_edit_split=True,
            can_delete=True,
        )

    def _resolve(self, image_ref: str) -> Path:
        candidate = (self.images_dir / image_ref).resolve()
        try:
            candidate.relative_to(self.images_dir)
        except ValueError as error:
            raise ValidationError("Đường dẫn ảnh không hợp lệ") from error
        return candidate

    def _unique_target(self, filename: str) -> Path:
        name = Path(filename).name
        if "." in name:
            stem, ext = name.rsplit(".", 1)
            ext = f".{ext.lower()}"
        else:
            stem, ext = name, ""
        if ext not in IMAGE_EXTENSIONS:
            raise ValidationError(f"Định dạng ảnh không hỗ trợ: {ext or '(không có đuôi)'}")
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "image"
        target = self.images_dir / f"{stem}{ext}"
        counter = 1
        while target.exists():
            counter += 1
            target = self.images_dir / f"{stem}_{counter}{ext}"
        return target
