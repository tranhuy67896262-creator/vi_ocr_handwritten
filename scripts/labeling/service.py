"""Tầng ứng dụng: điều phối nguồn dữ liệu và tạo DTO cho web layer.

Web layer chỉ biết class này (DIP), không biết CSV/Hugging Face cụ thể.
"""

from __future__ import annotations

from urllib.parse import quote

from .domain import LabelRecord, ValidationError
from .interfaces import MutableSource, SplitEditableSource, TextEditableSource
from .registry import DatasetRegistry


class LabelService:
    """Use-case gắn nhãn: liệt kê, sửa, thêm, xóa, lấy ảnh."""

    def __init__(
        self,
        registry: DatasetRegistry,
        image_endpoint: str = "/api/image",
        max_limit: int = 200,
    ) -> None:
        """Khởi tạo service với sổ đăng ký nguồn và cấu hình phân trang."""
        self._registry = registry
        self._image_endpoint = image_endpoint
        self._max_limit = max_limit

    def describe_sources(self) -> list[dict[str, object]]:
        """Thông tin mô tả từng nguồn cho UI."""
        return [
            {
                "name": source.name,
                "label": source.label,
                "available": source.is_available(),
                "error": source.availability_error(),
                "can_add": isinstance(source, MutableSource),
            }
            for source in self._registry.all()
        ]

    def list_records(
        self,
        source: str = "",
        query: str = "",
        split: str = "",
        offset: int = 0,
        limit: int | None = None,
    ) -> dict[str, object]:
        """Liệt kê bản ghi của một nguồn (mặc định nguồn đầu tiên)."""
        selected = self._registry.get(source) if source else self._registry.default()
        offset = max(offset, 0)
        effective_limit = self._max_limit if limit is None else min(max(limit, 1), self._max_limit)
        total = selected.count(query, split)
        rows = selected.list_records(query, split, offset, effective_limit)
        return {
            "source": selected.name,
            "total": total,
            "offset": offset,
            "limit": effective_limit,
            "has_more": offset + len(rows) < total,
            "rows": [self._public(record) for record in rows],
        }

    def get_image(self, uid: str) -> tuple[bytes, str]:
        """Trả ảnh theo uid toàn cục."""
        source, image_ref = self._registry.resolve(uid)
        return source.image(image_ref)

    def update_text(self, uid: str, text: str) -> dict[str, object]:
        """Cập nhật text cho bản ghi."""
        source, image_ref = self._registry.resolve(uid)
        if not isinstance(source, TextEditableSource):
            raise ValidationError("Nguồn này không cho sửa text")
        source.update_text(image_ref, text or "")
        return {"uid": uid, "text": text or ""}

    def update_split(self, uid: str, split: str) -> dict[str, object]:
        """Cập nhật split cho bản ghi."""
        source, image_ref = self._registry.resolve(uid)
        if not isinstance(source, SplitEditableSource):
            raise ValidationError("Nguồn này không cho đổi split")
        source.update_split(image_ref, split)
        return {"uid": uid, "split": split}

    def add_record(
        self,
        source: str,
        filename: str,
        content: bytes,
        text: str,
        split: str,
    ) -> dict[str, object]:
        """Thêm ảnh mới vào nguồn (chỉ nguồn mutable)."""
        selected = self._registry.get(source) if source else self._registry.default()
        if not isinstance(selected, MutableSource):
            raise ValidationError("Nguồn này không cho thêm ảnh")
        record = selected.add_record(filename, content, text, split)
        return self._public(record)

    def delete_record(self, uid: str) -> dict[str, object]:
        """Xóa bản ghi (chỉ nguồn mutable)."""
        source, image_ref = self._registry.resolve(uid)
        if not isinstance(source, MutableSource):
            raise ValidationError("Nguồn này không cho xóa")
        deleted_image = source.delete_record(image_ref)
        return {"uid": uid, "deleted_image": deleted_image}

    def _public(self, record: LabelRecord) -> dict[str, object]:
        return {
            "uid": record.uid,
            "source": record.source,
            "image_ref": record.image_ref,
            "image_url": f"{self._image_endpoint}?uid={quote(record.uid, safe='')}",
            "text": record.text,
            "split": record.split,
            "can_edit_text": record.can_edit_text,
            "can_edit_split": record.can_edit_split,
            "can_delete": record.can_delete,
        }
