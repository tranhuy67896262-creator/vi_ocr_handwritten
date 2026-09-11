"""Sổ đăng ký nguồn dữ liệu.

OCP: thêm nguồn mới chỉ cần tạo class hiện thực ``LabelSource`` và đăng ký,
không phải sửa service hay web layer.
"""

from __future__ import annotations

from collections.abc import Iterable

from .domain import RecordNotFound, LabelRecord
from .interfaces import LabelSource


class DatasetRegistry:
    """Tập hợp các nguồn dữ liệu, tra cứu theo tên hoặc uid."""

    def __init__(self, sources: Iterable[LabelSource]) -> None:
        """Đăng ký các nguồn; tên nguồn phải duy nhất."""
        self._sources: dict[str, LabelSource] = {}
        for source in sources:
            self._sources[source.name] = source

    def all(self) -> list[LabelSource]:
        """Danh sách nguồn theo thứ tự đăng ký."""
        return list(self._sources.values())

    def default(self) -> LabelSource:
        """Nguồn mặc định (đầu tiên); báo lỗi nếu registry rỗng."""
        sources = self.all()
        if not sources:
            raise RecordNotFound("Chưa cấu hình nguồn dữ liệu nào")
        return sources[0]

    def get(self, name: str) -> LabelSource:
        """Lấy nguồn theo tên."""
        source = self._sources.get(name)
        if source is None:
            raise RecordNotFound(f"Không có nguồn: {name}")
        return source

    def resolve(self, uid: str) -> tuple[LabelSource, str]:
        """Tách uid thành ``(nguồn, ref ảnh)``."""
        name, image_ref = LabelRecord.split_uid(uid)
        return self.get(name), image_ref
