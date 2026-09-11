"""Các giao diện (ABC) cho nguồn dữ liệu.

Áp dụng ISP: tách quyền đọc / sửa text / sửa split / thêm-xóa thành các
interface nhỏ để nguồn chỉ phải hiện thực đúng năng lực của mình (LSP).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .domain import LabelRecord

# Chữ ký interface cố định, implementation dùng tham số — pylint báo oan.
# pylint: disable=unused-argument


class LabelSource(ABC):
    """Nguồn dữ liệu chỉ đọc."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Định danh ngắn của nguồn (dùng trong uid và tham số API)."""

    @property
    @abstractmethod
    def label(self) -> str:
        """Tên hiển thị tiếng Việt cho UI."""

    @abstractmethod
    def is_available(self) -> bool:
        """Nguồn có sẵn sàng phục vụ không (không tải dữ liệu nặng)."""

    @abstractmethod
    def availability_error(self) -> str:
        """Mô tả lý do nguồn chưa sẵn sàng; rỗng nếu bình thường."""

    @abstractmethod
    def count(self, query: str = "", split: str = "") -> int:
        """Số bản ghi khớp bộ lọc."""

    @abstractmethod
    def list_records(
        self,
        query: str = "",
        split: str = "",
        offset: int = 0,
        limit: int | None = None,
    ) -> list[LabelRecord]:
        """Danh sách bản ghi (phân trang) khớp bộ lọc."""

    @abstractmethod
    def image(self, image_ref: str) -> tuple[bytes, str]:
        """Trả ``(bytes ảnh, content-type)`` theo ref nội bộ của nguồn."""


class TextEditableSource(LabelSource, ABC):
    """Nguồn cho phép sửa phần text của bản ghi."""

    @abstractmethod
    def update_text(self, image_ref: str, text: str) -> None:
        """Ghi đè text cho bản ghi."""


class SplitEditableSource(LabelSource, ABC):
    """Nguồn cho phép đổi split train/test của bản ghi."""

    @abstractmethod
    def update_split(self, image_ref: str, split: str) -> None:
        """Đổi split cho bản ghi."""


class MutableSource(TextEditableSource, SplitEditableSource, ABC):
    """Nguồn đầy đủ: thêm ảnh mới và xóa bản ghi."""

    @abstractmethod
    def add_record(self, filename: str, content: bytes, text: str, split: str) -> LabelRecord:
        """Thêm ảnh mới + dòng nhãn, trả bản ghi vừa tạo."""

    @abstractmethod
    def delete_record(self, image_ref: str) -> bool:
        """Xóa bản ghi (và file ảnh nếu có); trả True nếu xóa được ảnh."""
