"""Thực thể và lỗi nghiệp vụ của module labeling.

Không phụ thuộc hạ tầng (CSV, Hugging Face, HTTP) — chỉ mô tả dữ liệu.
"""

from __future__ import annotations

from dataclasses import dataclass

VALID_SPLITS = ("train", "test")


class LabelError(Exception):
    """Lỗi nghiệp vụ cơ sở của module labeling."""


class ValidationError(LabelError):
    """Dữ liệu đầu vào không hợp lệ."""


class RecordNotFound(LabelError):
    """Không tìm thấy bản ghi theo uid."""


class SourceUnavailable(LabelError):
    """Nguồn dữ liệu chưa sẵn sàng (vd chưa tải được dataset HF)."""


@dataclass(slots=True)
class LabelRecord:  # pylint: disable=too-many-instance-attributes  # 3 cờ năng lực là cố ý
    """Một mẫu dữ liệu gắn nhãn, độc lập với nguồn lưu trữ.

    ``uid`` là định danh toàn cục dạng ``"<source>:<image_ref>"``.
    Các cờ ``can_*`` cho biết thao tác nào được phép trên bản ghi này.
    """

    # pylint: disable=too-many-instance-attributes  # 3 cờ năng lực là cố ý
    uid: str
    source: str
    image_ref: str
    text: str
    split: str
    can_edit_text: bool = False
    can_edit_split: bool = False
    can_delete: bool = False

    @staticmethod
    def build_uid(source: str, image_ref: str) -> str:
        """Ghép uid duy nhất từ tên nguồn và ref ảnh."""
        return f"{source}:{image_ref}"

    @staticmethod
    def split_uid(uid: str) -> tuple[str, str]:
        """Tách uid thành ``(tên nguồn, ref ảnh)``."""
        source, _, image_ref = uid.partition(":")
        return source, image_ref
