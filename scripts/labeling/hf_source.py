"""Nguồn dữ liệu Hugging Face: đọc dataset (lazy) + sửa text qua file override."""

from __future__ import annotations

import mimetypes
import os
import threading
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

from .csv_store import CsvTableRepository
from .domain import LabelRecord, RecordNotFound, SourceUnavailable
from .interfaces import TextEditableSource

OVERRIDE_FIELDS = ["ref", "text"]


@dataclass(frozen=True)
class HfConfig:
    """Cấu hình bất biến cho nguồn Hugging Face (truyền vào từ composition root)."""

    dataset_id: str
    token: str
    cache_dir: Path
    overrides_path: Path
    source_name: str = "huggingface"
    label: str = "Hugging Face"


@dataclass
class _HfState:
    """Trạng thái nội bộ sau khi tải dataset (tách khỏi logic để giữ class gọn)."""

    datasets: dict[str, object] = field(default_factory=dict)
    texts: dict[str, list[str]] = field(default_factory=dict)
    image_cols: dict[str, str] = field(default_factory=dict)
    overrides: dict[str, str] = field(default_factory=dict)
    loaded: bool = False
    error: str = ""


class HuggingFaceSource(TextEditableSource):
    """Dataset HF chỉ đọc về ảnh; cho phép sửa text lưu vào ``hf_overrides.csv``."""

    def __init__(self, config: HfConfig) -> None:
        """Khởi tạo nguồn HF từ cấu hình."""
        self._config = config
        self._repo = CsvTableRepository(config.overrides_path, OVERRIDE_FIELDS)
        self._state = _HfState(overrides=self._load_overrides())
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        """Định danh nguồn."""
        return self._config.source_name

    @property
    def label(self) -> str:
        """Tên hiển thị."""
        return self._config.label

    def is_available(self) -> bool:
        """Sẵn sàng nếu chưa gặp lỗi tải (chưa tải vẫn coi là sẵn sàng)."""
        return not self._state.error

    def availability_error(self) -> str:
        """Lỗi tải dataset nếu có."""
        return self._state.error

    def count(self, query: str = "", split: str = "") -> int:
        """Số bản ghi khớp bộ lọc (đã áp override text)."""
        self._ensure_loaded()
        return sum(1 for _ in self._iter_matches(query, split))

    def list_records(
        self,
        query: str = "",
        split: str = "",
        offset: int = 0,
        limit: int | None = None,
    ) -> list[LabelRecord]:
        """Danh sách bản ghi HF (phân trang) khớp bộ lọc."""
        self._ensure_loaded()
        start = max(offset, 0)
        stop = None if limit is None else start + limit
        return [
            self._to_record(ref, text, split_name)
            for ref, text, split_name in _window(self._iter_matches(query, split), start, stop)
        ]

    def image(self, image_ref: str) -> tuple[bytes, str]:
        """Giải mã ảnh từ dataset theo ref ``<split>/<index>``."""
        self._ensure_loaded()
        split_name, _, index_str = image_ref.rpartition("/")
        dataset = self._state.datasets.get(split_name)
        if dataset is None or not index_str.isdigit():
            raise RecordNotFound(f"Ref ảnh không hợp lệ: {image_ref}")
        index = int(index_str)
        if index < 0 or index >= len(dataset):
            raise RecordNotFound(f"Ref ảnh ngoài phạm vi: {image_ref}")
        column = self._state.image_cols[split_name]
        value = dataset[index][column]
        return _encode_image(value)

    def update_text(self, image_ref: str, text: str) -> None:
        """Lưu text override cho bản ghi HF (không đụng dataset gốc)."""
        rows = self._repo.read()
        for row in rows:
            if row.get("ref") == image_ref:
                row["text"] = text or ""
                break
        else:
            rows.append({"ref": image_ref, "text": text or ""})
        self._repo.write(rows)
        self._state.overrides[image_ref] = text or ""

    # ------------------------------------------------------------------ helpers
    def _load_overrides(self) -> dict[str, str]:
        return {
            row["ref"]: row.get("text", "")
            for row in self._repo.read()
            if row.get("ref")
        }

    def _ensure_loaded(self) -> None:
        if self._state.loaded:
            return
        if self._state.error:
            raise SourceUnavailable(self._state.error)
        with self._lock:
            if self._state.loaded:
                return
            self._load_dataset()

    def _load_dataset(self) -> None:
        try:
            from datasets import load_dataset

            from src.datasets.dataset import detect_columns

            self._config.cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_DATASETS_CACHE", str(self._config.cache_dir))
            bundle = load_dataset(self._config.dataset_id, token=self._config.token or None)
            for split_name, dataset in bundle.items():
                image_col, text_col = detect_columns(dataset)
                self._state.datasets[split_name] = dataset
                self._state.image_cols[split_name] = image_col
                self._state.texts[split_name] = list(dataset[text_col])
            self._state.loaded = True
        except SourceUnavailable:
            raise
        except Exception as error:  # noqa: BLE001 - báo lỗi thân thiện cho UI
            self._state.error = f"Không tải được {self._config.dataset_id}: {error}"
            raise SourceUnavailable(self._state.error) from error

    def _iter_matches(self, query: str, split: str):
        query_l = query.strip().lower()
        for split_name, texts in self._state.texts.items():
            if split and split_name != split:
                continue
            for index, default_text in enumerate(texts):
                ref = f"{split_name}/{index}"
                text = self._state.overrides.get(ref, default_text)
                if query_l and query_l not in text.lower():
                    continue
                yield ref, text, split_name

    def _to_record(self, ref: str, text: str, split_name: str) -> LabelRecord:
        return LabelRecord(
            uid=LabelRecord.build_uid(self._config.source_name, ref),
            source=self._config.source_name,
            image_ref=ref,
            text=text,
            split=split_name,
            can_edit_text=True,
            can_edit_split=False,
            can_delete=False,
        )


def _window(iterable, start: int, stop: int | None):
    """Bỏ qua ``start`` phần tử rồi lấy tối đa ``stop - start`` phần tử."""
    for position, item in enumerate(iterable):
        if position < start:
            continue
        if stop is not None and position >= stop:
            return
        yield item


def _encode_image(value) -> tuple[bytes, str]:
    """Chuẩn hóa ảnh HF (PIL / dict bytes / dict path) về PNG/bytes."""
    if hasattr(value, "save") and hasattr(value, "convert"):
        buffer = BytesIO()
        value.convert("RGB").save(buffer, format="PNG")
        return buffer.getvalue(), "image/png"
    if isinstance(value, dict):
        raw = value.get("bytes")
        if raw:
            path = value.get("path") or ""
            return raw, mimetypes.guess_type(path)[0] or "image/png"
        path = value.get("path")
        if path:
            return Path(path).read_bytes(), mimetypes.guess_type(path)[0] or "image/png"
    raise SourceUnavailable("Không giải mã được ảnh từ dataset")
