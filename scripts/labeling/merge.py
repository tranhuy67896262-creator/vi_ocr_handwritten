"""Gộp dataset Hugging Face + ảnh cá nhân đã gắn nhãn thành ``DatasetDict``.

Có kiểm tra khớp schema (cột ảnh/text, kiểu Image, split) và báo lỗi rõ ràng
khi dataset HF chuẩn không khớp với ảnh cá nhân. Dùng trước khi train trên GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .csv_store import CsvTableRepository
from .domain import VALID_SPLITS, SourceUnavailable, ValidationError
from .hf_source import OVERRIDE_FIELDS
from .local_source import LocalImageSource

SOURCE_COLUMN = "source"
MAX_REPORTED_WARNINGS = 10


@dataclass(frozen=True)
class MergeConfig:  # pylint: disable=too-many-instance-attributes  # cấu hình gộp đầy đủ
    """Cấu hình bất biến cho quá trình gộp dataset."""

    hf_dataset_id: str
    token: str
    cache_dir: Path
    images_dir: Path
    csv_path: Path
    overrides_path: Path
    output_dir: Path
    source_local: str = "local"
    source_hf: str = "huggingface"


@dataclass
class MergeReport:
    """Kết quả gộp: số dòng mỗi split, số ảnh bỏ qua, cảnh báo."""

    counts: dict[str, int] = field(default_factory=dict)
    skipped_local: int = 0
    warnings: list[str] = field(default_factory=list)


class DatasetMerger:
    """Ghép base HF (đã áp override text) với dòng local thành dataset mới."""

    def __init__(self, config: MergeConfig) -> None:
        """Lưu cấu hình gộp."""
        self._config = config

    def merge(self, dry_run: bool = False) -> MergeReport:
        """Chạy gộp (hoặc chỉ kiểm tra nếu ``dry_run``), trả báo cáo."""
        report = MergeReport()
        base = self._load_base()
        overrides = self._read_overrides()
        local_by_split = self._collect_local(report)
        hf_splits, image_col, text_col = self._merge_hf(base, overrides, report)
        combined = self._append_local(hf_splits, local_by_split, image_col, text_col)
        report.counts = {split: len(dataset) for split, dataset in combined.items()}
        if not dry_run:
            self._save(combined)
        return report

    # ------------------------------------------------------------------ loading
    def _load_base(self) -> dict:
        from datasets import load_dataset

        try:
            bundle = load_dataset(self._config.hf_dataset_id, token=self._config.token or None)
        except Exception as error:  # noqa: BLE001 - báo lỗi thân thiện
            raise SourceUnavailable(
                f"Không tải được dataset HF '{self._config.hf_dataset_id}': {error}. "
                "Kiểm tra HF_TOKEN (dataset gated) và tên dataset."
            ) from error
        if hasattr(bundle, "items"):
            return dict(bundle.items())
        return {"train": bundle}

    def _read_overrides(self) -> dict[str, str]:
        repo = CsvTableRepository(self._config.overrides_path, OVERRIDE_FIELDS)
        return {row["ref"]: row.get("text", "") for row in repo.read() if row.get("ref")}

    def _collect_local(self, report: MergeReport) -> dict[str, list]:
        source = LocalImageSource(self._config.images_dir, self._config.csv_path)
        grouped: dict[str, list] = {}
        for record in source.list_records():
            if record.split not in VALID_SPLITS:
                raise ValidationError(
                    f"split không hợp lệ cho '{record.image_ref}': '{record.split}' "
                    "(chỉ nhận train hoặc test)."
                )
            if not (record.text or "").strip():
                self._skip(report, f"Bỏ qua '{record.image_ref}': chưa có nhãn")
                continue
            if not (self._config.images_dir / record.image_ref).is_file():
                self._skip(report, f"Bỏ qua '{record.image_ref}': thiếu file ảnh")
                continue
            grouped.setdefault(record.split, []).append(record)
        if not grouped:
            raise ValidationError(
                f"Không có ảnh cá nhân hợp lệ (có nhãn + đủ file) trong {self._config.csv_path}."
            )
        return grouped

    # ------------------------------------------------------------------ merging
    def _merge_hf(self, base: dict, overrides: dict[str, str], report: MergeReport):
        from datasets import Image as HFImage

        from src.datasets.dataset import detect_columns

        if "train" not in base:
            self._warn(
                report,
                f"Dataset HF không có split 'train' (chỉ có {list(base.keys())}) — training sẽ lỗi.",
            )
        result: dict[str, object] = {}
        image_col, text_col = "image", "text"
        for split_name, dataset in base.items():
            image_col, text_col = self._detect_columns(dataset, split_name, detect_columns, HFImage)
            texts = list(dataset[text_col])
            for index, _ in enumerate(texts):
                ref = f"{split_name}/{index}"
                if ref in overrides:
                    texts[index] = overrides[ref]
            dataset = dataset.remove_columns([text_col]).add_column(text_col, texts)
            if SOURCE_COLUMN in dataset.column_names:
                dataset = dataset.remove_columns([SOURCE_COLUMN])
            result[split_name] = dataset.add_column(SOURCE_COLUMN, [self._config.source_hf] * len(dataset))
        return result, image_col, text_col

    @staticmethod
    def _detect_columns(dataset, split_name: str, detect_columns, image_feature):
        try:
            image_col, text_col = detect_columns(dataset)
        except Exception as error:  # noqa: BLE001 - thiếu cột là lỗi dataset
            raise ValidationError(
                f"Split '{split_name}' của dataset HF không có cặp cột ảnh/text "
                f"(các cột: {list(dataset.column_names)}). Dataset không khớp pipeline."
            ) from error
        feature = dataset.features[image_col]
        if not isinstance(feature, image_feature):
            raise ValidationError(
                f"Cột ảnh '{image_col}' của split '{split_name}' có kiểu {feature}, "
                "không phải Image — dataset HF không khớp."
            )
        return image_col, text_col

    def _append_local(self, hf_splits, grouped, image_col, text_col):
        from datasets import Image, concatenate_datasets

        combined = dict(hf_splits)
        for split_name, records in grouped.items():
            local = self._build_local(records, image_col, text_col, Image())
            target = combined.get(split_name)
            if target is None:
                combined[split_name] = local
                continue
            local = _align_columns(local, target.column_names)
            try:
                combined[split_name] = concatenate_datasets([target, local])
            except Exception as error:  # noqa: BLE001 - cột không khớp giữa HF và local
                raise ValidationError(
                    f"Không ghép được split '{split_name}': cột HF {target.column_names} "
                    f"không khớp cột local {local.column_names}: {error}"
                ) from error
        return combined

    def _build_local(self, records, image_col, text_col, image_feature):
        from datasets import Dataset

        paths = [str((self._config.images_dir / record.image_ref).resolve()) for record in records]
        local = Dataset.from_dict({image_col: paths, text_col: [record.text for record in records]})
        local = local.cast_column(image_col, image_feature)
        return local.add_column(SOURCE_COLUMN, [self._config.source_local] * len(local))

    def _save(self, combined) -> None:
        from datasets import DatasetDict

        self._config.output_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            DatasetDict(combined).save_to_disk(str(self._config.output_dir))
        except Exception as error:  # noqa: BLE001 - lỗi ghi đĩa
            raise ValidationError(
                f"Không lưu được dataset vào {self._config.output_dir}: {error}"
            ) from error

    @staticmethod
    def _warn(report: MergeReport, message: str) -> None:
        if len(report.warnings) < MAX_REPORTED_WARNINGS:
            report.warnings.append(message)

    def _skip(self, report: MergeReport, message: str) -> None:
        report.skipped_local += 1
        self._warn(report, message)


def _align_columns(dataset, columns: list[str]):
    """Bổ sung cột thiếu (rỗng) rồi sắp xếp đúng thứ tự cột đích."""
    for column in columns:
        if column not in dataset.column_names:
            dataset = dataset.add_column(column, [""] * len(dataset))
    return dataset.select_columns(columns)
