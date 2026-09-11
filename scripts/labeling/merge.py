"""Gộp dataset Hugging Face + ảnh cá nhân đã gắn nhãn thành ``DatasetDict``.

Dùng cho bước build trước khi train trên server GPU mạnh (xem ``run_label_train.sh``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .csv_store import CsvTableRepository
from .hf_source import OVERRIDE_FIELDS
from .local_source import LocalImageSource


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


class DatasetMerger:
    """Ghép base HF (đã áp override text) với dòng local thành dataset mới."""

    def __init__(self, config: MergeConfig) -> None:
        """Lưu cấu hình gộp."""
        self._config = config

    def merge(self) -> dict[str, int]:
        """Chạy gộp, lưu ra ``output_dir`` và trả số dòng mỗi split."""
        from datasets import DatasetDict, Image, concatenate_datasets, load_dataset

        from src.datasets.dataset import detect_columns

        config = self._config
        base = load_dataset(config.hf_dataset_id, token=config.token or None)
        overrides = self._read_overrides()
        local_records = LocalImageSource(config.images_dir, config.csv_path).list_records()

        image_col, text_col = "image", "text"
        result: dict[str, object] = {}
        for split_name, dataset in base.items():
            image_col, text_col = detect_columns(dataset)
            texts = list(dataset[text_col])
            for index, _ in enumerate(texts):
                ref = f"{split_name}/{index}"
                if ref in overrides:
                    texts[index] = overrides[ref]
            dataset = dataset.remove_columns([text_col]).add_column(text_col, texts)
            if "source" in dataset.column_names:
                dataset = dataset.remove_columns(["source"])
            result[split_name] = dataset.add_column("source", [config.source_hf] * len(dataset))

        for split_name, records in self._group_local(local_records).items():
            local = self._build_local(records, image_col, text_col, Image())
            target = result.get(split_name)
            if target is not None:
                local = _align_columns(local, target.column_names)
                result[split_name] = concatenate_datasets([target, local])
            else:
                result[split_name] = local

        output = DatasetDict(result)
        output.save_to_disk(str(config.output_dir))
        return {split: len(dataset) for split, dataset in output.items()}

    def _read_overrides(self) -> dict[str, str]:
        repo = CsvTableRepository(self._config.overrides_path, OVERRIDE_FIELDS)
        return {row["ref"]: row.get("text", "") for row in repo.read() if row.get("ref")}

    @staticmethod
    def _group_local(records) -> dict[str, list]:
        grouped: dict[str, list] = {}
        for record in records:
            grouped.setdefault(record.split, []).append(record)
        return grouped

    def _build_local(self, records, image_col, text_col, image_feature):
        from datasets import Dataset

        paths = [str((self._config.images_dir / record.image_ref).resolve()) for record in records]
        local = Dataset.from_dict({image_col: paths, text_col: [record.text for record in records]})
        local = local.cast_column(image_col, image_feature)
        return local.add_column("source", [self._config.source_local] * len(local))


def _align_columns(dataset, columns: list[str]):
    """Bổ sung cột thiếu (rỗng) rồi sắp xếp đúng thứ tự cột đích."""
    for column in columns:
        if column not in dataset.column_names:
            dataset = dataset.add_column(column, [""] * len(dataset))
    return dataset.select_columns(columns)
