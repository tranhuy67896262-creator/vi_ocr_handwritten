"""Tra cứu registry lịch sử train (models/runs.json)."""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.configs import Configs
from src.storage.run_store import RunStore


def _print_row(row):
    """In 1 dòng run gọn, đủ để biết nạp adapter nào để tiếp tục train."""
    cer = f"{row['cer']:.4f}" if row["cer"] is not None else "-"
    wer = f"{row['wer']:.4f}" if row["wer"] is not None else "-"
    print(
        f"#{row['id']:<3} {row['run_name'] or '-':<12} {row['created_at']} "
        f"samples={row['train_samples']} steps={row['train_steps']} "
        f"cer={cer} wer={wer}"
    )
    print(f"      model={row['model']}")
    print(f"      adapter={row['adapter_dir']}")
    if row["init_adapter"]:
        print(f"      init_adapter={row['init_adapter']}")
    if row["hub_repo"]:
        rev = row["hub_revision"] or "main"
        print(f"      hub={row['hub_repo']}@{rev}")


def main():
    """Parse CLI và in danh sách run."""
    parser = argparse.ArgumentParser(
        description="Tra cứu lịch sử train trong registry (JSON)"
    )
    parser.add_argument("--limit", type=int, default=20, help="Số run gần nhất (mặc định 20)")
    parser.add_argument("--file", type=str, default=None,
                        help="Đường dẫn file JSON (mặc định models/runs.json)")
    args = parser.parse_args()

    config = Configs()
    file_path = args.file or str(config.RUNS_FILE)
    rows = RunStore(file_path).list(limit=args.limit)
    if not rows:
        print(f"Chưa có run nào trong {file_path}.")
        return
    for row in rows:
        _print_row(row)
        print("-" * 70)


if __name__ == "__main__":
    main()
