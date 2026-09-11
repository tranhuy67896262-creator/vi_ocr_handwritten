"""CLI cho module labeling: chạy editor (serve) hoặc gộp dataset (merge).

Đây là composition root: ráp các implementation cụ thể vào service.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from configs.configs import Configs

from .hf_source import HfConfig, HuggingFaceSource
from .local_source import LocalImageSource
from .merge import DatasetMerger, MergeConfig
from .registry import DatasetRegistry
from .server import DEFAULT_TEMPLATE, serve
from .service import LabelService


def _default_paths() -> dict[str, Path]:
    root = Configs.PROJECT_ROOT
    images = root / "assets"
    return {
        "images_dir": images,
        "csv_path": images / "labels.csv",
        "overrides_path": images / "hf_overrides.csv",
        "hf_cache": root / "data" / "hf-cache",
        "output_dir": root / "data" / "combined",
    }


def _add_common(parser: argparse.ArgumentParser, paths: dict[str, Path]) -> None:
    parser.add_argument("--images-dir", type=Path, default=paths["images_dir"],
                        help="Thu muc anh ca nhan (mac dinh: assets/)")
    parser.add_argument("--csv", type=Path, default=paths["csv_path"],
                        help="File labels.csv cua anh ca nhan")
    parser.add_argument("--overrides", type=Path, default=paths["overrides_path"],
                        help="File chua text override cho nguon Hugging Face")
    parser.add_argument("--hf-dataset", default=Configs.DATASET_NAME,
                        help="Ten dataset Hugging Face lam nen")
    parser.add_argument("--hf-cache", type=Path, default=paths["hf_cache"],
                        help="Thu muc cache dataset Hugging Face")


def build_parser() -> argparse.ArgumentParser:
    """Tạo parser với 2 subcommand: serve (mặc định) và merge."""
    paths = _default_paths()
    parser = argparse.ArgumentParser(prog="python -m scripts.labeling",
                                     description="Cong cu gan nhan OCR: HF + anh ca nhan")
    sub = parser.add_subparsers(dest="command")

    serve_parser = sub.add_parser("serve", help="Chay label editor tren trinh duyet")
    _add_common(serve_parser, paths)
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=9000)
    serve_parser.add_argument("--no-hf", action="store_true", help="Chi gan nhan anh ca nhan")
    serve_parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    serve_parser.add_argument("--token", default="", help="HF token (mac dinh: tu .env.dev)")

    merge_parser = sub.add_parser("merge", help="Gop HF + anh ca nhan -> data/combined")
    _add_common(merge_parser, paths)
    merge_parser.add_argument("--output", type=Path, default=paths["output_dir"])
    merge_parser.add_argument("--token", default="", help="HF token (mac dinh: tu .env.dev)")
    return parser


def _resolve_token(explicit: str) -> str:
    return explicit or Configs().HF_TOKEN


def _run_serve(args: argparse.Namespace) -> None:
    sources: list = [LocalImageSource(args.images_dir, args.csv)]
    if not args.no_hf:
        sources.append(HuggingFaceSource(HfConfig(
            dataset_id=args.hf_dataset,
            token=_resolve_token(args.token),
            cache_dir=args.hf_cache,
            overrides_path=args.overrides,
        )))
    service = LabelService(DatasetRegistry(sources))
    serve(service, host=args.host, port=args.port, template_path=args.template)


def _run_merge(args: argparse.Namespace) -> None:
    merger = DatasetMerger(MergeConfig(
        hf_dataset_id=args.hf_dataset,
        token=_resolve_token(args.token),
        cache_dir=args.hf_cache,
        images_dir=args.images_dir,
        csv_path=args.csv,
        overrides_path=args.overrides,
        output_dir=args.output,
    ))
    counts = merger.merge()
    print(f"Da gop dataset vao: {args.output}")
    for split, total in counts.items():
        print(f"  {split}: {total} rows")


def main(argv: list[str] | None = None) -> None:
    """Entry point CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "merge":
        _run_merge(args)
    elif args.command == "serve":
        _run_serve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
