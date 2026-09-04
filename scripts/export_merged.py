import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from peft import PeftModel
from transformers import Qwen2_5_VLForConditionalGeneration

from configs.configs import Configs
from src.model.load import load_processor
from src.utils.logging import log_and_exit, setup_file_logging


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter vào base model và export full model (chạy độc lập được)"
    )
    parser.add_argument("--adapter", type=str, default=None,
                        help="Thư mục adapter hoặc repo id (mặc định: từ config)")
    parser.add_argument("--output", type=str, default=None,
                        help="Thư mục xuất full model (mặc định: models/<adapter>-merged)")
    args = parser.parse_args()

    config = Configs()
    setup_file_logging(config.MODELS_DIR / "export.log")
    adapter = args.adapter or str(config.ADAPTER_DIR)
    out = Path(args.output) if args.output else config.MODELS_DIR / f"{config.ADAPTER_DIR.name}-merged"

    print(f"Base : {config.MODEL_NAME}")
    print(f"LoRA : {adapter}")
    print(f"Export: {out}")

    try:
        processor = load_processor(config)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.MODEL_NAME,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",
            attn_implementation=config.ATTN_IMPLEMENTATION,
            token=config.HF_TOKEN or None,
        )
        model = PeftModel.from_pretrained(model, adapter, token=config.HF_TOKEN or None)

        print("Dang merge LoRA vao base model...")
        merged = model.merge_and_unload()

        out.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(out)
        processor.save_pretrained(out)
        print(f"Da export full model vao: {out}")
    except Exception as exc:
        log_and_exit(exc, stage="EXPORT",
                     extra_hint="VRAM thap: merged model can ~2x VRAM cua model (bf16). Export tren GPU >=16GB (3B) hoac 32GB (7B).")


if __name__ == "__main__":
    main()