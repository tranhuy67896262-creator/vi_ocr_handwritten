from pathlib import Path

import torch
from peft import PeftModel
from transformers import Qwen2_5_VLForConditionalGeneration

from src.model.load import load_processor


def load_ocr_model(config, adapter_dir=None):
    """Load model gốc + LoRA adapter để OCR.

    adapter_dir có thể là đường dẫn local hoặc repo id trên HF Hub (vd 'owner/repo').
    """
    processor = load_processor(config)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        attn_implementation=config.ATTN_IMPLEMENTATION,
        token=config.HF_TOKEN or None,
    )
    if adapter_dir:
        is_local = Path(adapter_dir).exists()
        if not is_local and "/" not in str(adapter_dir):
            raise ValueError(
                f"Không thấy adapter '{adapter_dir}'. Truyền đường dẫn local hoặc repo id dạng 'owner/repo'."
            )
        print(f"Load adapter từ: {adapter_dir}")
        model = PeftModel.from_pretrained(model, adapter_dir, token=config.HF_TOKEN or None)
    model.eval()
    return model, processor


def predict_image(config, model, processor, image, system_prompt=None):
    """OCR 1 ảnh PIL -> trả về chuỗi chữ viết tay đọc được."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": system_prompt or config.SYSTEM_PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=config.MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
        )
    output_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()