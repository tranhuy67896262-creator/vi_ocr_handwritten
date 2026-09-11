"""Load Qwen2.5-VL + processor, gắn LoRA.

Chọn mức precision theo *strategy* (OCP): thêm mode mới (vd 8-bit) chỉ cần tạo
một ``PrecisionLoader`` rồi đăng ký, không sửa luồng load chính.
"""
from abc import ABC, abstractmethod

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor


def load_processor(config, model_name=None):
    """Load processor + tokenizer Qwen2.5-VL."""
    processor = Qwen2_5_VLProcessor.from_pretrained(
        model_name or config.MODEL_NAME, token=config.HF_TOKEN or None
    )
    processor.tokenizer.padding_side = "right"  # pylint: disable=no-member
    return processor


def resolve_attn_implementation(config):
    """Chọn attention: 'auto' -> flash_attention_2 nếu có (Linux/GPU lớn),
    ngược lại sdpa (Windows hoặc thiếu flash_attn)."""
    if config.ATTN_IMPLEMENTATION != "auto":
        return config.ATTN_IMPLEMENTATION
    try:
        import flash_attn  # noqa: F401  # pylint: disable=unused-import
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


class PrecisionLoader(ABC):
    """Cách nạp base model theo một mức precision (strategy)."""

    # pylint: disable=unused-argument  # hook có tham số đồng nhất cho mọi loader
    uses_4bit = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Tên hiển thị của chế độ precision."""

    @abstractmethod
    def is_available(self) -> bool:
        """Môi trường có đáp ứng được chế độ này không (không nạp model)."""

    def quantization_config(self, config, compute_dtype):
        """Cấu hình lượng tử hóa (None nếu không quantize)."""
        return None

    def post_load(self, model, config):
        """Tinh chỉnh model sau khi nạp (mặc định giữ nguyên)."""
        return model


class QloraLoader(PrecisionLoader):
    """4-bit NF4 QLoRA qua bitsandbytes (cần CUDA + bitsandbytes)."""

    uses_4bit = True

    @property
    def name(self) -> str:
        """Tên hiển thị."""
        return "4-bit QLoRA"

    def is_available(self) -> bool:
        """Cần CUDA và bitsandbytes cài được."""
        if not torch.cuda.is_available():
            return False
        try:
            import bitsandbytes  # noqa: F401  # pylint: disable=unused-import
            return True
        except ImportError:
            return False

    def quantization_config(self, config, compute_dtype):
        """Cấu hình BitsAndBytes 4-bit theo config."""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=config.BNB_4BIT_USE_DOUBLE_QUANT,
        )

    def post_load(self, model, config):
        """Chuẩn bị model 4-bit cho train (kbit + gradient checkpointing)."""
        return prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.GRADIENT_CHECKPOINTING
        )


class Bf16Loader(PrecisionLoader):
    """LoRA full-precision bf16 (không quantize)."""

    uses_4bit = False

    @property
    def name(self) -> str:
        """Tên hiển thị."""
        return "bf16 LoRA (full-precision)"

    def is_available(self) -> bool:
        """Luôn khả dụng (chỉ cần GPU đủ VRAM)."""
        return True


def resolve_loader(config) -> PrecisionLoader:
    """Chọn loader theo config; fallback về bf16 nếu QLoRA không khả dụng."""
    if config.USE_4BIT:
        qlora = QloraLoader()
        if qlora.is_available():
            return qlora
        print("bitsandbytes/CUDA không có sẵn -> chuyển sang LoRA full-precision (bf16).")
    return Bf16Loader()


def load_model_and_processor(config):
    """Load Qwen2.5-VL theo precision đã chọn và trả ``(model, processor, use_4bit)``."""
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    loader = resolve_loader(config)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.MODEL_NAME,
        quantization_config=loader.quantization_config(config, compute_dtype),
        torch_dtype=compute_dtype,
        device_map="auto",
        attn_implementation=resolve_attn_implementation(config),
        token=config.HF_TOKEN or None,
    )
    model = loader.post_load(model, config)
    processor = load_processor(config)

    print(f"Precision: {loader.name} | dtype: {compute_dtype}")
    return model, processor, loader.uses_4bit


def build_lora_model(config, model):
    """Gắn LoRA adapter lên model. Chỉ adapter được train, base model đóng băng."""
    peft_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    if config.GRADIENT_CHECKPOINTING:
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model
