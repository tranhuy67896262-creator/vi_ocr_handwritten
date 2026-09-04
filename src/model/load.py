import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor


def load_processor(config):
    processor = Qwen2_5_VLProcessor.from_pretrained(config.MODEL_NAME, token=config.HF_TOKEN or None)
    processor.tokenizer.padding_side = "right"
    return processor


def _get_quant_config(config, compute_dtype):
    return BitsAndBytesConfig(
        load_in_4bit=config.USE_4BIT,
        bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config.BNB_4BIT_USE_DOUBLE_QUANT,
    )


def load_model_and_processor(config):
    """Load Qwen2.5-VL-7B với QLoRA 4-bit (base model đóng băng)."""
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_4bit = config.USE_4BIT and torch.cuda.is_available()

    if use_4bit:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            print("bitsandbytes không có sẵn -> chuyển sang LoRA full-precision.")
            use_4bit = False

    quantization_config = _get_quant_config(config, compute_dtype) if use_4bit else None

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.MODEL_NAME,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        device_map="auto",
        attn_implementation=config.ATTN_IMPLEMENTATION,
        token=config.HF_TOKEN or None,
    )
    processor = load_processor(config)

    if use_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.GRADIENT_CHECKPOINTING
        )

    print(f"4-bit QLoRA: {'bật' if use_4bit else 'tắt'} | dtype: {compute_dtype}")
    return model, processor


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