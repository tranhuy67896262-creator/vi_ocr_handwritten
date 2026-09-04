import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env.dev")


class Configs:
    """Cấu hình cho project Vi-OCR-Handwritten (Qwen2.5-VL-7B + LoRA/QLoRA)"""

    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
    ADAPTER_DIR = MODELS_DIR / "qwen25vl-7b-vi-hwr-lora"

    # Hugging Face
    HF_TOKEN = os.getenv("HF_TOKEN", "")
    DATASET_NAME = "tranhuy67896262/Viet-Handwriting-OCR-v2-bucket"
    MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
    PUSH_TO_HUB = False
    HUB_ADAPTER_ID = ""  # Truyền qua CLI: --hub-repo <owner>/<repo>

    # Prompt hệ thống cho OCR
    SYSTEM_PROMPT = (
        "Đọc chính xác toàn bộ chữ viết tay trong ảnh. "
        "Chỉ trả về văn bản đã đọc được, không thêm lời giải thích."
    )

    # Dữ liệu
    TRAIN_SPLIT = "train"
    TEST_SPLIT = "test"
    VAL_RATIO = 0.002
    MIN_PIXELS = 256 * 28 * 28
    MAX_PIXELS = 1280 * 28 * 28

    # Training (QLoRA: giữ nguyên kiến thức model gốc)
    NUM_EPOCHS = 1
    BATCH_SIZE = 2
    GRADIENT_ACCUMULATION_STEPS = 8
    LEARNING_RATE = 2e-5
    LR_SCHEDULER = "cosine"
    WARMUP_RATIO = 0.03
    MAX_SEQ_LEN = 1024
    GRADIENT_CHECKPOINTING = True
    LOGGING_STEPS = 50
    SAVE_STEPS = 500
    EVAL_STEPS = 250
    SEED = 42

    # LoRA
    USE_4BIT = True
    LORA_R = 32
    LORA_ALPHA = 64
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    # 4-bit quantization
    BNB_4BIT_QUANT_TYPE = "nf4"
    BNB_4BIT_COMPUTE_DTYPE = "bf16"
    BNB_4BIT_USE_DOUBLE_QUANT = True

    # Inference
    MAX_NEW_TOKENS = 256
    ATTN_IMPLEMENTATION = "sdpa"  # Windows dùng "sdpa"; Linux có thể "flash_attention_2"

    def __init__(self):
        """Tạo thư mục nếu chưa tồn tại"""
        self.DATA_DIR.mkdir(exist_ok=True)
        self.MODELS_DIR.mkdir(exist_ok=True)
        self.NOTEBOOKS_DIR.mkdir(exist_ok=True)
        self.ADAPTER_DIR.mkdir(exist_ok=True)