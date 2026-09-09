"""Cấu hình tập trung cho project Vi-OCR-Handwritten."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env.dev")


def _adapter_tag(model_name):
    """Rút kích thước model từ tên (vd 'Qwen2.5-VL-3B-Instruct' -> '3b')."""
    name = model_name.rsplit("/", 1)[-1].lower()
    for size in ("3b", "7b", "14b", "32b", "72b"):
        if size in name:
            return size
    return "model"


class Configs:
    """Cấu hình cho project Vi-OCR-Handwritten (Qwen2.5-VL + LoRA/QLoRA)"""

    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

    # Hugging Face
    HF_TOKEN = os.getenv("HF_TOKEN", "")
    DATASET_NAME = "5CD-AI/Viet-Handwriting-OCR-v2"
    # CHỈ cần sửa MODEL_NAME khi muốn đổi model (3B/7B/...) — ADAPTER_DIR tự suy ra sau.
    MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
    PUSH_TO_HUB = False
    HUB_ADAPTER_ID = ""  # Truyền qua CLI: --hub-repo <owner>/<repo>

    # Thư mục adapter tự suy từ MODEL_NAME (vd 3B -> qwen25vl-3b-vi-hwr-lora)
    # Đặt SAU MODEL_NAME vì class body chạy tuần tự.
    ADAPTER_DIR = MODELS_DIR / f"qwen25vl-{_adapter_tag(MODEL_NAME)}-vi-hwr-lora"

    # Prompt hệ thống cho OCR
    SYSTEM_PROMPT = (
        "Đọc chính xác toàn bộ chữ viết tay trong ảnh. "
        "Chỉ trả về văn bản đã đọc được, không thêm lời giải thích."
    )

    # Dữ liệu
    TRAIN_SPLIT = "train"
    TEST_SPLIT = "test"
    VAL_RATIO = 0.002
    # Chuẩn hóa mọi ảnh về tỉ lệ khổ A4 (pad nền trắng, không crop) trước khi
    # vào processor — áp dụng cả train (dataset.py) lẫn inference (predict.py).
    A4_STANDARDIZE = True
    MIN_PIXELS = 256 * 28 * 28
    # 768 tile 28x28 ~= 768 image-token: vừa khung MAX_SEQ_LEN=1024 (cả text).
    # Để 1280 như trước → image-token đã ~1456, truncation cắt vào vùng ảnh
    # gây crash "Mismatch in image token count" + train chậm (50s+/it).
    MAX_PIXELS = 768 * 28 * 28

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

    # Chống mất kiến thức gốc: cộng KL-divergence(model gốc || model LoRA) vào loss.
    # Model gốc = forward cùng batch với LoRA tạm tắt (disable_adapter), không tốn thêm model copy.
    # Tắt (False) nếu VRAM hẹp — tốn thêm 1 forward no_grad mỗi step.
    KL_REGULARIZATION = True
    KL_COEFFICIENT = 0.5

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
    PDF_DPI = 200  # DPI render trang PDF scan (khớp file scan thực tế ~200dpi)
    # Số lát cắt mỗi trang A4 scan (PDF/DOCX): chẻ TRƯỚC khi thu nhỏ để mỗi lát
    # giữ chi tiết chữ nhỏ. Chỉ áp dụng khi ảnh lớn (cạnh dài > 1200px).
    PAGE_TILES = 2
    ATTN_IMPLEMENTATION = "auto"  # auto: flash_attention_2 nếu có (Linux/GPU lớn), nếu không sdpa

    def __init__(self):
        """Tạo thư mục nếu chưa tồn tại"""
        self.DATA_DIR.mkdir(exist_ok=True)
        self.MODELS_DIR.mkdir(exist_ok=True)
        self.NOTEBOOKS_DIR.mkdir(exist_ok=True)
        self.ADAPTER_DIR.mkdir(exist_ok=True)
        # Đọc token fresh mỗi lần khởi tạo (ưu tiên env, fallback .env.dev):
        # class attribute HF_TOKEN chỉ tính 1 lần lúc import — token lưu giữa
        # session qua tab Settings sẽ không thấy nếu không làm bước này.
        self.HF_TOKEN = os.getenv("HF_TOKEN", "") or self._read_dotenv_token()

    @staticmethod
    def _read_dotenv_token():
        try:
            for line in (Configs.PROJECT_ROOT / ".env.dev").read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("HF_TOKEN"):
                    _, _, v = line.partition("=")
                    return v.strip()
        except OSError:
            pass
        return ""
