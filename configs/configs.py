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


def adapter_dir(models_dir, model_name, use_4bit=True):
    """Đường dẫn adapter LoRA, tách theo precision để QLoRA và bf16 không ghi đè nhau."""
    precision = "lora" if use_4bit else "lora-bf16"
    return Path(models_dir) / f"qwen25vl-{_adapter_tag(model_name)}-vi-hwr-{precision}"


def default_adapter_dir(models_dir, model_name):
    """Thư mục adapter mặc định cho eval/export/UI: ưu tiên bản đã có trọng số.

    Train staged mặc định bf16 (``...-lora-bf16``) nhưng flow QLoRA cũ dùng
    ``...-lora`` — trỏ nhầm là test sai adapter (vd 7B đọc kém oan).
    Tiêu chí "đã có": chứa ``adapter_config.json``.
    """
    bf16 = adapter_dir(models_dir, model_name, False)
    if (bf16 / "adapter_config.json").is_file():
        return bf16
    return adapter_dir(models_dir, model_name, True)


def default_hub_repo(model_name):
    """Repo adapter Hub mặc định theo model (khớp default_hub_for_model trong common.sh).

    UI hiện thẳng repo id này khi máy chưa có weight local — user không phải gõ tay.
    """
    name = model_name.rsplit("/", 1)[-1].lower()
    size = "3b" if "3b" in name else "7b"
    return f"tranhuy67896262/qwen25vl-{size}-vi-hwr-lora"


class Configs:
    """Cấu hình cho project Vi-OCR-Handwritten (Qwen2.5-VL + LoRA/QLoRA)"""

    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
    # Registry JSON lưu lịch sử các lần train/eval (tiện tra cứu; nguồn chân lý
    # vẫn là adapter trên HF Hub). File nằm trong models/ -> đã git-ignore.
    RUNS_FILE = MODELS_DIR / "runs.json"

    # Hugging Face
    HF_TOKEN = os.getenv("HF_TOKEN", "")
    DATASET_NAME = "tranhuy67896262/Viet-Handwriting-OCR-v2-local"
    # CHỈ cần sửa MODEL_NAME khi muốn đổi model (3B/7B/...) — ADAPTER_DIR tự suy ra sau.
    MODEL_NAME = "tranhuy67896262/Qwen2.5-VL-3B-Instruct-private"
    PUSH_TO_HUB = False
    HUB_ADAPTER_ID = ""  # Truyền qua CLI: --hub-repo <owner>/<repo>

    # Thư mục adapter tự suy từ MODEL_NAME + precision (3B QLoRA -> qwen25vl-3b-vi-hwr-lora;
    # bf16 -> qwen25vl-3b-vi-hwr-lora-bf16). Đặt SAU MODEL_NAME vì class body chạy tuần tự.
    ADAPTER_DIR = adapter_dir(MODELS_DIR, MODEL_NAME, True)

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
    # Ảnh crop dòng: MAX_PIXELS chặn image-token ở mức 768 (768 tile 28x28).
    # MAX_SEQ_LEN là budget TEXT; collator tự cộng thêm budget image-token
    # (MAX_PIXELS/784) vào max_length -> tổng chuỗi = text + image, không còn
    # truncation cắt image-token gây crash "Mismatch in image token count".
    MAX_PIXELS = 768 * 28 * 28

    # Training (QLoRA: giữ nguyên kiến thức model gốc)
    NUM_EPOCHS = 1
    BATCH_SIZE = 2
    GRADIENT_ACCUMULATION_STEPS = 8
    LEARNING_RATE = 2e-5
    LR_SCHEDULER = "cosine"
    WARMUP_RATIO = 0.03
    # Budget TEXT (token). Image-token (≤768) được cộng thêm trong collator.
    MAX_SEQ_LEN = 1024
    GRADIENT_CHECKPOINTING = True
    # Colab/Linux: nạp ảnh và chạy processor song song, giảm thời gian GPU chờ batch.
    DATALOADER_NUM_WORKERS = 4
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
        # Adapter mặc định ưu tiên bản đã có trọng số (bf16 staged trước,
        # QLoRA sau) — eval/export không truyền --adapter vẫn trúng.
        self.ADAPTER_DIR = default_adapter_dir(self.MODELS_DIR, self.MODEL_NAME)
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
