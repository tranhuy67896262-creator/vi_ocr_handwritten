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
    MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
    PUSH_TO_HUB = False
    HUB_ADAPTER_ID = ""  # Truyền qua CLI: --hub-repo <owner>/<repo>

    # Thư mục adapter tự suy từ MODEL_NAME + precision (7B QLoRA -> qwen25vl-7b-vi-hwr-lora;
    # bf16 -> qwen25vl-7b-vi-hwr-lora-bf16). Đặt SAU MODEL_NAME vì class body chạy tuần tự.
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
    # 1280 tile 28x28 ~= 1280 image-token. Dataset 5CD-AI là ảnh crop DÒNG nên độ
    # phân giải là nút cổ chai của dấu thanh — 768 (bản cũ) làm mất dấu. Đổi lại
    # phải nâng MAX_SEQ_LEN lên 1536 để chứa đủ image-token + text.
    MAX_PIXELS = 1280 * 28 * 28
    # Shuffle cố định trước khi cắt N mẫu: 5 mốc (5k/10k/20k/40k/59k) vẫn LỒNG NHAU
    # (cùng permutation) nhưng không còn lấy N dòng đầu theo thứ tự parquet.
    DATA_SHUFFLE_SEED = 42
    # Eval cố định lấy từ TEST_SPLIT -> mọi mốc đo cùng một thước (so CER/WER được).
    EVAL_SAMPLES = 300
    EVAL_SHUFFLE_SEED = 42

    # Training (QLoRA: giữ nguyên kiến thức model gốc)
    NUM_EPOCHS = 1
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4  # effective batch = 16 (giữ ~8.4k step cho 5 mốc)
    LEARNING_RATE = 1e-4
    LR_SCHEDULER = "cosine"
    WARMUP_RATIO = 0.03
    MAX_SEQ_LEN = 1536
    GRADIENT_CHECKPOINTING = True
    # Colab/Linux: nạp ảnh và chạy processor song song, giảm thời gian GPU chờ batch.
    DATALOADER_NUM_WORKERS = 4
    LOGGING_STEPS = 50
    SAVE_STEPS = 500
    EVAL_STEPS = 250
    SEED = 42

    # Chống mất kiến thức gốc: cộng KL-divergence(model gốc || model LoRA) vào loss.
    # Model gốc = forward cùng batch với LoRA tạm tắt (disable_adapter), không tốn thêm model copy.
    # MẶC ĐỊNH TẮT cho OCR chuyên dụng: KL kéo phân phối đáp án về base model — chính
    # cái đang đọc sai dấu — nên nó chống lại việc cần học. Bật lại bằng cờ `--kl`.
    KL_REGULARIZATION = False
    KL_COEFFICIENT = 0.5

    # LoRA
    USE_4BIT = True
    LORA_R = 64
    LORA_ALPHA = 128
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
    # dtype khi OCR/export: fp16 (mặc định — hợp T4 và mọi GPU) | bf16 | auto (bf16 nếu hỗ trợ).
    INFER_DTYPE = "fp16"
    # Hậu xử lý tiếng Việt (chữa dấu/lỗi ký tự OCR) — model HF seq2seq, bật qua cờ CLI/UI.
    SPELLFIX_MODEL = "nrl-ai/vn-spell-correction-small"
    PDF_DPI = 200  # DPI render trang PDF scan (khớp file scan thực tế ~200dpi)
    # Số lát cắt mỗi trang scan (PDF/DOCX): chẻ TRƯỚC khi thu nhỏ để mỗi lát
    # giữ chi tiết chữ nhỏ. Chỉ áp dụng khi ảnh lớn (cạnh dài > 1200px).
    PAGE_TILES = 2
    ATTN_IMPLEMENTATION = "auto"  # auto: flash_attention_2 nếu có (Linux/GPU lớn), nếu không sdpa

    # Staged training (cumulative): 5 mốc chia 59,247 dòng train, chạy bằng
    # `bash scripts/train_stages.sh`. Mỗi mốc train tiếp từ adapter mốc trước
    # (--init-adapter), eval cùng tập cố định rồi push 1 revision riêng lên Hub.
    STAGE_SAMPLES = (5000, 10000, 20000, 40000, 59247)

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
