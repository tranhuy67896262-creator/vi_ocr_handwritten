#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Portable qua máy khác: cache và target có thể nằm khác filesystem
# (vd mount Drive, ổ đĩa khác) -> copy thay vì hardlink, tắt warning
# "Failed to hardlink files; falling back to full copy".
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

echo "=============================================="
echo "  Vi-OCR-Handwritten - Setup & Train (Colab/Linux)"
echo "=============================================="

# Nhận token từ đối số đầu tiên (nếu không bắt đầu bằng --) hoặc biến env HF_TOKEN
TOKEN=""
if [ $# -ge 1 ] && [[ "$1" != --* ]]; then
    TOKEN="$1"
    shift
fi

if [ -n "$TOKEN" ]; then
    printf 'HF_TOKEN = %s\n' "$TOKEN" > .env.dev
    echo "Da ghi HF_TOKEN vao .env.dev"
elif [ -z "${HF_TOKEN:-}" ] && ! grep -q "^HF_TOKEN" .env.dev 2>/dev/null; then
    echo "[WARN] Chua co HF_TOKEN. Truyen token lam doi so dau:"
    echo "  $0 hf_xxxxxxxx --max-samples 100"
    echo "  (Lay token tai: https://huggingface.co/settings/tokens)"
fi

# Cờ riêng của run_train.sh (không truyền xuống train_qlora.py):
#   --ui          : chỉ cài môi trường + mở UI Gradio (không train)
#   --eval[=NNN]  : sau khi train, chạy eval CER/WER trên NNN mẫu (mặc định 100)
DO_UI=""
DO_EVAL=""
EVAL_NUM="100"
NEW_ARGS=()
HAS_ARGS=""
for arg in "$@"; do
    case "$arg" in
        --ui)     DO_UI=1 ;;
        --eval)   DO_EVAL=1 ;;
        --eval=*) DO_EVAL=1; EVAL_NUM="${arg#--eval=}" ;;
        *)        NEW_ARGS+=("$arg"); HAS_ARGS=1 ;;
    esac
done
# Chi nhan so cho --eval (train_qlora.py khong co flag --eval* nao khac)
case "$EVAL_NUM" in
    ''|*[!0-9]*) EVAL_NUM="100" ;;
esac

# Cache model/dataset vào thư mục relative của project (vd .hf_cache) — dễ mang theo.
# Muốn cache chỗ khác/Drive: export HF_HOME=<path> trước khi chạy.
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
echo "HF cache: $HF_HOME"

# Cài uv nếu chưa có (thay pip/venv — nhanh hơn nhiều)
if ! command -v uv >/dev/null 2>&1; then
    echo "Dang cai uv..."
    python -m pip install -q uv 2>/dev/null || python3 -m pip install -q uv 2>/dev/null || pip install -q uv
fi

# Chọn python: ưu tiên venv .venv (tạo bằng uv), không thì python hệ thống (đường dẫn tuyệt đối
# để uv không cài nhầm vào venv đang active của shell)
if [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif [ -x ".venv/Scripts/python.exe" ]; then
    PYTHON=".venv/Scripts/python.exe"
else
    PYTHON="$(command -v python || command -v python3 || echo python)"
fi
unset VIRTUAL_ENV
echo "Python: $("$PYTHON" --version 2>/dev/null || echo "khong xac dinh")"
if ! "$PYTHON" -c "import sys; assert sys.version_info >= (3, 10)" 2>/dev/null; then
    echo "[WARN] Can Python >= 3.10 - da test ky tren 3.13.x."
fi

# Chế độ --ui: lên UI ngay, KHÔNG cài torch-CUDA / requirements / data.
# Chỉ cần gradio + dotenv để UI mở nhanh. Nút Train/OCR trong UI sẽ báo lỗi
# nếu thiếu deps — khi đó chạy lại script không --ui để cài full.
if [ -n "$DO_UI" ]; then
    echo "===== Mo UI Gradio (che do nhe: chua tai torch/data) ====="
    if ! "$PYTHON" -c "import gradio, dotenv" 2>/dev/null; then
        echo "Dang cai gradio (toi thieu cho UI)..."
        uv pip install --python "$PYTHON" gradio python-dotenv
    fi
    "$PYTHON" scripts/ui.py
    exit 0
fi

# Kiểm tra GPU (chỉ ở chế độ train — giong run_train.bat)
if ! "$PYTHON" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "[WARN] Khong thay GPU. Tren Colab: Runtime > Change runtime type > GPU (T4/A100/H100)."
fi

# Cài torch/torchvision bản CUDA nếu chưa có (PyPI mặc định là bản CPU)
if ! "$PYTHON" -c "import torch, torchvision; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "Dang cai torch + torchvision ban CUDA..."
    uv pip install --python "$PYTHON" torch torchvision --index-url https://download.pytorch.org/whl/cu128
fi

# Cài dependencies còn thiếu (Colab đã có sẵn torch CUDA + torchvision)
if ! "$PYTHON" -c "import transformers, peft, bitsandbytes, torchvision" 2>/dev/null; then
    echo "Dang cai requirements..."
    uv pip install --python "$PYTHON" -r requirements.txt
fi

echo "GPU: $("$PYTHON" -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo 'CPU')"

echo "Bat dau train..."
echo "  Smoke test: $0 --max-samples 100"
echo "  Train full: $0"
echo "  Push Hub  : $0 --push --hub-repo <owner>/qwen25vl-7b-vi-hwr-lora"
echo

# HAS_ARGS thay cho [ ${#NEW_ARGS[@]} -gt 0 ]: mảng rỗng + set -u crash
# trên bash cũ (< 4.4, vd macOS) với lỗi "unbound variable".
if [ -n "$HAS_ARGS" ]; then
    "$PYTHON" scripts/train_qlora.py "${NEW_ARGS[@]}"
else
    "$PYTHON" scripts/train_qlora.py
fi

# Sao lưu adapter + log (tuỳ chọn) — vd: export BACKUP_DIR=/content/drive/MyDrive/vi_ocr_handwritten_models
if [ -n "${BACKUP_DIR:-}" ] && [ -d models ]; then
    mkdir -p "$BACKUP_DIR"
    cp -r models/. "$BACKUP_DIR/" 2>/dev/null
    echo "Da sao luu ket qua vao: $BACKUP_DIR"
fi

# Chạy eval CER/WER nếu có --eval
if [ -n "$DO_EVAL" ]; then
    echo "===== Eval CER/WER tren $EVAL_NUM mau test ====="
    "$PYTHON" scripts/eval_ocr.py --num-test "$EVAL_NUM"
fi

echo
echo "Xong. Ket qua o: models/qwen25vl-7b-vi-hwr-lora/ (xem training_metadata.json + training.log)"