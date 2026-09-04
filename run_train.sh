#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

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

# Cài uv nếu chưa có (thay pip/venv — nhanh hơn nhiều)
if ! command -v uv >/dev/null 2>&1; then
    echo "Dang cai uv..."
    python -m pip install -q uv
fi

# Chọn python: ưu tiên venv .venv (tạo bằng uv), không thì dùng python hệ thống
if [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif [ -x ".venv/Scripts/python.exe" ]; then
    PYTHON=".venv/Scripts/python.exe"
else
    PYTHON="python"
fi
echo "Python: $("$PYTHON" --version 2>/dev/null || echo "khong xac dinh")"

# Kiểm tra GPU
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

"$PYTHON" scripts/train_qlora.py "$@"

echo
echo "Xong. Ket qua o: models/qwen25vl-7b-vi-hwr-lora/ (xem training_metadata.json + training.log)"