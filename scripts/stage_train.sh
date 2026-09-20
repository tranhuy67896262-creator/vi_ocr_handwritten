#!/usr/bin/env bash
# Train 1 moc staged tren may GPU truc tiep (khong qua Modal).
# Dung uv nhu run_train.sh: tu dung .venv + cai deps (1 lan) roi chay
# train.py voi cau hinh chuan 7B bf16 LoRA, chay nen bang nohup.
#
# Cach dung:
#   export HF_TOKEN=<token owner tranhuy67896262>   # hoac ghi HF_TOKEN=... vao .env.dev
#   ./scripts/stage_train.sh <start> <count> <init_rev> <hub_rev> <run_name> [save_steps]
# CHU Y: <count> = SO LUONG mau (khong phai end-offset): end = start + count
# (dataset.py: end = start + max_samples). VD moc 5k->15k: start=5000 count=10000.
# VD:
#   ./scripts/stage_train.sh 5000 10000 stage-5k stage-15k run-15k
#   ./scripts/stage_train.sh 5000 10000 stage-5k stage-15k run-15k 50  # push moi 50 step
# Xem tien do: tail -f train-run-15k.log
set -euo pipefail

START=${1:?can start-samples = offset bo qua (vd 5000)}
MAX=${2:?can max-samples = SO LUONG mau (vd 10000 cho lat 5k->15k)}
INIT_REV=${3:?can init revision (vd stage-5k)}
HUB_REV=${4:?can hub-revision (vd stage-15k)}
RUN=${5:?can run-name (vd run-15k)}
SAVE_STEPS=${6:-100}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"

if [ -f .env.dev ]; then
  set -a
  # shellcheck disable=SC1091
  source .env.dev
  set +a
fi
: "${HF_TOKEN:?chua co HF_TOKEN - export HF_TOKEN hoac ghi vao .env.dev}"

# Cai uv neu chua co (thay pip/venv - nhanh hon nhieu)
if ! command -v uv >/dev/null 2>&1; then
  echo "Dang cai uv..."
  python3 -m pip install -q uv
fi

# Chon python: uu tien .venv (tao bang uv), khong thi python he thong
if [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
else
  PYTHON="$(command -v python3 || command -v python || echo python3)"
  echo "Tao .venv bang uv..."
  uv venv .venv
  PYTHON=".venv/bin/python"
fi
echo "Python: $("$PYTHON" --version 2>/dev/null || echo "khong xac dinh")"

# torch/torchvision ban CUDA (PyPI mac dinh la ban CPU)
if ! "$PYTHON" -c "import torch, torchvision; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "Dang cai torch + torchvision ban CUDA..."
  uv pip install --python "$PYTHON" torch torchvision --index-url https://download.pytorch.org/whl/cu128
fi

# flash-attn best-effort (tang toc A100/H100, thieu thi fallback sdpa)
if ! "$PYTHON" -c "import flash_attn" 2>/dev/null; then
  uv pip install --python "$PYTHON" flash-attn 2>/dev/null || echo "[WARN] Bo qua flash-attn."
fi

# requirements (transformers, peft, ...)
if ! "$PYTHON" -c "import transformers, peft, torchvision" 2>/dev/null; then
  echo "Dang cai requirements..."
  uv pip install --python "$PYTHON" -r requirements.txt
fi

echo "GPU: $("$PYTHON" -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo 'CPU')"

LOG="train-${RUN}.log"
# Tu resume: neu run nay da co checkpoint local (chet giua chung) thi tiep tuc
# dung step cu thay vi hoc lai tu dau.
# INIT_REV=none: train.py hiểu --init-adapter none là train trắng (vd chuỗi -v2),
# shell cứ truyền flag python bình thường.
# Gom co thanh chuoi (thay mang) de tuong thich bash cu voi set -u; gia tri
# khong chua khoang trang nen word-splitting o day an toan.
# shellcheck disable=SC2086
EXTRA_FLAGS="--auto-progress"
if ls -d "models/checkpoints/${RUN}"/checkpoint-* >/dev/null 2>&1; then
  echo "Thay checkpoint local -> resume dung step cu (--resume)."
  EXTRA_FLAGS="$EXTRA_FLAGS --resume"
fi
INIT_ADAPTER="none"
# Repo đích: mặc định repo cũ; train sang repo mới thì export HUB_REPO=owner/repo-moi.
HUB_REPO="${HUB_REPO:-tranhuy67896262/qwen25vl-7b-vi-hwr-lora}"
if [ "$INIT_REV" != "none" ]; then
  INIT_ADAPTER="${HUB_REPO}@${INIT_REV}"
else
  echo "INIT_REV=none -> train adapter moi tu dau (--init-adapter none)."
fi
nohup "$PYTHON" scripts/train.py \
  --dataset tranhuy67896262/Viet-Handwriting-OCR-v2-local \
  --model tranhuy67896262/Qwen2.5-VL-7B-Instruct-private \
  --start-samples "$START" --max-samples "$MAX" --no-4bit --batch-size 4 \
  --gradient-accumulation-steps 4 --lr 2e-5 --epochs 1 \
  --init-adapter "$INIT_ADAPTER" \
  --push --push-every-save --save-steps "$SAVE_STEPS" \
  --hub-repo "$HUB_REPO" \
  --hub-revision "$HUB_REV" --run-name "$RUN" $EXTRA_FLAGS \
  > "$LOG" 2>&1 &
echo "Dang chay ${RUN} (PID $!), log: ${LOG}"
