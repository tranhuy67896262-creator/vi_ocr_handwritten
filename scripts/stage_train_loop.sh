#!/usr/bin/env bash
# Train nhieu moc lien tiep, moi moc 5000 mau moi (sequential).
# Chay FOREGROUND tung moc (moc sau doi moc truoc) - bat buoc chay trong tmux
# de rot SSH khong chet ca chuoi.
#
# Cach dung:
#   export HF_TOKEN=<token owner tranhuy67896262>   # hoac ghi HF_TOKEN=... vao .env.dev
#   ./scripts/stage_train_loop.sh <start> <end> <init_rev> [step] [save_steps]
# VD: chay 15k -> het data, moi moc 5k:
#   ./scripts/stage_train_loop.sh 15000 59247 stage-15k
# VD: buoc 5000, push moi 50 step:
#   ./scripts/stage_train_loop.sh 15000 59247 stage-15k 5000 50
set -euo pipefail

START=${1:?can start (vd 15000)}
END=${2:?can end (vd 59247)}
INIT_REV=${3:?can init revision (vd stage-15k)}
STEP=${4:-5000}
SAVE_STEPS=${5:-100}

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

if [ -n "${TMUX:-}" ]; then
  echo "Dang trong tmux - tot (rot SSH khong chet chuoi train)."
else
  echo "[WARN] Khong thay tmux - nen chay trong tmux (tmux new -s train)."
fi

# Cai uv neu chua co (thay pip/venv - nhanh hon nhieu)
if ! command -v uv >/dev/null 2>&1; then
  echo "Dang cai uv..."
  python3 -m pip install -q uv
fi

# Chon python: uu tien .venv (tao bang uv), khong thi tao moi
if [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
else
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

DATASET="tranhuy67896262/Viet-Handwriting-OCR-v2-local"
MODEL="tranhuy67896262/Qwen2.5-VL-7B-Instruct-private"
HUB_REPO="tranhuy67896262/qwen25vl-7b-vi-hwr-lora"

S="$START"
PREV_REV="$INIT_REV"
while [ "$S" -lt "$END" ]; do
  E=$((S + STEP))
  if [ "$E" -gt "$END" ]; then
    E="$END"
  fi
  # CHU Y: --max-samples la SO LUONG (count), khong phai end-offset
  # (dataset.py: end = start + max). Nen truyen count = E - S.
  COUNT=$((E - S))
  K=$((E / 1000))
  HUB_REV="stage-${K}k"
  RUN="run-${K}k"
  echo "===== Moc ${S} -> ${E} (init ${PREV_REV} -> ${HUB_REV}) ====="

  # Tu resume neu moc nay da co checkpoint local (chet giua chung).
  # Dung chuoi co thay mang (tranh unbound variable tren bash cu voi set -u).
  DO_RESUME=""
  if ls -d "models/checkpoints/${RUN}"/checkpoint-* >/dev/null 2>&1; then
    echo "Thay checkpoint local -> resume dung step cu (--resume)."
    DO_RESUME=1
  fi
  if [ -n "$DO_RESUME" ]; then
    "$PYTHON" scripts/train.py \
      --dataset "$DATASET" --model "$MODEL" \
      --start-samples "$S" --max-samples "$COUNT" --no-4bit --batch-size 4 \
      --gradient-accumulation-steps 4 --lr 2e-5 --epochs 1 \
      --init-adapter "${HUB_REPO}@${PREV_REV}" \
      --push --push-every-save --save-steps "$SAVE_STEPS" \
      --hub-repo "$HUB_REPO" --hub-revision "$HUB_REV" --run-name "$RUN" --resume
  else
    "$PYTHON" scripts/train.py \
      --dataset "$DATASET" --model "$MODEL" \
      --start-samples "$S" --max-samples "$COUNT" --no-4bit --batch-size 4 \
      --gradient-accumulation-steps 4 --lr 2e-5 --epochs 1 \
      --init-adapter "${HUB_REPO}@${PREV_REV}" \
      --push --push-every-save --save-steps "$SAVE_STEPS" \
      --hub-repo "$HUB_REPO" --hub-revision "$HUB_REV" --run-name "$RUN"
  fi

  echo "===== Xong moc ${HUB_REV} - nho eval truoc khi moc sau chay tiep ====="
  PREV_REV="$HUB_REV"
  S="$E"
done

echo "Xong chuoi ${START} -> ${END}."
