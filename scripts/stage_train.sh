#!/usr/bin/env bash
# Train 1 moc staged tren may GPU truc tiep (khong qua Modal).
# Tu dung .venv + cai deps (1 lan) roi chay train.py voi cau hinh chuan 7B bf16 LoRA.
#
# Cach dung:
#   export HF_TOKEN=<token owner tranhuy67896262>   # hoac ghi HF_TOKEN=... vao .env.dev
#   ./scripts/stage_train.sh <start> <max> <init_rev> <hub_rev> <run_name>
# VD moc 15k:
#   ./scripts/stage_train.sh 5000 15000 stage-5k stage-15k run-15k
# Xem tien do: tail -f train-run-15k.log
set -euo pipefail

START=${1:?can start-samples (vd 5000)}
MAX=${2:?can max-samples (vd 15000)}
INIT_REV=${3:?can init revision (vd stage-5k)}
HUB_REV=${4:?can hub-revision (vd stage-15k)}
RUN=${5:?can run-name (vd run-15k)}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -f .env.dev ]; then
  set -a
  # shellcheck disable=SC1091
  source .env.dev
  set +a
fi
: "${HF_TOKEN:?chua co HF_TOKEN - export HF_TOKEN hoac ghi vao .env.dev}"

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

if [ ! -f .venv/.setup_done ]; then
  pip install --upgrade pip
  pip install torch --index-url https://download.pytorch.org/whl/cu128
  pip install -r requirements.txt
  touch .venv/.setup_done
fi

LOG="train-${RUN}.log"
nohup python scripts/train.py \
  --dataset tranhuy67896262/Viet-Handwriting-OCR-v2-local \
  --model tranhuy67896262/Qwen2.5-VL-7B-Instruct-private \
  --start-samples "$START" --max-samples "$MAX" --no-4bit --batch-size 4 \
  --gradient-accumulation-steps 4 --lr 2e-5 --epochs 1 \
  --init-adapter "tranhuy67896262/qwen25vl-7b-vi-hwr-lora@${INIT_REV}" \
  --push --push-every-save --hub-repo tranhuy67896262/qwen25vl-7b-vi-hwr-lora \
  --hub-revision "$HUB_REV" --run-name "$RUN" \
  > "$LOG" 2>&1 &
echo "Dang chay ${RUN} (PID $!), log: ${LOG}"
