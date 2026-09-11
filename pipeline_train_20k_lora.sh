#!/usr/bin/env bash
# Pipeline train 20k cho LoRA full-precision (bf16) — repo + toan bo data assets.
# Wrapper: bat USE_4BIT=0 roi goi pipeline_train_20k.sh (mot implementation duy nhat).
# Chi chay sau khi smoke test thanh cong.
# Vi du:
#   ./pipeline_train_20k_lora.sh 7b <token>
#   MAX_SAMPLES=5000 MAX_SEQ_LEN=1024 ./pipeline_train_20k_lora.sh 7b <token>
set -euo pipefail
cd "$(dirname "$0")"

export USE_4BIT=0

exec bash pipeline_train_20k.sh "$@"
