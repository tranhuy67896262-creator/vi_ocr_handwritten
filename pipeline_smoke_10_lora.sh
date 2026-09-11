#!/usr/bin/env bash
# Pipeline smoke cho LoRA full-precision (bf16) — 10 anh repo + toan bo data assets.
# Wrapper: bat USE_4BIT=0 roi goi pipeline_smoke_10.sh (mot implementation duy nhat).
# Vi du:
#   ./pipeline_smoke_10_lora.sh 3b <token>
#   MAX_SEQ_LEN=1024 BATCH_SIZE=2 ./pipeline_smoke_10_lora.sh 3b <token>
set -euo pipefail
cd "$(dirname "$0")"

export USE_4BIT=0

exec bash pipeline_smoke_10.sh "$@"
