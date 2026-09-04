#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Export GGUF từ merged HF model.
# Bước 1: python scripts/export_merged.py        (cần trước)
# Bước 2: ./scripts/export_gguf.sh                (convert + quantize)
#
# Biến môi trường tuỳ chọn:
#   LLAMA_CPP_DIR  thư mục llama.cpp (mặc định /content/llama.cpp)
#   QUANT          loại quantize (mặc định Q4_K_M; để "none" bỏ qua)
#   LLAMA_CUBLAS   "ON" để build bản GPU (mặc định CPU — đủ cho convert/quantize)

MERGE_DIR="${1:-models/qwen25vl-3b-vi-hwr-lora-merged}"
OUT_DIR="${2:-models/gguf}"
LLAMA_CPP="${LLAMA_CPP_DIR:-/content/llama.cpp}"
QUANT="${QUANT:-Q4_K_M}"
TAG=$(basename "$MERGE_DIR" | sed 's/-merged//')

if [ ! -d "$MERGE_DIR" ]; then
    echo "[ERR] Chua co merged model: $MERGE_DIR"
    echo "  Chay truoc: python scripts/export_merged.py"
    exit 1
fi

# Build llama.cpp nếu chưa có
if [ ! -f "$LLAMA_CPP/build/bin/llama-quantize" ]; then
    echo "Dang build llama.cpp tai: $LLAMA_CPP"
    [ -d "$LLAMA_CPP/.git" ] || git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP"
    cmake -S "$LLAMA_CPP" -B "$LLAMA_CPP/build" ${LLAMA_CUBLAS:+-DLLAMA_CUBLAS=ON}
    cmake --build "$LLAMA_CPP/build" --config Release -j"$(nproc)"
fi

mkdir -p "$OUT_DIR"

echo "Convert HF -> GGUF (f16)..."
"$LLAMA_CPP/convert_hf_to_gguf.py" "$MERGE_DIR" \
    --outfile "$OUT_DIR/$TAG-f16.gguf" --outtype f16

echo "Output:"
ls -lh "$OUT_DIR"/*.gguf

if [ "$QUANT" != "none" ]; then
    echo "Quantize -> $QUANT..."
    "$LLAMA_CPP/build/bin/llama-quantize" \
        "$OUT_DIR/$TAG-f16.gguf" \
        "$OUT_DIR/$TAG-$QUANT.gguf" "$QUANT"
    echo "GGUF da xong:"
    ls -lh "$OUT_DIR"/*.gguf
else
    echo "Bo qua quantize (QUANT=none). GGUF f16 da xong."
fi