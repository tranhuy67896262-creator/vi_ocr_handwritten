#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Export GGUF từ merged HF model (gom ca text model + mmproj vision).
# Bước 1: python scripts/export_merged.py        (cần trước)
# Bước 2: ./scripts/export_gguf.sh                (convert + quantize + mmproj + Modelfile)
# Output trong OUT_DIR: *-f16.gguf, *-Q4_K_M.gguf, mmproj-*.gguf, Modelfile (FROM + ADAPTER).
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

if [ "$QUANT" != "none" ]; then
    # Convert thang ra ban quantize (nhanh gap doi: bo qua file f16 trung gian ~6GB).
    echo "Convert HF -> GGUF ($QUANT, truc tiep)..."
    if "$LLAMA_CPP/convert_hf_to_gguf.py" "$MERGE_DIR" \
        --outfile "$OUT_DIR/$TAG-$QUANT.gguf" \
        --outtype "$(echo "$QUANT" | tr '[:upper:]' '[:lower:]')"; then
        echo "GGUF da xong:"
        ls -lh "$OUT_DIR"/*.gguf
    else
        echo "[WARN] convert truc tiep that bai, fallback f16 + llama-quantize..."
        "$LLAMA_CPP/convert_hf_to_gguf.py" "$MERGE_DIR" \
            --outfile "$OUT_DIR/$TAG-f16.gguf" --outtype f16
        echo "Quantize -> $QUANT..."
        "$LLAMA_CPP/build/bin/llama-quantize" \
            "$OUT_DIR/$TAG-f16.gguf" \
            "$OUT_DIR/$TAG-$QUANT.gguf" "$QUANT"
        echo "GGUF da xong:"
        ls -lh "$OUT_DIR"/*.gguf
    fi
else
    echo "Convert HF -> GGUF (f16)..."
    "$LLAMA_CPP/convert_hf_to_gguf.py" "$MERGE_DIR" \
        --outfile "$OUT_DIR/$TAG-f16.gguf" --outtype f16
    echo "Bo qua quantize (QUANT=none). GGUF f16 da xong."
fi

# mmproj (vision projector): Qwen-VL OCR ảnh BẮT BUỘC cần file này.
# Converter chạy lần 2 với --mmproj (llama.cpp mới hỗ trợ qwen2/2.5vl).
echo "Convert mmproj (projector vision - OCR anh can file nay)..."
"$LLAMA_CPP/convert_hf_to_gguf.py" --mmproj "$MERGE_DIR" \
    --outfile "$OUT_DIR/mmproj-$TAG-f16.gguf" --outtype f16 \
    || echo "[WARN] convert mmproj that bai (llama.cpp cu?). Tiep tuc khong co mmproj."

# mmproj: tim file thuc te (converter co the dat ten hoi khac), khong doan ten.
# Ollama nap vision qua Modelfile (FROM text + ADAPTER mmproj).
MMPROJ="$(ls "$OUT_DIR"/mmproj-*.gguf 2>/dev/null | head -n 1 || true)"
if [ -z "$MMPROJ" ]; then
    echo "[WARN] Khong thay file mmproj-*.gguf trong $OUT_DIR."
    echo "  Ollama/llama.cpp OCR anh se loi 'provide the mmproj'."
    echo "  Nguyen nhan thuong gap: llama.cpp cu (clone tu truoc, chua ho tro qwen25vl)."
    echo "  Fix: rm -rf \"$LLAMA_CPP\" && $0 \"$MERGE_DIR\" \"$OUT_DIR\""
else
    echo "mmproj: $MMPROJ"
    OLLAMA_GGUF="$OUT_DIR/$TAG-$QUANT.gguf"
    [ -f "$OLLAMA_GGUF" ] || OLLAMA_GGUF="$OUT_DIR/$TAG-f16.gguf"
    cat > "$OUT_DIR/Modelfile" <<EOF
FROM ./$(basename "$OLLAMA_GGUF")
ADAPTER ./$(basename "$MMPROJ")
EOF
    echo "Da viet Modelfile: $OUT_DIR/Modelfile"
    echo
    echo "Chay tren Ollama:"
    echo "  cd $OUT_DIR && ollama create <ten-model> -f Modelfile"
    echo "  ollama run <ten-model> \"Doc chu trong anh\" -- /path/to/anh.jpg"
    echo "  ollama push <owner>/<ten-model>"
fi