#!/usr/bin/env bash
set -euo pipefail

# Import cac model GGUF da fine-tune vao Ollama.
# Mac dinh import ca smoke 10 anh va model 20k.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

SMOKE_DIR="${SMOKE_DIR:-models/gguf-smoke}"
FULL_DIR="${FULL_DIR:-models/gguf-20k}"
SMOKE_NAME="${SMOKE_NAME:-qwen25vl-7b-vi-hwr-smoke}"
FULL_NAME="${FULL_NAME:-qwen25vl-7b-vi-hwr-20k}"

if [[ -z "$SMOKE_NAME" || -z "$FULL_NAME" ]]; then
    echo "[ERR] Ten model Ollama khong duoc rong."
    exit 1
fi
for model_name in "$SMOKE_NAME" "$FULL_NAME"; do
    if [[ ! "$model_name" =~ ^[a-z0-9][a-z0-9._:-]*$ ]]; then
        echo "[ERR] Ten model Ollama khong hop le: $model_name"
        exit 1
    fi
done
if [[ ! -f "scripts/ollama_create.sh" ]]; then
    echo "[ERR] Thieu scripts/ollama_create.sh"
    exit 1
fi

if ! command -v ollama >/dev/null 2>&1; then
    echo "[ERR] Chua cai Ollama CLI."
    exit 1
fi

if ! ollama list >/dev/null 2>&1; then
    if [[ "${OLLAMA_AUTO_START:-1}" != "1" ]]; then
        echo "[ERR] Ollama server chua chay. Chay 'ollama serve' roi thu lai."
        exit 1
    fi
    echo "Khoi dong Ollama server..."
    ollama serve > /tmp/ollama-vi-ocr.log 2>&1 &
    for _ in $(seq 1 30); do
        sleep 1
        ollama list >/dev/null 2>&1 && break
    done
    ollama list >/dev/null 2>&1 || {
        echo "[ERR] Khong khoi dong duoc Ollama. Xem /tmp/ollama-vi-ocr.log"
        exit 1
    }
fi

import_one() {
    local model_name="$1"
    local gguf_dir="$2"

    if [[ ! -d "$gguf_dir" ]]; then
        echo "[WARN] Bo qua model chua ton tai: $gguf_dir"
        return 0
    fi
    if [[ ! -s "$gguf_dir/Modelfile" ]]; then
        echo "[ERR] Thieu $gguf_dir/Modelfile"
        echo "  Hay chay pipeline fine-tune va export truoc."
        exit 1
    fi
    shopt -s nullglob
    local mmproj_files=("$gguf_dir"/mmproj-*.gguf)
    local main_gguf_files=()
    local gguf_file
    for gguf_file in "$gguf_dir"/*.gguf; do
        [[ "$(basename "$gguf_file")" == mmproj-* ]] || main_gguf_files+=("$gguf_file")
    done
    shopt -u nullglob
    if [[ ${#mmproj_files[@]} -eq 0 || ! -s "${mmproj_files[0]}" ]]; then
        echo "[ERR] Thieu mmproj vision trong $gguf_dir"
        exit 1
    fi
    if [[ ${#main_gguf_files[@]} -eq 0 || ! -s "${main_gguf_files[0]}" ]]; then
        echo "[ERR] Thieu GGUF model text trong $gguf_dir"
        exit 1
    fi

    echo "=== Import $model_name tu $gguf_dir ==="
    if ! bash scripts/ollama_create.sh "$model_name" "$gguf_dir"; then
        echo "[ERR] Import Ollama that bai: $model_name"
        exit 1
    fi
    IMPORTED_COUNT=$((IMPORTED_COUNT + 1))
}

IMPORTED_COUNT=0
import_one "$SMOKE_NAME" "$SMOKE_DIR"
import_one "$FULL_NAME" "$FULL_DIR"

if [[ "$IMPORTED_COUNT" -eq 0 ]]; then
    echo "[ERR] Khong co model GGUF nao de import."
    exit 1
fi

echo
echo "Danh sach model Ollama:"
ollama list
echo
echo "Test model 20k:"
echo "  ollama run $FULL_NAME \"Doc chu trong anh\" -- /path/to/anh.jpg"
