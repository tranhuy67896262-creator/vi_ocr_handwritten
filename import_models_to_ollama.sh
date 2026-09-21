#!/usr/bin/env bash
set -euo pipefail

# Import cac model GGUF da fine-tune vao Ollama.
# Mac dinh import ca smoke 10 anh va model 20k.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "scripts/lib/common.sh"

# Trả về đúng 1 thư mục khớp glob, hoặc rỗng (nhiều/không khớp -> để user chỉ tay).
discover_one_dir() {
    shopt -s nullglob
    local matches=($1)
    shopt -u nullglob
    local dirs=()
    local m
    for m in ${matches[@]+"${matches[@]}"}; do
        [[ -d "$m" ]] && dirs+=("$m")
    done
    if [[ ${#dirs[@]} -eq 1 ]]; then
        echo "${dirs[0]}"
    fi
    return 0
}

# gguf-smoke-7b-lora -> qwen25vl-7b-vi-hwr-smoke-lora (khớp tên pipeline.sh đặt).
# Trả về rỗng nếu thư mục không theo quy ước (caller dùng tên mặc định).
derive_ollama_name() {
    local base="${1#models/gguf-}" out=""
    if [[ "$2" == "smoke" ]]; then
        out="$(echo "$base" | sed -E 's/^smoke-([a-z0-9]+)-(.+)$/qwen25vl-\1-vi-hwr-smoke-\2/')"
    else
        out="$(echo "$base" | sed -E 's/^20k-([a-z0-9]+)-(.+)$/qwen25vl-\1-vi-hwr-20k-\2/')"
    fi
    case "$out" in
        qwen25vl-*) echo "$out" ;;
    esac
    return 0
}

# Mặc định tự dò output của pipeline.sh; nhiều model cùng loại thì chỉ tay qua env
# (vd SMOKE_DIR=models/gguf-smoke-3b-lora bash import_models_to_ollama.sh).
SMOKE_DIR="${SMOKE_DIR:-$(discover_one_dir "models/gguf-smoke-*")}"
FULL_DIR="${FULL_DIR:-$(discover_one_dir "models/gguf-20k-*")}"
SMOKE_MERGED_DIR="${SMOKE_MERGED_DIR:-$(discover_one_dir "models/qwen25vl-*-smoke-merged")}"
FULL_MERGED_DIR="${FULL_MERGED_DIR:-$(discover_one_dir "models/qwen25vl-*-20k-merged")}"
if [[ -z "${SMOKE_NAME:-}" ]]; then
    SMOKE_NAME="$(derive_ollama_name "$SMOKE_DIR" smoke)"
fi
if [[ -z "${FULL_NAME:-}" ]]; then
    FULL_NAME="$(derive_ollama_name "$FULL_DIR" 20k)"
fi
SMOKE_NAME="${SMOKE_NAME:-qwen25vl-7b-vi-hwr-smoke}"
FULL_NAME="${FULL_NAME:-qwen25vl-7b-vi-hwr-20k}"
QUANT="${QUANT:-Q6_K}"

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
if [[ "$QUANT" != "Q6_K" ]]; then
    echo "[ERR] Script nay chi import ban Q6_K, QUANT hien tai: $QUANT"
    exit 1
fi
if [[ ! -f "scripts/ollama_create.sh" ]]; then
    echo "[ERR] Thieu scripts/ollama_create.sh"
    exit 1
fi

ensure_ollama_serve

import_one() {
    local model_name="$1"
    local gguf_dir="$2"
    local merged_dir="$3"

    if [[ ! -d "$gguf_dir" ]]; then
        if [[ ! -d "$merged_dir" ]]; then
            echo "[WARN] Bo qua model chua co GGUF/merged: $model_name"
            return 0
        fi
        mkdir -p "$gguf_dir"
    fi

    shopt -s nullglob
    local q6_files=("$gguf_dir"/*-Q6_K.gguf)
    shopt -u nullglob
    if [[ ${#q6_files[@]} -eq 0 || ! -s "${q6_files[0]}" || ! -s "$gguf_dir/Modelfile" ]] || \
       ! grep -Fq "Q6_K.gguf" "$gguf_dir/Modelfile"; then
        if [[ ! -d "$merged_dir" ]]; then
            echo "[ERR] Chua co GGUF Q6_K va thieu merged model: $merged_dir"
            echo "  Hay chay pipeline fine-tune va export truoc."
            exit 1
        fi
        if [[ ! -f "scripts/export_gguf.sh" ]]; then
            echo "[ERR] Thieu scripts/export_gguf.sh"
            exit 1
        fi
        echo "=== Export lai Q6_K cho $model_name ==="
        LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-/content/llama.cpp}" \
            QUANT="$QUANT" bash scripts/export_gguf.sh "$merged_dir" "$gguf_dir"
    fi

    shopt -s nullglob
    local mmproj_files=("$gguf_dir"/mmproj-*.gguf)
    q6_files=("$gguf_dir"/*-Q6_K.gguf)
    shopt -u nullglob
    if [[ ${#q6_files[@]} -eq 0 || ! -s "${q6_files[0]}" ]]; then
        echo "[ERR] Khong tao duoc GGUF Q6_K trong $gguf_dir"
        exit 1
    fi
    if [[ ${#mmproj_files[@]} -eq 0 || ! -s "${mmproj_files[0]}" ]]; then
        echo "[ERR] Thieu mmproj vision trong $gguf_dir"
        exit 1
    fi
    if [[ ! -s "$gguf_dir/Modelfile" ]] || ! grep -Fq "Q6_K.gguf" "$gguf_dir/Modelfile"; then
        echo "[ERR] Modelfile khong tro toi GGUF Q6_K: $gguf_dir/Modelfile"
        exit 1
    fi

    echo "=== Import $model_name tu $gguf_dir ==="
    if ! bash scripts/ollama_create.sh "$model_name" "$gguf_dir"; then
        echo "[ERR] Import Ollama that bai: $model_name"
        exit 1
    fi
    IMPORTED_COUNT=$((IMPORTED_COUNT + 1))
    if [[ "$model_name" == "$FULL_NAME" ]]; then
        FULL_IMPORTED=1
    fi
}

IMPORTED_COUNT=0
FULL_IMPORTED=0
import_one "$SMOKE_NAME" "$SMOKE_DIR" "$SMOKE_MERGED_DIR"
import_one "$FULL_NAME" "$FULL_DIR" "$FULL_MERGED_DIR"

if [[ "$IMPORTED_COUNT" -eq 0 ]]; then
    echo "[ERR] Khong co model GGUF nao de import."
    exit 1
fi

echo
echo "Danh sach model Ollama:"
ollama list
if [[ "$FULL_IMPORTED" -eq 1 ]]; then
    echo
    echo "Test model 20k:"
    echo "  ollama run $FULL_NAME \"Doc chu trong anh\" -- /path/to/anh.jpg"
else
    echo
    echo "[INFO] Chua co model 20k; da import cac model hien co."
fi
