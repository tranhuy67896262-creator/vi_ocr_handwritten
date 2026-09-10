#!/usr/bin/env bash
set -euo pipefail

# Chi chay sau khi pipeline_smoke_10.sh da thanh cong.
# Pipeline 20k cung duoc test HF, export GGUF, import va test Ollama.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

HF_TOKEN_ARG=()
if [[ $# -gt 0 && "${1:0:2}" != "--" ]]; then
    HF_TOKEN_ARG=("$1")
    shift
fi
if [[ $# -gt 0 ]]; then
    echo "[ERR] Tham so khong hop le: $1"
    echo "Dung: bash pipeline_train_20k.sh [HF_TOKEN]"
    exit 1
fi
if [[ ${#HF_TOKEN_ARG[@]} -eq 0 && -z "${HF_TOKEN:-}" ]] && ! grep -q '^HF_TOKEN' .env.dev 2>/dev/null; then
    echo "[ERR] Thieu HF_TOKEN cho gated dataset."
    echo "Dung: bash pipeline_train_20k.sh hf_xxxxx"
    echo "Hoac export HF_TOKEN / tao .env.dev truoc khi chay."
    exit 1
fi

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"
BATCH_SIZE="${BATCH_SIZE:-4}"
TEST_IMAGE="${TEST_IMAGE:-assets/vi-handwriting-sample-1.png}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen25vl-7b-vi-hwr-20k}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-/content/llama.cpp}"
if [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="${PYTHON:-python}"
fi

ADAPTER_DIR="models/qwen25vl-7b-vi-hwr-lora"
MERGED_DIR="models/qwen25vl-7b-vi-hwr-lora-20k-merged"
GGUF_DIR="models/gguf-20k"
MARKER="models/.pipeline_20k.ok"
rm -f "$MARKER"

if [[ "$MODEL_NAME" != "Qwen/Qwen2.5-VL-7B-Instruct" ]]; then
    echo "[ERR] Pipeline nay dang co dinh cho Qwen2.5-VL-7B-Instruct: $MODEL_NAME"
    exit 1
fi
if [[ ! "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERR] BATCH_SIZE phai la so nguyen duong: $BATCH_SIZE"
    exit 1
fi
if [[ ! -f "models/.pipeline_smoke_10.ok" ]]; then
    echo "[ERR] Smoke test chua thanh cong. Chay truoc:"
    echo "  bash pipeline_smoke_10.sh"
    exit 1
fi
if ! grep -Fqx "model=$MODEL_NAME" models/.pipeline_smoke_10.ok; then
    echo "[ERR] Smoke marker khong dung model: $MODEL_NAME"
    echo "  Xoa marker cu va chay lai pipeline_smoke_10.sh."
    exit 1
fi
if [[ ! -s "$TEST_IMAGE" ]]; then
    echo "[ERR] Khong tim thay anh test: $TEST_IMAGE"
    exit 1
fi
for required_file in run_train.sh scripts/eval_ocr.py scripts/export_merged.py \
                     scripts/export_gguf.sh scripts/ollama_create.sh; do
    if [[ ! -f "$required_file" ]]; then
        echo "[ERR] Thieu file can thiet: $required_file"
        exit 1
    fi
done
if ! command -v "$PYTHON" >/dev/null 2>&1 && [[ ! -x "$PYTHON" ]]; then
    echo "[ERR] Khong tim thay Python: $PYTHON"
    exit 1
fi
if ! "$PYTHON" -c "import torch, transformers, peft" >/dev/null 2>&1; then
    echo "[ERR] Python thieu torch/transformers/peft: $PYTHON"
    exit 1
fi

echo "=== 1/5 Train toi da 20.000 anh ==="
bash run_train.sh "${HF_TOKEN_ARG[@]}" --train \
    --model "$MODEL_NAME" \
    --max-samples 2000 \
    --batch-size "$BATCH_SIZE" \
    --epochs 1 \
    --save-steps 100

if [[ ! -f "$ADAPTER_DIR/adapter_config.json" ]]; then
    echo "[ERR] Khong thay adapter sau khi train: $ADAPTER_DIR"
    exit 1
fi

echo "=== 2/5 Eval 200 mau test bang adapter ==="
"$PYTHON" scripts/eval_ocr.py \
    --model "$MODEL_NAME" \
    --adapter "$ADAPTER_DIR" \
    --num-test 200

echo "=== 3/5 Merge LoRA vao model ==="
"$PYTHON" scripts/export_merged.py \
    --model "$MODEL_NAME" \
    --adapter "$ADAPTER_DIR" \
    --output "$MERGED_DIR"

echo "=== 4/5 Convert GGUF + mmproj vision ==="
mkdir -p "$GGUF_DIR"
rm -f "$GGUF_DIR"/mmproj-*.gguf "$GGUF_DIR/Modelfile"
LLAMA_CPP_DIR="$LLAMA_CPP_DIR" bash scripts/export_gguf.sh "$MERGED_DIR" "$GGUF_DIR"

shopt -s nullglob
MM_PROJ_FILES=("$GGUF_DIR"/mmproj-*.gguf)
MAIN_GGUF_FILES=()
for gguf_file in "$GGUF_DIR"/*.gguf; do
    [[ "$(basename "$gguf_file")" == mmproj-* ]] || MAIN_GGUF_FILES+=("$gguf_file")
done
shopt -u nullglob
if [[ ${#MM_PROJ_FILES[@]} -eq 0 || ! -s "${MM_PROJ_FILES[0]}" ]]; then
    echo "[ERR] Khong co mmproj vision. Khong duoc import Ollama."
    exit 1
fi
if [[ ${#MAIN_GGUF_FILES[@]} -eq 0 || ! -s "${MAIN_GGUF_FILES[0]}" ]]; then
    echo "[ERR] Khong co GGUF model text trong $GGUF_DIR"
    exit 1
fi
if [[ ! -s "$GGUF_DIR/Modelfile" ]]; then
    echo "[ERR] Khong co Modelfile trong $GGUF_DIR"
    exit 1
fi

echo "=== 5/5 Import va test Ollama ==="
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

if ! bash scripts/ollama_create.sh "$OLLAMA_MODEL" "$GGUF_DIR"; then
    echo "[ERR] Ollama create that bai cho model $OLLAMA_MODEL"
    exit 1
fi
set +e
OLLAMA_OUTPUT="$(ollama run "$OLLAMA_MODEL" \
    "Doc toan bo chu viet tay trong anh, chi tra ve van ban." \
    -- "$TEST_IMAGE" 2>&1)"
OLLAMA_STATUS=$?
set -e

if [[ "$OLLAMA_STATUS" -ne 0 || -z "${OLLAMA_OUTPUT//[[:space:]]/}" ]]; then
    echo "[ERR] Ollama run that bai (exit=$OLLAMA_STATUS):"
    printf '%s\n' "$OLLAMA_OUTPUT"
    exit 1
fi
printf '%s\n' "$OLLAMA_OUTPUT"

printf 'model=%s\nimage=%s\n' "$MODEL_NAME" "$TEST_IMAGE" > "$MARKER"
echo "OK: pipeline 20k + export + Ollama vision thanh cong."
echo "Marker: $MARKER"
