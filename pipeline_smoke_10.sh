#!/usr/bin/env bash
set -euo pipefail

# Chay smoke test 10 anh: train -> OCR HF -> merge -> GGUF -> Ollama -> OCR anh.
# Chi ghi marker sau khi Ollama tra ve ket qua khong rong.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Nhan doi so vi tri: [3b|7b] [HF_TOKEN] (model la tuy chon, token la cai con lai).
MODEL_CHOICE=""
if [[ $# -gt 0 && "${1:0:2}" != "--" ]]; then
    case "${1,,}" in
        3b) MODEL_CHOICE="3b"; shift ;;
        7b) MODEL_CHOICE="7b"; shift ;;
    esac
fi
HF_TOKEN_ARG=()
if [[ $# -gt 0 && "${1:0:2}" != "--" ]]; then
    HF_TOKEN_ARG=("$1")
    shift
fi
if [[ $# -gt 0 ]]; then
    echo "[ERR] Tham so khong hop le: $1"
    echo "Dung: bash pipeline_smoke_10.sh [3b|7b] [HF_TOKEN]"
    echo "Mac dinh gop data assets/ + repo. Tat: MERGE=0 bash pipeline_smoke_10.sh [3b|7b] [HF_TOKEN]"
    exit 1
fi
if [[ ${#HF_TOKEN_ARG[@]} -eq 0 && -z "${HF_TOKEN:-}" ]] && ! grep -q '^HF_TOKEN' .env.dev 2>/dev/null; then
    echo "[ERR] Thieu HF_TOKEN cho gated dataset."
    echo "Dung: bash pipeline_smoke_10.sh [3b|7b] hf_xxxxx"
    echo "Hoac export HF_TOKEN / tao .env.dev truoc khi chay."
    exit 1
fi

if [[ -n "$MODEL_CHOICE" && -z "${MODEL_NAME:-}" ]]; then
    MODEL_NAME="Qwen/Qwen2.5-VL-${MODEL_CHOICE^^}-Instruct"
fi
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"
case "$MODEL_NAME" in
    *3B*|*3b*) MODEL_TAG="3b" ;;
    *7B*|*7b*) MODEL_TAG="7b" ;;
    *) MODEL_TAG="" ;;
esac
if [[ -z "$MODEL_TAG" ]]; then
    echo "[ERR] Chi ho tro Qwen2.5-VL 3B hoac 7B: $MODEL_NAME"
    exit 1
fi
DATASET_NAME="${DATASET_NAME:-}"
# Mac dinh batch 2 (vua GPU ~15GB nhu truoc). Neu van OOM: NO_KL=1 hoac MAX_SEQ_LEN=512.
BATCH_SIZE="${BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
NO_KL="${NO_KL:-0}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-}"
TEST_IMAGE="${TEST_IMAGE:-assets/vi-handwriting-sample-1.png}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen25vl-${MODEL_TAG}-vi-hwr-smoke}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-/content/llama.cpp}"
# Mac dinh MERGE=1: gop Hugging Face + assets/labels.csv -> data/combined roi train.
# Dat MERGE=0 de chi train tren dataset HF (bo qua data noi bo).
MERGE="${MERGE:-1}"
if [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="${PYTHON:-python}"
fi

ADAPTER_DIR="models/qwen25vl-${MODEL_TAG}-vi-hwr-lora"
MERGED_DIR="models/qwen25vl-${MODEL_TAG}-vi-hwr-lora-smoke-merged"
GGUF_DIR="models/gguf-smoke-${MODEL_TAG}"
MARKER="models/.pipeline_smoke_10.ok"
rm -f "$MARKER"

if [[ ! "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERR] BATCH_SIZE phai la so nguyen duong: $BATCH_SIZE"
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
    echo "[WARN] Python thieu torch/transformers/peft — run_train.sh (--train) se cai truoc khi train."
fi

# Dem so mau local co nhan trong assets/labels.csv (cong vao --max-samples).
count_local_samples() {
    "$PYTHON" -c 'import csv
n = 0
try:
    with open("assets/labels.csv", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if (row.get("text") or "").strip():
                n += 1
except OSError:
    n = 0
print(n)' 2>/dev/null || echo 0
}

echo "=== 1/5 Train data repo + data assets ==="
TRAIN_DATASET_ARG=()
if [[ -n "$DATASET_NAME" ]]; then
    TRAIN_DATASET_ARG=(--dataset "$DATASET_NAME")
fi
REPO_SAMPLES="${REPO_SAMPLES:-10}"
MERGE_ARG=()
LOCAL_COUNT=0
if [[ "$MERGE" == "1" ]]; then
    MERGE_ARG=(--merge)
    LOCAL_COUNT="$(count_local_samples)"
    case "$LOCAL_COUNT" in ''|*[!0-9]*) LOCAL_COUNT=0 ;; esac
    echo "Che do MERGE: Hugging Face + assets/labels.csv ($LOCAL_COUNT mau local) -> data/combined"
fi
SMOKE_MAX=$((REPO_SAMPLES + LOCAL_COUNT))
echo "  -> train $SMOKE_MAX mau ($REPO_SAMPLES tu repo + $LOCAL_COUNT tu assets) | batch=$BATCH_SIZE no_kl=$NO_KL"
EXTRA_TRAIN_ARGS=()
if [[ "$NO_KL" == "1" ]]; then
    EXTRA_TRAIN_ARGS+=(--no-kl)
fi
if [[ -n "$MAX_SEQ_LEN" ]]; then
    EXTRA_TRAIN_ARGS+=(--max-seq-len "$MAX_SEQ_LEN")
fi
bash run_train.sh "${HF_TOKEN_ARG[@]}" --train \
    "${MERGE_ARG[@]}" \
    --model "$MODEL_NAME" \
    "${TRAIN_DATASET_ARG[@]}" \
    --max-samples "$SMOKE_MAX" \
    --batch-size "$BATCH_SIZE" \
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
    "${EXTRA_TRAIN_ARGS[@]}" \
    --epochs 1 \
    --save-steps 5

if [[ ! -f "$ADAPTER_DIR/adapter_config.json" ]]; then
    echo "[ERR] Khong thay adapter sau khi train: $ADAPTER_DIR"
    exit 1
fi

echo "=== 2/5 Test adapter bang pipeline HF ==="
EVAL_DATASET_ARG=()
if [[ -n "$DATASET_NAME" ]]; then
    EVAL_DATASET_ARG=(--dataset "$DATASET_NAME")
fi
"$PYTHON" scripts/eval_ocr.py \
    --model "$MODEL_NAME" \
    --adapter "$ADAPTER_DIR" \
    "${EVAL_DATASET_ARG[@]}" \
    --image "$TEST_IMAGE"

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
    echo "[ERR] Khong co mmproj vision. Khong duoc tiep tuc import Ollama."
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

mkdir -p "$(dirname "$MARKER")"
printf 'model=%s\nimage=%s\n' "$MODEL_NAME" "$TEST_IMAGE" > "$MARKER"
echo "OK: smoke test 10 anh + export + Ollama vision thanh cong."
echo "Marker: $MARKER"
