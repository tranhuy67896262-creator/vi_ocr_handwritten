#!/usr/bin/env bash
set -euo pipefail

# Pipeline end-to-end: train -> eval -> merge -> GGUF -> Ollama.
# (Gộp pipeline_smoke_10.sh + pipeline_train_20k.sh cũ, behavior giữ nguyên.)
#
#   bash pipeline.sh <smoke|train> [3b|7b] [HF_TOKEN]
#
# Mapping từ lệnh cũ:
#   pipeline_smoke_10.sh 3b <token>       -> pipeline.sh smoke 3b <token>
#   pipeline_train_20k.sh 7b <token>      -> pipeline.sh train 7b <token>
#   pipeline_smoke_10_lora.sh ...         -> USE_4BIT=0 pipeline.sh smoke ...
#   pipeline_train_20k_lora.sh ...        -> USE_4BIT=0 pipeline.sh train ...
#
# Env chung: MODEL_NAME, DATASET_NAME, BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
#   NO_KL, USE_4BIT (=0 chạy LoRA bf16, tách artifact -lora-bf16), MAX_SEQ_LEN,
#   TEST_IMAGE, OLLAMA_MODEL, LLAMA_CPP_DIR, MERGE (=1 gộp assets/labels.csv,
#   tắt bằng MERGE=0), OLLAMA_AUTO_START, PYTHON.
# Riêng smoke: REPO_SAMPLES (=10), save-steps 5, eval 1 ảnh.
# Riêng train: MAX_SAMPLES (=5000 repo + local; 0 = toàn bộ), EPOCHS (=1),
#   EVAL_SAMPLES (=200), save-steps 100. Chỉ chạy sau khi smoke cùng model OK.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "scripts/lib/common.sh"

MODE="${1:-}"
case "$MODE" in
    smoke|train) shift ;;
    *) echo "[ERR] Thieu mode: bash pipeline.sh <smoke|train> [3b|7b] [HF_TOKEN]"; exit 1 ;;
esac

# Đối số vị trí: [3b|7b] [HF_TOKEN] (model là tùy chọn, token là cái còn lại).
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
    echo "Dung: bash pipeline.sh $MODE [3b|7b] [HF_TOKEN]"
    exit 1
fi
if [[ ${#HF_TOKEN_ARG[@]} -eq 0 ]]; then
    require_hf_token "bash pipeline.sh $MODE [3b|7b] hf_xxxxx"
fi

if [[ -n "$MODEL_CHOICE" && -z "${MODEL_NAME:-}" ]]; then
    MODEL_NAME="tranhuy67896262/Qwen2.5-VL-${MODEL_CHOICE^^}-Instruct-private"
fi
MODEL_NAME="${MODEL_NAME:-tranhuy67896262/Qwen2.5-VL-7B-Instruct-private}"
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
NO_KL="${NO_KL:-0}"
USE_4BIT="${USE_4BIT:-1}"
# Tách artifact qlora (lora) và bf16 (lora-bf16) để không ghi đè nhau.
if [[ "$USE_4BIT" == "0" ]]; then ADAPTER_SUFFIX="lora-bf16"; else ADAPTER_SUFFIX="lora"; fi
MAX_SEQ_LEN="${MAX_SEQ_LEN:-}"
TEST_IMAGE="${TEST_IMAGE:-assets/vi-handwriting-sample-1.png}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-/content/llama.cpp}"
# Mặc định MERGE=1: gộp Hugging Face + assets/labels.csv -> data/combined rồi train.
# Đặt MERGE=0 để chỉ train trên dataset HF (bỏ qua data nội bộ).
MERGE="${MERGE:-1}"
if [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="${PYTHON:-python}"
fi

if [[ "$MODE" == "smoke" ]]; then
    BATCH_SIZE="${BATCH_SIZE:-2}"
    SAVE_STEPS=5
    ADAPTER_DIR="models/qwen25vl-${MODEL_TAG}-vi-hwr-${ADAPTER_SUFFIX}"
    MERGED_DIR="models/qwen25vl-${MODEL_TAG}-vi-hwr-${ADAPTER_SUFFIX}-smoke-merged"
    GGUF_DIR="models/gguf-smoke-${MODEL_TAG}-${ADAPTER_SUFFIX}"
    MARKER="models/.pipeline_smoke_10.ok"
    OLLAMA_MODEL="${OLLAMA_MODEL:-qwen25vl-${MODEL_TAG}-vi-hwr-smoke-${ADAPTER_SUFFIX}}"
    REPO_SAMPLES="${REPO_SAMPLES:-10}"
else
    BATCH_SIZE="${BATCH_SIZE:-8}"
    SAVE_STEPS=100
    ADAPTER_DIR="models/qwen25vl-${MODEL_TAG}-vi-hwr-${ADAPTER_SUFFIX}"
    MERGED_DIR="models/qwen25vl-${MODEL_TAG}-vi-hwr-${ADAPTER_SUFFIX}-20k-merged"
    GGUF_DIR="models/gguf-20k-${MODEL_TAG}-${ADAPTER_SUFFIX}"
    MARKER="models/.pipeline_20k.ok"
    OLLAMA_MODEL="${OLLAMA_MODEL:-qwen25vl-${MODEL_TAG}-vi-hwr-20k-${ADAPTER_SUFFIX}}"
    MAX_SAMPLES="${MAX_SAMPLES:-5000}"
    EVAL_SAMPLES="${EVAL_SAMPLES:-200}"
    EPOCHS="${EPOCHS:-1}"
fi
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
rm -f "$MARKER"

if [[ ! "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERR] BATCH_SIZE phai la so nguyen duong: $BATCH_SIZE"
    exit 1
fi
if [[ "$MODE" == "train" ]]; then
    if [[ ! -f "models/.pipeline_smoke_10.ok" ]]; then
        echo "[ERR] Smoke test chua thanh cong. Chay truoc:"
        echo "  bash pipeline.sh smoke"
        exit 1
    fi
    if ! grep -Fqx "model=$MODEL_NAME" models/.pipeline_smoke_10.ok; then
        echo "[ERR] Smoke marker khong dung model: $MODEL_NAME"
        echo "  Xoa marker cu va chay lai bash pipeline.sh smoke."
        exit 1
    fi
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

echo "=== 1/5 Train data repo + data assets ==="
TRAIN_DATASET_ARG=()
if [[ -n "$DATASET_NAME" ]]; then
    TRAIN_DATASET_ARG=(--dataset "$DATASET_NAME")
fi
MERGE_ARG=()
LOCAL_COUNT=0
if [[ "$MERGE" == "1" ]]; then
    MERGE_ARG=(--merge)
    LOCAL_COUNT="$(count_local_samples)"
    case "$LOCAL_COUNT" in ''|*[!0-9]*) LOCAL_COUNT=0 ;; esac
    echo "Che do MERGE: Hugging Face + assets/labels.csv ($LOCAL_COUNT mau local) -> data/combined"
fi
MAX_SAMPLES_ARG=()
EPOCHS_ARG=(--epochs 1)
if [[ "$MODE" == "smoke" ]]; then
    SMOKE_MAX=$((REPO_SAMPLES + LOCAL_COUNT))
    MAX_SAMPLES_ARG=(--max-samples "$SMOKE_MAX")
    echo "  -> train $SMOKE_MAX mau ($REPO_SAMPLES tu repo + $LOCAL_COUNT tu assets) | batch=$BATCH_SIZE no_kl=$NO_KL use_4bit=$USE_4BIT"
else
    EPOCHS_ARG=(--epochs "$EPOCHS")
    if [[ "$MAX_SAMPLES" == "0" ]]; then
        echo "  -> train TOAN BO dataset (khong gioi han mau) | batch=$BATCH_SIZE no_kl=$NO_KL use_4bit=$USE_4BIT epochs=$EPOCHS"
    else
        REPO_SAMPLES_TRAIN="${REPO_SAMPLES:-$MAX_SAMPLES}"
        MAX_SAMPLES=$((REPO_SAMPLES_TRAIN + LOCAL_COUNT))
        MAX_SAMPLES_ARG=(--max-samples "$MAX_SAMPLES")
        echo "  -> train $MAX_SAMPLES mau ($REPO_SAMPLES_TRAIN tu repo + $LOCAL_COUNT tu assets) | batch=$BATCH_SIZE no_kl=$NO_KL use_4bit=$USE_4BIT epochs=$EPOCHS"
    fi
fi
EXTRA_TRAIN_ARGS=()
if [[ "$NO_KL" == "1" ]]; then
    EXTRA_TRAIN_ARGS+=(--no-kl)
fi
if [[ "$USE_4BIT" == "0" ]]; then
    EXTRA_TRAIN_ARGS+=(--no-4bit)
fi
if [[ -n "$MAX_SEQ_LEN" ]]; then
    EXTRA_TRAIN_ARGS+=(--max-seq-len "$MAX_SEQ_LEN")
fi
bash run_train.sh "${HF_TOKEN_ARG[@]}" --train \
    "${MERGE_ARG[@]}" \
    --model "$MODEL_NAME" \
    "${TRAIN_DATASET_ARG[@]}" \
    "${MAX_SAMPLES_ARG[@]}" \
    --batch-size "$BATCH_SIZE" \
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
    "${EXTRA_TRAIN_ARGS[@]}" \
    "${EPOCHS_ARG[@]}" \
    --save-steps "$SAVE_STEPS"

if [[ ! -f "$ADAPTER_DIR/adapter_config.json" ]]; then
    echo "[ERR] Khong thay adapter sau khi train: $ADAPTER_DIR"
    exit 1
fi

EVAL_DATASET_ARG=()
if [[ -n "$DATASET_NAME" ]]; then
    EVAL_DATASET_ARG=(--dataset "$DATASET_NAME")
fi
if [[ "$MODE" == "smoke" ]]; then
    echo "=== 2/5 Test adapter bang pipeline HF ==="
    "$PYTHON" scripts/eval_ocr.py \
        --model "$MODEL_NAME" \
        --adapter "$ADAPTER_DIR" \
        "${EVAL_DATASET_ARG[@]}" \
        --image "$TEST_IMAGE"
else
    echo "=== 2/5 Eval $EVAL_SAMPLES mau test bang adapter ==="
    "$PYTHON" scripts/eval_ocr.py \
        --model "$MODEL_NAME" \
        --adapter "$ADAPTER_DIR" \
        "${EVAL_DATASET_ARG[@]}" \
        --num-test "$EVAL_SAMPLES"
fi

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
ensure_ollama_serve

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
if [[ "$MODE" == "smoke" ]]; then
    echo "OK: smoke test + export + Ollama vision thanh cong."
else
    echo "OK: pipeline train $MAX_SAMPLES anh + export + Ollama vision thanh cong."
fi
echo "Marker: $MARKER"
