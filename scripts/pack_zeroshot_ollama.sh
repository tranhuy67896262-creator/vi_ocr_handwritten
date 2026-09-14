#!/usr/bin/env bash
set -euo pipefail

# Dong goi base Qwen2.5-VL -> GGUF + mmproj -> Ollama (mac dinh ZERO-SHOT: KHONG train).
# Neu dat ADAPTER=<thu muc LoRA local> thi merge adapter vao base TRUOC khi convert.
#
# Vi du:
#   bash scripts/pack_zeroshot_ollama.sh 7b <hf_token>
#   ADAPTER=models/qwen25vl-7b-vi-hwr-lora bash scripts/pack_zeroshot_ollama.sh 7b <hf_token>
#   OLLAMA_REPO=<user>/qwen25vl-vi-ocr PUSH=1 bash scripts/pack_zeroshot_ollama.sh 7b <hf_token>
#
# Bien moi truong:
#   MODEL_NAME     base HF (mac dinh theo 3b/7b)
#   ADAPTER        (tuy chon) thu muc LoRA adapter local -> merge vao base truoc khi convert
#   QUANT          Q6_K (mac dinh) | Q8_0 | none
#   OLLAMA_MODEL   ten model local (mac dinh: OLLAMA_REPO neu co, khong thi qwen25vl-<tag>-vi-ocr-zeroshot)
#   OLLAMA_REPO    namespace/name de push (bat buoc khi PUSH=1)
#   PUSH           1 = ollama push len registry sau khi create
#   TEST_IMAGE     anh de test sau khi create (mac dinh assets/vi-handwriting-sample-1.png; rong de bo qua)
#   LLAMA_CPP_DIR  thu muc llama.cpp (mac dinh /content/llama.cpp)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Nhan doi so vi tri: [3b|7b] [HF_TOKEN]
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
    echo "Dung: bash scripts/pack_zeroshot_ollama.sh [3b|7b] [HF_TOKEN]"
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

QUANT="${QUANT:-Q6_K}"
ADAPTER="${ADAPTER:-}"
OLLAMA_REPO="${OLLAMA_REPO:-}"
PUSH="${PUSH:-0}"
OLLAMA_MODEL="${OLLAMA_MODEL:-${OLLAMA_REPO:-qwen25vl-${MODEL_TAG}-vi-ocr-zeroshot}}"
TEST_IMAGE="${TEST_IMAGE:-assets/vi-handwriting-sample-1.png}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-/content/llama.cpp}"

BASE_DIR="models/base-qwen25vl-${MODEL_TAG}"
MERGED_DIR="models/base-qwen25vl-${MODEL_TAG}-merged"
GGUF_DIR="models/gguf-zeroshot-${MODEL_TAG}-${ADAPTER:+merged}"

if [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="${PYTHON:-python}"
fi

for required_file in scripts/export_gguf.sh scripts/ollama_create.sh; do
    if [[ ! -f "$required_file" ]]; then
        echo "[ERR] Thieu file can thiet: $required_file"
        exit 1
    fi
done
if [[ -n "$ADAPTER" && ! -f scripts/export_merged.py ]]; then
    echo "[ERR] Thieu scripts/export_merged.py (can de merge ADAPTER)."
    exit 1
fi
if [[ -n "$ADAPTER" && ! -f "$ADAPTER/adapter_config.json" ]]; then
    echo "[ERR] ADAPTER khong hop le (thieu adapter_config.json): $ADAPTER"
    exit 1
fi
if ! command -v "$PYTHON" >/dev/null 2>&1 && [[ ! -x "$PYTHON" ]]; then
    echo "[ERR] Khong tim thay Python: $PYTHON"
    exit 1
fi
if [[ "$PUSH" == "1" && -z "$OLLAMA_REPO" ]]; then
    echo "[ERR] PUSH=1 nhung thieu OLLAMA_REPO (vd owner/qwen25vl-vi-ocr-zeroshot)."
    exit 1
fi

# 1/4 Base model: snapshot (bo qua neu co ADAPTER — export_merged.py tu tai base theo HF id)
echo "=== 1/4 Base model ==="
if [[ -n "$ADAPTER" ]]; then
    echo "  -> Co ADAPTER: base se tai qua scripts/export_merged.py (HF id). Bo qua snapshot."
elif [[ -f "$BASE_DIR/config.json" ]]; then
    echo "  -> Da co san: $BASE_DIR"
else
    echo "  -> snapshot_download $MODEL_NAME -> $BASE_DIR"
    MODEL_NAME="$MODEL_NAME" BASE_DIR="$BASE_DIR" \
        HF_TOKEN="${HF_TOKEN_ARG[0]:-${HF_TOKEN:-}}" HF_HOME="${HF_HOME:-$ROOT/.hf_cache}" \
        "$PYTHON" -c "import os; from huggingface_hub import snapshot_download; snapshot_download(os.environ['MODEL_NAME'], local_dir=os.environ['BASE_DIR'], token=os.environ.get('HF_TOKEN') or None)"
fi

# 2/4 (tuy chon) merge LoRA -> convert GGUF + mmproj
echo "=== 2/4 Convert GGUF + mmproj vision ==="
SRC_DIR="$BASE_DIR"
if [[ -n "$ADAPTER" ]]; then
    echo "  -> merge LoRA '$ADAPTER' vao base -> $MERGED_DIR"
    if ! "$PYTHON" scripts/export_merged.py --model "$MODEL_NAME" \
            --adapter "$ADAPTER" --output "$MERGED_DIR"; then
        echo "[ERR] export_merged.py that bai (can GPU, VRAM ~2x model)."
        exit 1
    fi
    SRC_DIR="$MERGED_DIR"
else
    echo "  -> zero-shot (khong adapter): convert thang base"
fi
mkdir -p "$GGUF_DIR"
rm -f "$GGUF_DIR"/mmproj-*.gguf "$GGUF_DIR/Modelfile"
LLAMA_CPP_DIR="$LLAMA_CPP_DIR" QUANT="$QUANT" bash scripts/export_gguf.sh "$SRC_DIR" "$GGUF_DIR"

shopt -s nullglob
MM_PROJ_FILES=("$GGUF_DIR"/mmproj-*.gguf)
MAIN_GGUF_FILES=()
for gguf_file in "$GGUF_DIR"/*.gguf; do
    [[ "$(basename "$gguf_file")" == mmproj-* ]] || MAIN_GGUF_FILES+=("$gguf_file")
done
shopt -u nullglob
if [[ ${#MM_PROJ_FILES[@]} -eq 0 || ! -s "${MM_PROJ_FILES[0]}" ]]; then
    echo "[ERR] Khong co mmproj vision trong $GGUF_DIR"
    exit 1
fi
if [[ ${#MAIN_GGUF_FILES[@]} -eq 0 || ! -s "${MAIN_GGUF_FILES[0]}" ]]; then
    echo "[ERR] Khong co GGUF text trong $GGUF_DIR"
    exit 1
fi
if [[ ! -s "$GGUF_DIR/Modelfile" ]]; then
    echo "[ERR] Khong co Modelfile trong $GGUF_DIR"
    exit 1
fi

# 3/4 Import vao Ollama
echo "=== 3/4 Import vao Ollama: $OLLAMA_MODEL ==="
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
bash scripts/ollama_create.sh "$OLLAMA_MODEL" "$GGUF_DIR"

# 4/4 Test (tuy chon) + push (tuy chon)
echo "=== 4/4 Test + push ==="
if [[ -n "$TEST_IMAGE" && -s "$TEST_IMAGE" ]]; then
    echo "  -> test OCR: $TEST_IMAGE"
    set +e
    # CLI attach anh bang cach nhan path thanh mot arg rieng (KHONG dung "-- path").
    OLLAMA_OUTPUT="$(ollama run "$OLLAMA_MODEL" "$TEST_IMAGE" \
        "Doc toan bo chu viet tay trong anh, chi tra ve van ban." 2>&1)"
    OLLAMA_STATUS=$?
    set -e
    printf '%s\n' "$OLLAMA_OUTPUT"
    if [[ "$OLLAMA_STATUS" -ne 0 || "$OLLAMA_OUTPUT" != *"Added image"* ]]; then
        echo "[WARN] Test that bai hoac chua nap duoc anh (thieu 'Added image'). Tiep tuc."
    fi
else
    echo "  -> Bo qua test (TEST_IMAGE rong/khong ton tai)."
fi

if [[ "$PUSH" == "1" ]]; then
    echo "  -> ollama push $OLLAMA_REPO"
    ollama push "$OLLAMA_REPO"
    echo "OK: da push. Nguoi khac: ollama pull $OLLAMA_REPO"
else
    echo "  -> Khong push (dat PUSH=1 OLLAMA_REPO=<user>/<name> de push)."
fi

echo "OK: Ollama model san sang: $OLLAMA_MODEL"
if [[ -n "$ADAPTER" ]]; then
    echo "  (da merge adapter: $ADAPTER)"
else
    echo "  (zero-shot, khong adapter)"
fi
echo "GGUF: $GGUF_DIR"
