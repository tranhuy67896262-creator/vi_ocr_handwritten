#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Import .gguf + Modelfile vào Ollama (chạy sau export_gguf.sh).
# Dùng: ./scripts/ollama_create.sh [ten-model] [thu-muc-gguf]
# Vd:   ./scripts/ollama_create.sh qwen25vl-7b-vi-hwr models/gguf

MODEL_NAME="${1:-qwen25vl-7b-vi-hwr}"
OUT_DIR="${2:-models/gguf}"

if [ ! -f "$OUT_DIR/Modelfile" ]; then
    echo "[ERR] Thieu $OUT_DIR/Modelfile — chay ./scripts/export_gguf.sh truoc."
    exit 1
fi
if ! command -v ollama >/dev/null 2>&1; then
    echo "[ERR] Chua co Ollama CLI — cai: curl -fsSL https://ollama.com/install.sh | sh"
    exit 1
fi
if ! ollama list >/dev/null 2>&1; then
    echo "[ERR] Ollama server chua chay — chay: ollama serve (Colab chay nen: ollama serve > /tmp/ollama.log 2>&1 &)"
    exit 1
fi

echo "=== $OUT_DIR/Modelfile ==="
cat "$OUT_DIR/Modelfile"
echo "=========================="
echo "> ollama create $MODEL_NAME -f Modelfile (cwd=$OUT_DIR)"
(cd "$OUT_DIR" && ollama create "$MODEL_NAME" -f Modelfile)
echo "OK — test: ollama run $MODEL_NAME \"Doc chu trong anh\" -- /path/to/anh.jpg"
ollama list
