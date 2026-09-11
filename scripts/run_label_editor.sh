#!/usr/bin/env bash
# Mo OCR Label Editor (Hugging Face + anh ca nhan) tren cong 9000.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif [ -x ".venv/Scripts/python.exe" ]; then
    PYTHON=".venv/Scripts/python.exe"
else
    PYTHON="$(command -v python3 || command -v python)"
fi

exec "$PYTHON" -m scripts.labeling serve --port "${PORT:-9000}" "$@"
