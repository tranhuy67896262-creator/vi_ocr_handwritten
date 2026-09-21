#!/usr/bin/env bash
# Post pipeline sau 1 mốc staged: merge adapter -> GGUF(+mmproj/Modelfile) ->
# import Ollama -> test OCR thử -> push Ollama Hub. Chạy ngay sau train.py
# (stage_train.sh tự gọi; gọi tay cũng được).
#
# Cách dùng:
#   [POST_*=...] ./scripts/post_stage.sh <hub_rev> [run_name]
#   <hub_rev>: nhánh adapter vừa train xong (vd stage-5k).
#
# Env (đặt trước lệnh stage_train.sh là tự lan xuống):
#   MODEL         base model đầy đủ (bắt buộc, stage_train.sh tự có)
#   HUB_REPO      repo adapter owner/repo (bắt buộc, stage_train.sh tự có)
#   POST_MERGE=1  merge adapter@rev ra models/<repo>-<rev>-merged
#   POST_GGUF=1   convert merged -> models/gguf-<repo>-<rev> (+mmproj+Modelfile);
#                 tự bật POST_MERGE. Merge/convert cần GPU/RAM ~2x model.
#   POST_QUANT    loại quantize cho export_gguf.sh (mặc định Q6_K)
#   POST_OLLAMA=<ten>  ollama create <ten> từ gói gguf (tự bật POST_GGUF).
#                 Colab mới: thêm OLLAMA_AUTO_INSTALL=1 để tự cài CLI.
#   POST_TEST_IMG=<path>  OCR thử 1 ảnh sau khi create, in kết quả ra log.
#   OLLAMA_PUSH=<owner/model>  ollama push lên Ollama Hub (tự create nếu cần;
#                 cần đăng nhập trước: `ollama login`. Lỗi push chỉ WARN).
# Không bật gì -> in hướng dẫn rồi thoát 0 (không làm gì).
set -euo pipefail
cd "$(dirname "$0")/.."
LIB_DIR="$(pwd)/scripts/lib"
# shellcheck disable=SC1091
source "$LIB_DIR/common.sh"

HUB_REV="${1:?cần hub_rev (vd stage-5k)}"
RUN="${2:-$HUB_REV}"
MODEL="${MODEL:?cần env MODEL (base model đầy đủ)}"
HUB_REPO="${HUB_REPO:?cần env HUB_REPO (repo adapter owner/repo)}"
PYTHON="${PYTHON:-python3}"

# Bật bước sau thì tự kéo bước trước (ollama cần gguf, gguf cần merge).
if [ -n "${POST_OLLAMA:-}" ] || [ -n "${OLLAMA_PUSH:-}" ]; then
    POST_GGUF=1
fi
if [ "${POST_GGUF:-0}" = "1" ]; then
    POST_MERGE=1
fi

if [ "${POST_MERGE:-0}" != "1" ] && [ -z "${POST_OLLAMA:-}" ] && [ -z "${OLLAMA_PUSH:-}" ]; then
    echo "[post] Không bật POST_MERGE/POST_GGUF/POST_OLLAMA/OLLAMA_PUSH — bỏ qua."
    echo "  vd: POST_GGUF=1 POST_OLLAMA=vi-ocr-hwr:7b OLLAMA_PUSH=owner/vi-ocr-hwr:7b \\"
    echo "      ./scripts/stage_train.sh <token> qwenvl-7b <repo> 5k"
    exit 0
fi

TAG="$(basename "$HUB_REPO")-${HUB_REV}"
MERGE_DIR="models/${TAG}-merged"
GGUF_DIR="models/gguf-${TAG}"
echo "===== [post] ${RUN}: ${HUB_REPO}@${HUB_REV} -> ${GGUF_DIR} ====="

# 1. Merge adapter@rev vào base (thư mục theo rev nên dùng lại an toàn).
if [ "${POST_MERGE:-0}" = "1" ]; then
    if compgen -G "$MERGE_DIR/*.safetensors" > /dev/null; then
        echo "[post] Dùng merged có sẵn: $MERGE_DIR"
    else
        echo "[post] Merge ${HUB_REPO}@${HUB_REV} vào base..."
        "$PYTHON" scripts/export_merged.py \
            --model "$MODEL" --adapter "$HUB_REPO" \
            --adapter-revision "$HUB_REV" --output "$MERGE_DIR"
    fi
fi

# 2. Convert GGUF + mmproj + Modelfile (export_gguf.sh tự build llama.cpp).
if [ "${POST_GGUF:-0}" = "1" ]; then
    echo "[post] Convert GGUF (quant=${POST_QUANT:-Q6_K})..."
    QUANT="${POST_QUANT:-Q6_K}" scripts/export_gguf.sh "$MERGE_DIR" "$GGUF_DIR"
    MAIN_GGUF="$(ls "$GGUF_DIR"/*.gguf 2>/dev/null | grep -v mmproj | head -n 1 || true)"
    MMPROJ="$(ls "$GGUF_DIR"/mmproj-*.gguf 2>/dev/null | head -n 1 || true)"
    if [ -z "${MAIN_GGUF:-}" ] || [ ! -f "$GGUF_DIR/Modelfile" ]; then
        echo "[ERR] Thiếu gguf/Modelfile trong $GGUF_DIR — xem log convert."
        exit 1
    fi
    echo "[post] GGUF: $MAIN_GGUF"
    echo "[post] mmproj: ${MMPROJ:-KHÔNG CÓ (OCR ảnh sẽ lỗi 'provide the mmproj')}"
fi

# 3-5. Ollama create/test/push trong subshell: ensure_ollama_serve dùng `exit`
# nên lỗi ở đây không được giết cả job train (caller chỉ WARN).
if [ -n "${POST_OLLAMA:-}" ] || [ -n "${OLLAMA_PUSH:-}" ]; then
(
    ensure_ollama_serve
    NAME="${POST_OLLAMA:-$(basename "${OLLAMA_PUSH:?cần POST_OLLAMA hoặc OLLAMA_PUSH}")}"
    echo "[post] ollama create $NAME ..."
    (cd "$GGUF_DIR" && ollama create "$NAME" -f Modelfile)
    if [ -n "${POST_TEST_IMG:-}" ]; then
        echo "[post] OCR thử: $POST_TEST_IMG"
        "$PYTHON" -c 'import base64, json, sys, urllib.request
model, path = sys.argv[1], sys.argv[2]
with open(path, "rb") as fh:
    b64 = base64.b64encode(fh.read()).decode()
payload = {"model": model, "prompt": "Doc chinh xac toan bo chu viet tay trong anh. Chi tra ve van ban doc duoc.", "images": [b64], "stream": False}
req = urllib.request.Request("http://localhost:11434/api/generate",
                             data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=600) as res:
    print(json.load(res).get("response", ""))' \
            "$NAME" "$POST_TEST_IMG"
    fi
    if [ -n "${OLLAMA_PUSH:-}" ]; then
        if [ "$NAME" != "$OLLAMA_PUSH" ]; then
            ollama cp "$NAME" "$OLLAMA_PUSH"
        fi
        echo "[post] ollama push $OLLAMA_PUSH ..."
        ollama push "$OLLAMA_PUSH" \
            || echo "[WARN] Push Ollama Hub thất bại (thường do chưa 'ollama login'). Model local $NAME vẫn nguyên."
    fi
)
fi

echo "===== [post] Xong ${RUN} ====="
