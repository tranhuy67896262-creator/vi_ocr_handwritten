#!/usr/bin/env bash
# Lib dùng chung cho các script train/pipeline (chỉ source, không chạy trực tiếp).
#   source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"   # caller trong scripts/
#   source "scripts/lib/common.sh"                                     # caller ở root
# Caller tự `set -euo pipefail` và `cd` đúng thư mục trước khi gọi.
# Quy ước bash cũ + set -u: lib không dùng mảng (mảng do caller giữ),
# mọi biến env đều đọc qua ${VAR:-} nên source luôn an toàn.

# "10k"/"10K" -> 10000, số thường giữ nguyên.
to_samples() {
    case "$1" in
        *[kK]) echo $(( ${1%[kK]} * 1000 )) ;;
        *) echo "$1" ;;
    esac
}

# "3b"/"qwenvl-3b"/... -> tên model private đầy đủ; tên đầy đủ giữ nguyên.
resolve_model() {
    case "$(echo "$1" | tr '[:upper:]' '[:lower:]')" in
        3b|qwenvl-3b|qwen25vl-3b|qwen2.5-vl-3b) echo "tranhuy67896262/Qwen2.5-VL-3B-Instruct-private" ;;
        7b|qwenvl-7b|qwen25vl-7b|qwen2.5-vl-7b) echo "tranhuy67896262/Qwen2.5-VL-7B-Instruct-private" ;;
        *) echo "$1" ;;
    esac
}

# Repo adapter mặc định theo model (3b -> ...-3b-..., còn lại -> ...-7b-...).
default_hub_for_model() {
    case "$(echo "$1" | tr '[:upper:]' '[:lower:]')" in
        *3b*) echo "tranhuy67896262/qwen25vl-3b-vi-hwr-lora" ;;
        *) echo "tranhuy67896262/qwen25vl-7b-vi-hwr-lora" ;;
    esac
}

# Model suy từ tên repo adapter (ngược với hàm trên; rỗng nếu không rõ).
model_for_hub_repo() {
    case "$(echo "$1" | tr '[:upper:]' '[:lower:]')" in
        *3b*) echo "tranhuy67896262/Qwen2.5-VL-3B-Instruct-private" ;;
        *7b*) echo "tranhuy67896262/Qwen2.5-VL-7B-Instruct-private" ;;
        *) echo "" ;;
    esac
}

# Username của token (rỗng nếu không đọc được). Dùng để suy owner khi repo thiếu owner.
hf_api_user() {
    local token="$1" body="" user=""
    command -v curl >/dev/null 2>&1 || return 0
    body="$(curl --silent --max-time 15 \
        -H "Authorization: Bearer $token" \
        https://huggingface.co/api/whoami-v2 2>/dev/null || true)"
    user="$(echo "$body" | grep -o '"name":"[^"]*"' | head -n 1 | cut -d'"' -f4)"
    echo "$user"
    return 0
}

# Mốc stage-* cao nhất của repo Hub: echo "K rew_name" (K = số nghìn mẫu, vd "10 stage-10k-v2"),
# hoặc "NONE" (repo chưa có/chưa có nhánh stage), hoặc "ERR:<http|NET>".
latest_hub_stage() {
    local repo="$1" token="$2" code="" names="" line="" name="" knum="" maxk=-1 best=""
    code="$(curl --silent --max-time 20 -o /dev/null -w '%{http_code}' \
        -H "Authorization: Bearer $token" \
        "https://huggingface.co/api/models/${repo}/refs" 2>/dev/null || echo 000)"
    case "$code" in
        200) ;;
        404) echo "NONE"; return 0 ;;
        000) echo "ERR:NET"; return 0 ;;
        *) echo "ERR:$code"; return 0 ;;
    esac
    names="$(curl --silent --max-time 20 \
        -H "Authorization: Bearer $token" \
        "https://huggingface.co/api/models/${repo}/refs" 2>/dev/null \
        | grep -o '"name":"stage-[^"]*"' || true)"
    for line in $names; do
        name="${line#\"name\":\"stage-}"
        name="${name%\"}"
        knum="${name%%k*}"
        case "$knum" in
            ''|*[!0-9]*) continue ;;
        esac
        if [ "$knum" -gt "$maxk" ]; then
            maxk="$knum"
            best="$name"
        fi
    done
    if [ "$maxk" -lt 0 ]; then
        echo "NONE"
    else
        echo "$maxk stage-$best"
    fi
    return 0
}

# Thoát 1 + hướng dẫn nếu thiếu HF_TOKEN (env hoặc .env.dev).
# $1 = gợi ý cú pháp đúng (vd "bash pipeline.sh smoke 3b hf_xxxxx").
require_hf_token() {
    if [[ -z "${HF_TOKEN:-}" ]] && ! grep -q '^HF_TOKEN' .env.dev 2>/dev/null; then
        echo "[ERR] Thieu HF_TOKEN cho gated dataset."
        echo "Dung: $1"
        echo "Hoac export HF_TOKEN / tao .env.dev truoc khi chay."
        exit 1
    fi
}

# Thu wheel flash-attn build san tren GitHub release (mjun0812) theo dung combo
# (python, torch, cuda) dang chay — Colab pull ve cai ~1-2 phut thay vi build 20p.
# Tra 0 neu cai + import duoc; tra 1 de caller fallback build source.
install_flash_attn_github() {
    local py="$1" pytag torch_minor cudatag fa_url
    [[ "$(uname -m)" == "x86_64" ]] || return 1
    pytag="cp$("$py" -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')" 2>/dev/null)" || return 1
    torch_minor="$("$py" -c "import torch; print('.'.join(torch.__version__.split('.')[:2]))" 2>/dev/null)" || return 1
    cudatag="cu$("$py" -c "import torch; print((torch.version.cuda or '').replace('.', ''))" 2>/dev/null)" || return 1
    # Bang wheel da verify (Linux x86_64, torch cu128):
    case "${pytag}-torch${torch_minor}-${cudatag}" in
        cp313-torch2.8-cu128)
            fa_url="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.12/flash_attn-2.8.3%2Bcu128torch2.8-cp313-cp313-linux_x86_64.whl" ;;
        cp313-torch2.9-cu128)
            fa_url="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.15/flash_attn-2.8.3%2Bcu128torch2.9-cp313-cp313-linux_x86_64.whl" ;;
        *) return 1 ;;
    esac
    echo "Cai flash-attn tu GitHub release (${pytag}/torch${torch_minor}/${cudatag})..."
    if uv pip install --python "$py" "$fa_url" 2>/dev/null \
            && "$py" -c "import flash_attn" 2>/dev/null; then
        return 0
    fi
    return 1
}

# Cài uv + torch-CUDA + flash-attn (best-effort) + requirements nếu thiếu.
# Đặt biến toàn cục PYTHON. Gọi sau khi caller đã cd về ROOT.
setup_gpu_env() {
    export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
    # Colab: VM xóa mỗi phiên — mount Drive trước thì giữ cache uv + HF trên
    # Drive, phiên sau khỏi tải lại. Không Drive thì cache local như cũ.
    if [ -z "${HF_HOME:-}" ] && [ -d "/content/drive/MyDrive" ]; then
        HF_HOME="/content/drive/MyDrive/vi_ocr_handwritten/.hf_cache"
    fi
    export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
    if [ -z "${UV_CACHE_DIR:-}" ] && [ -d "/content/drive/MyDrive" ]; then
        UV_CACHE_DIR="/content/drive/MyDrive/vi_ocr_handwritten/.cache/uv"
    fi
    export UV_CACHE_DIR="${UV_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/uv}"

    if ! command -v uv >/dev/null 2>&1; then
        echo "Dang cai uv..."
        python3 -m pip install -q uv
    fi

    if [ -x ".venv/bin/python" ]; then
        PYTHON=".venv/bin/python"
    elif [ -x ".venv/Scripts/python.exe" ]; then
        PYTHON=".venv/Scripts/python.exe"
    else
        echo "Tao .venv bang uv..."
        uv venv .venv
        PYTHON=".venv/bin/python"
    fi
    echo "Python: $("$PYTHON" --version 2>/dev/null || echo "khong xac dinh")"

    # torch/torchvision bản CUDA (PyPI mặc định là bản CPU)
    if ! "$PYTHON" -c "import torch, torchvision; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "Dang cai torch + torchvision ban CUDA..."
        uv pip install --python "$PYTHON" torch torchvision --index-url https://download.pytorch.org/whl/cu128
    fi

    # flash-attn best-effort (tăng tốc A100/H100, thiếu thì fallback sdpa).
    # Thu tu: wheel chi dinh (FLASH_ATTN_WHEEL_URL) -> wheel GitHub release
    # tu dong theo combo may (~1-2p) -> build source --no-build-isolation
    # (build isolation che mat torch; MAX_JOBS chong OOM RAM).
    if ! "$PYTHON" -c "import flash_attn" 2>/dev/null; then
        _FLASH_OK=""
        if [[ -n "${FLASH_ATTN_WHEEL_URL:-}" ]]; then
            echo "Cai flash-attn tu wheel chi dinh..."
            if uv pip install --python "$PYTHON" "$FLASH_ATTN_WHEEL_URL" 2>/dev/null \
                    && "$PYTHON" -c "import flash_attn" 2>/dev/null; then
                _FLASH_OK=1
            fi
        fi
        if [[ -z "$_FLASH_OK" ]] && install_flash_attn_github "$PYTHON"; then
            _FLASH_OK=1
        fi
        if [[ -z "$_FLASH_OK" ]]; then
            echo "[WARN] Khong co wheel khop -> build source (10-20p)..."
            MAX_JOBS=4 uv pip install --python "$PYTHON" flash-attn --no-build-isolation 2>/dev/null \
                || echo "[WARN] Bo qua flash-attn."
        fi
    fi

    # requirements (transformers, peft, bitsandbytes, ...)
    if ! "$PYTHON" -c "import transformers, peft, bitsandbytes, torchvision" 2>/dev/null; then
        echo "Dang cai requirements..."
        uv pip install --python "$PYTHON" -r requirements.txt
    fi

    echo "GPU: $("$PYTHON" -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo 'CPU')"
}

# Cài Ollama CLI trên Colab/Linux (apt + install.sh) rồi trả 0; lỗi trả 1.
# Không hỗ trợ Windows/macOS (cài tay tại https://ollama.com/download).
install_ollama() {
    if command -v ollama >/dev/null 2>&1; then
        return 0
    fi
    case "$(uname -s)" in
        Linux) ;;
        *) echo "[ERR] Tu cai Ollama chi ho tro Linux/Colab."; return 1 ;;
    esac
    if ! command -v curl >/dev/null 2>&1; then
        echo "[ERR] Thieu curl — khong cai duoc Ollama."
        return 1
    fi
    echo "Cai Ollama (apt update + zstd/curl + install.sh, vai phut)..."
    sudo apt-get update || return 1
    sudo apt-get install -y zstd curl || return 1
    curl -fsSL https://ollama.com/install.sh | sh || return 1
    command -v ollama >/dev/null 2>&1 || {
        echo "[ERR] Cai xong ma khong thay ollama."
        return 1
    }
    echo "Da cai Ollama CLI."
}

# Bảo đảm Ollama CLI + server đang chạy (tự `ollama serve` nền nếu OLLAMA_AUTO_START=1;
# tự cài CLI nếu OLLAMA_AUTO_INSTALL=1, vd trên Colab mới).
ensure_ollama_serve() {
    if ! command -v ollama >/dev/null 2>&1; then
        if [[ "${OLLAMA_AUTO_INSTALL:-0}" == "1" ]]; then
            install_ollama || exit 1
        else
            echo "[ERR] Chua cai Ollama CLI (Colab: OLLAMA_AUTO_INSTALL=1 de tu cai)."
            exit 1
        fi
    fi
    if ollama list >/dev/null 2>&1; then
        return 0
    fi
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
}

# Đếm số mẫu có nhãn trong assets/labels.csv (cộng vào --max-samples khi MERGE=1).
count_local_samples() {
    "${PYTHON:-python}" -c 'import csv
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

# Kiểm tra token qua API Hub: còn sống + đọc được DATASET/MODEL + owner HUB_REPO.
# $1 = token. Yêu cầu DATASET/MODEL/HUB_REPO đã đặt (rỗng thì bỏ qua mục đó).
# Echo đúng 1 dòng: OK:<user> | BAD:<lý do> | NET-ERR (mất mạng/thiếu curl).
check_hf_token_remote() {
    local token="$1" body="" code="" user="" kind="" repo="" repo_kind="" url="" owner=""
    if ! command -v curl >/dev/null 2>&1; then
        echo "NET-ERR"
        return 0
    fi
    body="$(curl --silent --show-error --max-time 15 -w '\n%{http_code}' \
        -H "Authorization: Bearer $token" \
        https://huggingface.co/api/whoami-v2 2>/dev/null || true)"
    code="$(echo "$body" | tail -n 1 | tr -d '[:space:]')"
    case "$code" in
        200) ;;
        401|403) echo "BAD:token bị Hub từ chối (http $code)"; return 0 ;;
        000|"") echo "NET-ERR"; return 0 ;;
        *) echo "BAD:whoami lạ (http $code)"; return 0 ;;
    esac
    user="$(echo "$body" | grep -o '"name":"[^"]*"' | head -n 1 | cut -d'"' -f4)"
    if [[ -z "$user" ]]; then echo "BAD:không đọc được user"; return 0; fi
    # Quyền đọc dataset + model (repo private/gated; path local như data/combined thì bỏ qua).
    for repo_kind in "datasets:${DATASET:-}" "models:${MODEL:-}"; do
        kind="${repo_kind%%:*}"; repo="${repo_kind#*:}"
        [[ -n "$repo" ]] || continue
        case "$repo" in *"/"*) ;; *) continue ;; esac
        url="https://huggingface.co/api/${kind}/${repo}"
        code="$(curl --silent --max-time 15 -o /dev/null -w '%{http_code}' \
            -H "Authorization: Bearer $token" "$url" 2>/dev/null || echo 000)"
        case "$code" in
            200) ;;
            000) echo "NET-ERR"; return 0 ;;
            *) echo "BAD:không đọc được ${kind}/${repo} (http $code)"; return 0 ;;
        esac
    done
    # Push cần quyền Write: owner repo đích thường phải trùng user của token.
    owner="${HUB_REPO:-}"
    case "$owner" in
        *"/"*) owner="${owner%%/*}" ;;
        *) owner="" ;;
    esac
    if [[ -n "$owner" && "$owner" != "$user" ]]; then
        echo "[WARN] HUB_REPO thuộc '$owner' nhưng token là user '$user' — push cần quyền Write." >&2
    fi
    echo "OK:$user"
}

# Lưu token vào .env.dev (thay dòng cũ hoặc append), cho lần chạy sau.
save_hf_token_to_dotenv() {
    if [[ -f .env.dev ]] && grep -q '^HF_TOKEN' .env.dev 2>/dev/null; then
        sed -i 's|^HF_TOKEN.*|HF_TOKEN = '"$1"'|' .env.dev
    else
        printf 'HF_TOKEN = %s\n' "$1" >> .env.dev
    fi
    echo "Đã lưu HF_TOKEN vào .env.dev"
}

# Bảo đảm có HF_TOKEN hợp lệ trước khi train: env > .env.dev > hỏi user nhập.
# Kiểm tra token còn sống + đọc được DATASET/MODEL (repo train/push cần).
# Token nhập tay (ẩn, tối đa 3 lần) được kiểm tra rồi lưu vào .env.dev.
# Không có TTY và thiếu token -> thoát 1 ngay, không treo.
# Gọi sau khi caller đã cd về ROOT và đặt DATASET/MODEL/HUB_REPO.
ensure_hf_token() {
    # 1. Nạp từ .env.dev nếu env trống (dòng trống coi như chưa có — tránh 404 dataset).
    if [[ -z "${HF_TOKEN:-}" && -f .env.dev ]]; then
        HF_TOKEN="$(grep '^HF_TOKEN' .env.dev 2>/dev/null | tail -n 1 | cut -d= -f2- | tr -d '[:space:]' || true)"
    fi
    export HF_TOKEN="${HF_TOKEN:-}"
    # 2. Vòng kiểm tra + hỏi (tối đa 3 lần).
    local attempt=0 need_save=0 verdict=""
    while [ "$attempt" -lt 3 ]; do
        if [[ -n "$HF_TOKEN" ]]; then
            verdict="$(check_hf_token_remote "$HF_TOKEN")"
            case "$verdict" in
                OK:*)
                    export HF_USER="${verdict#OK:}"
                    if [ "$need_save" = "1" ]; then
                        save_hf_token_to_dotenv "$HF_TOKEN"
                    fi
                    echo "HF_TOKEN hợp lệ (user: $HF_USER)."
                    return 0 ;;
                NET-ERR)
                    echo "[WARN] Không nối được huggingface.co — bỏ qua kiểm tra, dùng token hiện có."
                    return 0 ;;
                *) echo "[WARN] Token không dùng được ($verdict)."; HF_TOKEN="" ;;
            esac
        fi
        attempt=$((attempt + 1))
        if [ "$attempt" -ge 3 ]; then break; fi
        echo "Cần HF_TOKEN đọc được dataset/model private và Write lên repo đích."
        echo "Lấy token tại: https://huggingface.co/settings/tokens"
        local input=""
        read -rsp "Dán HF_TOKEN (lần $attempt/3, Enter để bỏ qua): " input || true
        echo
        input="$(echo "$input" | tr -d '[:space:]' || true)"
        if [[ -z "$input" ]]; then
            echo "[ERR] Bỏ qua nhập token."
            break
        fi
        HF_TOKEN="$input"
        need_save=1
    done
    echo "[ERR] Chưa có HF_TOKEN hợp lệ — export HF_TOKEN hoặc ghi vào .env.dev rồi chạy lại."
    exit 1
}
