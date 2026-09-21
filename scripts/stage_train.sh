#!/usr/bin/env bash
# Train staged sequential, lát mới-only (gộp stage_train.sh 1-mốc + stage_train_loop.sh cũ).
# Dùng uv như run_train.sh: tự dựng .venv + cài deps (1 lần) rồi chạy train.py.
#
# Cách dùng (flow gọn, 1 lệnh duy nhất):
#   ./scripts/stage_train.sh [hf_token] <model> <repo> [count] [save_steps]
#   <model>: qwenvl-3b | qwenvl-7b (hoặc 3b/7b/tên đầy đủ)
#   <repo>: owner/repo (thiếu owner -> lấy user của token); [count] mặc định 5000,
#     [save_steps] mặc định 50. Token truyền tay được lưu vào .env.dev.
#   start/init TỰ DÒ từ nhánh stage-* cao nhất trên repo (repo mới -> train trắng từ 0).
#   vd: ./scripts/stage_train.sh hf_xxx qwenvl-3b owner/repo
#   vd: ./scripts/stage_train.sh qwenvl-3b owner/repo 10k 50   (token từ env/.env.dev)
# Flow cũ (tương thích): <model> <start> <end> <init_rev> [step] [save_steps] [hub_repo]
#   1 mốc duy nhất (end-start <= step) -> chạy nền nohup, log train-<run>.log
#   nhiều mốc -> chạy foreground nối tiếp, bắt buộc tmux (rớt SSH không chết chuỗi)
#   vd 3B 10k đầu trên A100 80GB (mặc định bf16 batch 8, eff 32):
#     ./scripts/stage_train.sh 3b 0 10k none
#
# Mapping từ lệnh cũ sang flow gọn (cùng kết quả khi repo đã tới đúng mốc init):
#   stage_train.sh 7b 5k 15k stage-5k 10k 50 -> stage_train.sh <token> qwenvl-7b <repo> 10k 50
#   (repo đã tới stage-5k; count 10k = lát 5k->15k; save_steps 50)
#   3B 10k đầu trên A100 80GB (mặc định bf16 batch 8, eff 32):
#     ./scripts/stage_train.sh <token> qwenvl-3b <repo> 10k
#
# Env: BATCH_SIZE (mặc định 8 cho cả 3b/7b), GRAD_ACCUM (=4), LR (=2e-5),
#   USE_4BIT (auto = 0 bf16 --no-4bit; GPU nhỏ ép QLoRA bằng USE_4BIT=1), EPOCHS (=1),
#   DATASET (mặc định mirror v2-local), HUB_REPO/INIT_REPO,
#   LORA_R/LORA_ALPHA (chỉ khi train trắng), SUFFIX (giữ -v2 khi nối chuỗi -v2 cũ),
#   FORCE_START (đổi dataset giữa chừng: ép start, init vẫn lấy mốc mới nhất),
#   DRY_RUN=1 (in kế hoạch, không train).
# Xem tiến độ: tail -n 5 train-<run>.log (không tail -f).
set -euo pipefail

LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/lib" && pwd)"
source "$LIB_DIR/common.sh"

# Token truyền tay (arg đầu, như run_train.sh): lưu vào .env.dev rồi xử tiếp.
if [[ "${1:-}" == hf_* ]]; then
    printf 'HF_TOKEN = %s\n' "$1" > .env.dev
    echo "Đã ghi HF_TOKEN vào .env.dev"
    shift
fi
# Phân biệt flow gọn ([model] <repo> [count] [save]) với flow cũ
# (<model> <start> <end> <init> ...): flow cũ thì $2 luôn là số mẫu,
# còn lại (repo có/không owner) là flow gọn.
NEW_FLOW=0
if [[ ! "${2:-}" =~ ^[0-9]+[kK]?$ ]]; then
    case "$(echo "${1:-}" | tr '[:upper:]' '[:lower:]')" in
        3b|7b|qwenvl-3b|qwenvl-7b|qwen25vl-3b|qwen25vl-7b|qwen2.5-vl-3b|qwen2.5-vl-7b|*qwen*)
            NEW_FLOW=1 ;;
    esac
fi

REPO_ARG=""
if [ "$NEW_FLOW" = "1" ]; then
    MODEL_ARG="$1"
    REPO_ARG="$2"
    COUNT=$(to_samples "${3:-5000}")
    SAVE_STEPS="${4:-50}"
    case "$COUNT/$SAVE_STEPS" in
        ''|*[!0-9/]*) echo "[ERR] count/save_steps phải là số (count hỗ trợ k): $COUNT/$SAVE_STEPS"; exit 1 ;;
    esac
    if [ "$COUNT" -le 0 ]; then echo "[ERR] count phải > 0"; exit 1; fi
    MODEL=$(resolve_model "$MODEL_ARG")
    case "$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')" in
        *3b*|*7b*) ;;
        *) echo "[ERR] model không rõ: $MODEL_ARG (dùng qwenvl-3b|qwenvl-7b)"; exit 1 ;;
    esac
    # Cảnh báo sớm nếu tag size trong tên repo lệch với model được chọn.
    REPO_MODEL="$(model_for_hub_repo "$REPO_ARG")"
    if [[ -n "$REPO_MODEL" && "$REPO_MODEL" != "$MODEL" ]]; then
        echo "[WARN] repo gợi ý $REPO_MODEL nhưng model chọn là $MODEL — kiểm tra lại."
    fi
    HUB_REPO="$REPO_ARG"
else
    MODEL_ARG=${1:?cần model (flow gọn: ./scripts/stage_train.sh [token] qwenvl-3b|qwenvl-7b <repo> [count] [save]; flow cũ: <model> <start> <end> <init> ...)}
    START=$(to_samples "${2:?cần start (vd 0, 5k)}")
    END=$(to_samples "${3:?cần end (vd 10k, 59247)}")
    INIT_REV=${4:?cần init revision (vd stage-15k, hoặc none để train trắng từ đầu)}
    STEP=$(to_samples "${5:-5000}")
    SAVE_STEPS=${6:-100}
    HUB_REPO_ARG=${7:-}

    case "$START/$END/$STEP" in
        ''|*[!0-9/]*) echo "[ERR] start/end/step phải là số mẫu (hỗ trợ hậu tố k): $START/$END/$STEP"; exit 1 ;;
    esac
    if [ "$END" -le "$START" ]; then echo "[ERR] end ($END) phải lớn hơn start ($START)"; exit 1; fi
    if [ "$STEP" -le 0 ]; then echo "[ERR] step phải > 0"; exit 1; fi

    MODEL=$(resolve_model "$MODEL_ARG")
fi
DATASET="${DATASET:-tranhuy67896262/Viet-Handwriting-OCR-v2-local}"
# Batch mặc định 8 cho cả 3b/7b (A100 80GB: eff 8x4=32). Ghi đè qua env.
DEFAULT_BATCH=8
BATCH_SIZE="${BATCH_SIZE:-$DEFAULT_BATCH}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-2e-5}"
LORA_R="${LORA_R:-}"
LORA_ALPHA="${LORA_ALPHA:-}"
# USE_4BIT=auto (mặc định): staged luôn bf16 --no-4bit (3b và 7b);
# cần QLoRA cho GPU nhỏ thì export USE_4BIT=1.
USE_4BIT="${USE_4BIT:-auto}"
if [ "$USE_4BIT" = "auto" ]; then
    USE_4BIT="0"
fi
PRECISION_FLAG="--no-4bit"
if [ "$USE_4BIT" = "1" ]; then
    PRECISION_FLAG=""
fi
# Repo đích: arg [hub_repo] > env HUB_REPO > default theo model.
HUB_REPO="${HUB_REPO_ARG:-${HUB_REPO:-$(default_hub_for_model "$MODEL")}}"
# Repo nguồn adapter: mặc định = HUB_REPO; nối weight repo cũ sang repo mới thì
# export INIT_REPO=owner/repo-cu.
INIT_REPO="${INIT_REPO:-$HUB_REPO}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Token: env > .env.dev > hỏi user nhập (kiểm tra còn sống + đọc được
# dataset/model private; token nhập tay được lưu vào .env.dev).
ensure_hf_token

if [ "$NEW_FLOW" = "1" ]; then
    # Repo thiếu owner -> lấy user của token (vừa kiểm tra hợp lệ ở trên).
    case "$HUB_REPO" in
        */*) ;;
        *) HUB_REPO="${HF_USER:?không đọc được user của token để suy owner repo}/${HUB_REPO}" ;;
    esac
    if [ "$INIT_REPO" = "$REPO_ARG" ]; then INIT_REPO="$HUB_REPO"; fi
    # Tự dò tiến độ: nhánh stage-* cao nhất -> train tiếp; repo mới -> trắng từ 0.
    echo "Kiểm tra tiến độ trên Hub: $HUB_REPO ..."
    PROG="$(latest_hub_stage "$HUB_REPO" "$HF_TOKEN")"
    case "$PROG" in
        NONE)
            START=0; INIT_REV=none
            echo "Repo mới/chưa có mốc stage -> train trắng từ 0." ;;
        ERR:*)
            echo "[ERR] Không đọc được repo $HUB_REPO (${PROG#ERR:}) — kiểm tra tên repo/quyền token."
            exit 1 ;;
        *)
            START=$(( ${PROG%% *} * 1000 )); INIT_REV="${PROG#* }"
            echo "Repo đã tới $INIT_REV -> train tiếp $COUNT mẫu từ $START." ;;
    esac
    # Đổi dataset giữa chừng (vd train xong bộ line, chuyển sang bộ đoạn văn trên
    # cùng 1 adapter): ép START về đầu dataset mới, INIT vẫn lấy weight mốc mới nhất.
    if [ -n "${FORCE_START:-}" ]; then
        START=$(to_samples "$FORCE_START")
        case "$START" in
            ''|*[!0-9]*) echo "[ERR] FORCE_START phải là số mẫu (hỗ trợ k): $FORCE_START"; exit 1 ;;
        esac
        echo "FORCE_START=$START (bỏ qua start tự dò; init giữ $INIT_REV)."
    fi
    END=$((START + COUNT)); STEP=$COUNT
fi

# Số epoch mỗi lát (mặc định 1; data nhỏ vd 1k mẫu thì EPOCHS=2-3 cho đủ step).
EPOCHS="${EPOCHS:-1}"
case "$EPOCHS" in
    ''|*[!0-9]*|0) echo "[ERR] EPOCHS phải là số nguyên > 0"; exit 1 ;;
esac

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[dry-run] bỏ qua setup_gpu_env + train.py thật."
else
    setup_gpu_env
fi

# Hậu tố tên nhánh: giữ -v2 xuyên suốt nếu nối chuỗi -v2 cũ.
# Chuỗi 7B hiện tại quy ước bắt đầu -v2 khi train trắng từ 0 trên repo default;
# model khác (vd 3B mới) thì tên sạch stage-Nk. Ghi đè tay bằng SUFFIX=-v2 / SUFFIX=.
DEFAULT_REPO="$(default_hub_for_model "$MODEL")"
SUFFIX="${SUFFIX:-}"
if [ -z "${SUFFIX:-}" ]; then
    case "$INIT_REV" in
        *-v2) SUFFIX="-v2" ;;
    esac
    case "$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')" in
        *7b*)
            if [ "$INIT_REV" = "none" ] && [ "$START" -eq 0 ] && [ "$HUB_REPO" = "$DEFAULT_REPO" ]; then
                SUFFIX="-v2"
            fi
            ;;
    esac
fi

# LORA_* chỉ truyền khi train trắng (nối adapter cũ thì r/alpha lấy theo adapter cũ).
LORA_FLAGS=""
if [ -n "$LORA_R" ]; then
    LORA_FLAGS="$LORA_FLAGS --lora-r $LORA_R"
fi
if [ -n "$LORA_ALPHA" ]; then
    LORA_FLAGS="$LORA_FLAGS --lora-alpha $LORA_ALPHA"
fi

# Chạy 1 mốc: $1=start $2=count $3=init_adapter $4=hub_rev $5=run $6=background(0/1).
# DRY_RUN=1: chỉ in kế hoạch, không train (dùng để kiểm tra start/init/rev).
run_stage() {
    local S="$1" COUNT="$2" STAGE_INIT="$3" HUB_REV="$4" RUN="$5" BG="$6"
    if [ "${DRY_RUN:-0}" = "1" ]; then
        local prec="qlora-4bit"
        if [ -n "$PRECISION_FLAG" ]; then prec="bf16-lora"; fi
        echo "[dry-run] model=$MODEL repo=$HUB_REPO start=$S count=$COUNT epochs=$EPOCHS init=$STAGE_INIT rev=$HUB_REV run=$RUN batch=$BATCH_SIZE accum=$GRAD_ACCUM precision=$prec"
        return 0
    fi
    # Tự resume nếu mốc này đã có checkpoint local (chết giữa chừng).
    # Dùng chuỗi có thay mảng (tránh unbound variable trên bash cũ với set -u).
    # shellcheck disable=SC2086
    local EXTRA_FLAGS="--auto-progress"
    if ls -d "models/checkpoints/${RUN}"/checkpoint-* >/dev/null 2>&1; then
        echo "Thấy checkpoint local -> resume đúng step cũ (--resume)."
        EXTRA_FLAGS="$EXTRA_FLAGS --resume"
    fi
    if [ "$BG" = "1" ]; then
        local LOG="train-${RUN}.log"
        # shellcheck disable=SC2086
        nohup "$PYTHON" scripts/train.py \
            --dataset "$DATASET" --model "$MODEL" \
            --start-samples "$S" --max-samples "$COUNT" $PRECISION_FLAG --batch-size "$BATCH_SIZE" \
            --gradient-accumulation-steps "$GRAD_ACCUM" --lr "$LR" --epochs "$EPOCHS" \
            --init-adapter "$STAGE_INIT" \
            --push --push-every-save --save-steps "$SAVE_STEPS" \
            --hub-repo "$HUB_REPO" --hub-revision "$HUB_REV" --run-name "$RUN" \
            $EXTRA_FLAGS $LORA_FLAGS \
            > "$LOG" 2>&1 &
        echo "Đang chạy ${RUN} (PID $!), log: ${LOG}"
    else
        # shellcheck disable=SC2086
        "$PYTHON" scripts/train.py \
            --dataset "$DATASET" --model "$MODEL" \
            --start-samples "$S" --max-samples "$COUNT" $PRECISION_FLAG --batch-size "$BATCH_SIZE" \
            --gradient-accumulation-steps "$GRAD_ACCUM" --lr "$LR" --epochs "$EPOCHS" \
            --init-adapter "$STAGE_INIT" \
            --push --push-every-save --save-steps "$SAVE_STEPS" \
            --hub-repo "$HUB_REPO" --hub-revision "$HUB_REV" --run-name "$RUN" \
            $EXTRA_FLAGS $LORA_FLAGS
    fi
}

# Mốc đầu chuỗi mới (INIT_REV=none): train.py hiểu --init-adapter none là train trắng.
resolve_init_adapter() {
    if [ "$1" = "none" ]; then
        echo "none"
    else
        echo "${INIT_REPO}@${1}"
    fi
}

S="$START"
PREV_REV="$INIT_REV"
# Đếm số mốc để quyết định nền/foreground.
STAGES=0
T="$S"
while [ "$T" -lt "$END" ]; do
    STAGES=$((STAGES + 1))
    T=$((T + STEP))
done

if [ "$STAGES" -eq 1 ]; then
    # 1 mốc duy nhất: chạy nền như stage_train.sh cũ.
    K=$((END / 1000))
    HUB_REV="stage-${K}k${SUFFIX}"
    RUN="run-${K}k${SUFFIX}"
    if [ "$PREV_REV" = "none" ]; then
        echo "Mốc đơn trắng từ đầu -> train adapter mới (--init-adapter none)."
    fi
    echo "===== Mốc ${S} -> ${END} (init ${PREV_REV} -> ${HUB_REV}) [background] ====="
    run_stage "$S" "$((END - S))" "$(resolve_init_adapter "$PREV_REV")" "$HUB_REV" "$RUN" 1
else
    # Nhiều mốc: foreground nối tiếp như stage_train_loop.sh cũ, bắt buộc tmux.
    if [ -n "${TMUX:-}" ]; then
        echo "Đang trong tmux - tốt (rớt SSH không chết chuỗi train)."
    else
        echo "[WARN] Không thấy tmux - nên chạy trong tmux (tmux new -s train)."
    fi
    if [ "$PREV_REV" = "none" ]; then
        echo "Mốc đầu chuỗi mới -> train adapter mới từ đầu (--init-adapter none)."
    fi
    while [ "$S" -lt "$END" ]; do
        E=$((S + STEP))
        if [ "$E" -gt "$END" ]; then
            E="$END"
        fi
        # CHÚ Ý: --max-samples là SỐ LƯỢNG (count), không phải end-offset
        # (dataset.py: end = start + max). Nên truyền count = E - S.
        COUNT=$((E - S))
        K=$((E / 1000))
        HUB_REV="stage-${K}k${SUFFIX}"
        RUN="run-${K}k${SUFFIX}"
        echo "===== Mốc ${S} -> ${E} (init ${PREV_REV} -> ${HUB_REV}) ====="
        run_stage "$S" "$COUNT" "$(resolve_init_adapter "$PREV_REV")" "$HUB_REV" "$RUN" 0
        echo "===== Xong mốc ${HUB_REV} - nhớ eval trước khi mốc sau chạy tiếp ====="
        PREV_REV="$HUB_REV"
        S="$E"
    done
    echo "Xong chuỗi ${START} -> ${END}."
fi
