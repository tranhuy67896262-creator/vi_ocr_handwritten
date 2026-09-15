#!/usr/bin/env bash
# Staged training cho OCR chữ viết tay tiếng Việt: chia data thành 5 mốc CUMULATIVE
# (mặc định 5k -> 10k -> 20k -> 40k -> 59k, lấy từ Configs.STAGE_SAMPLES).
#
# Mỗi mốc:
#   1. train TIẾP từ adapter mốc trước (--init-adapter), lấy đúng N mẫu đã shuffle cố định
#   2. eval CER/WER trên CÙNG một tập eval cố định (so sánh được giữa các mốc)
#   3. push adapter lên Hub dưới revision riêng: stage-<N> (không đè 'main')
#
# Cách dùng:
#   bash scripts/train_stages.sh --hub-repo <owner>/<repo>
#   bash scripts/train_stages.sh --hub-repo <owner>/<repo> --eval-samples 300 --no-spell-fix
#   START_STAGE=3 bash scripts/train_stages.sh --hub-repo <owner>/<repo>   # chạy tiếp từ mốc 3
#
# Biến môi trường:
#   HF_TOKEN        token Write (hoặc dùng --token)
#   STAGES          danh sách mốc, vd "5000 10000 20000 59247" (mặc định: từ configs.py)
#   EVAL_SAMPLES    số mẫu eval mỗi mốc (mặc định 300)
#   SPELL_FIX       1 = eval kèm spell-fix (mặc định 1)
#   START_STAGE     bỏ qua các mốc trước (dùng khi chạy tiếp sau khi đứt)
#   EXTRA_TRAIN_ARGS  cờ thêm cho train.py, vd "--no-4bit --batch-size 2"
#
# Kết quả từng mốc ghi vào models/stage_results.tsv (đường CER-vs-lượng-data).
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

HUB_REPO=""
TOKEN="${HF_TOKEN:-}"
EVAL_SAMPLES="${EVAL_SAMPLES:-300}"
SPELL_FIX="${SPELL_FIX:-1}"
START_STAGE="${START_STAGE:-1}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

usage() {
  sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hub-repo)      HUB_REPO="$2"; shift 2 ;;
    --token)         TOKEN="$2"; shift 2 ;;
    --eval-samples)  EVAL_SAMPLES="$2"; shift 2 ;;
    --start-stage)   START_STAGE="$2"; shift 2 ;;
    --no-spell-fix)  SPELL_FIX=0; shift ;;
    -h|--help)       usage; exit 0 ;;
    *) echo "[ERR] Tham số lạ: $1"; usage; exit 2 ;;
  esac
done

if [[ -z "$HUB_REPO" || "$HUB_REPO" != */* ]]; then
  echo "[ERR] Cần --hub-repo <owner>/<repo> (owner/repo) — nơi push từng mốc."
  usage
  exit 2
fi
if [[ -n "$TOKEN" ]]; then
  export HF_TOKEN="$TOKEN"
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[WARN] Không thấy HF_TOKEN — bước push sẽ lỗi. Đặt env HF_TOKEN hoặc truyền --token."
fi

PY="python"
if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
fi

# Danh sách mốc: ưu tiên STAGES (env), nếu không lấy từ nguồn sự thật configs.py.
MILEPOSTS=()
if [[ -n "${STAGES:-}" ]]; then
  read -r -a MILEPOSTS <<< "$STAGES"
else
  while IFS= read -r _n; do
    [[ -n "$_n" ]] && MILEPOSTS+=("$_n")
  done < <("$PY" -c "from configs.configs import Configs; print('\n'.join(str(n) for n in Configs.STAGE_SAMPLES))")
fi
if [[ "${#MILEPOSTS[@]}" -eq 0 ]]; then
  echo "[ERR] Không có mốc nào (kiểm tra Configs.STAGE_SAMPLES hoặc env STAGES)."
  exit 2
fi

ADAPTER_DIR="$("$PY" -c "from configs.configs import Configs; print(Configs().ADAPTER_DIR)")"
mkdir -p models
RESULTS="models/stage_results.tsv"
if [[ ! -f "$RESULTS" ]]; then
  printf 'stage\tsamples\trevision\tcer\twer\tcer_spellfix\n' > "$RESULTS"
fi

TOTAL="${#MILEPOSTS[@]}"
echo "Repo Hub : $HUB_REPO"
echo "Adapter  : $ADAPTER_DIR"
echo "Mốc      : ${MILEPOSTS[*]}"
echo "Eval     : $EVAL_SAMPLES mẫu cố định | spell-fix=$SPELL_FIX"
echo

prev_rev=""
for i in "${!MILEPOSTS[@]}"; do
  idx=$((i + 1))
  N="${MILEPOSTS[$i]}"
  rev="stage-${N}"

  if (( idx < START_STAGE )); then
    echo "[SKIP] Mốc $idx/$TOTAL ($N mẫu) — giả định đã chạy."
    prev_rev="$rev"
    continue
  fi

  echo "=========== Mốc $idx/$TOTAL: $N mẫu -> revision '$rev' ==========="

  init_args=()
  if [[ -n "$prev_rev" ]]; then
    init_args=(--init-adapter "$HUB_REPO" --adapter-revision "$prev_rev")
    echo "Train tiếp từ: $HUB_REPO@$prev_rev"
  else
    echo "Mốc đầu — train adapter mới từ base model."
  fi

  # shellcheck disable=SC2086  # EXTRA_TRAIN_ARGS cố ý tách theo khoảng trắng
  "$PY" scripts/train.py \
    --max-samples "$N" \
    --run-name "stage-$N" \
    --push --hub-repo "$HUB_REPO" --hub-revision "$rev" \
    $EXTRA_TRAIN_ARGS \
    ${init_args[@]+"${init_args[@]}"}

  # --- eval trên cùng một tập cố định (shuffle seed trong configs.py) ---
  spell_arg=""
  if [[ "$SPELL_FIX" == "1" ]]; then
    spell_arg="--spell-fix"
  fi
  eval_log="models/stage-$N-eval.log"
  echo "--- Eval mốc $N ($EVAL_SAMPLES mẫu) ---"
  set +e
  # shellcheck disable=SC2086  # spell_arg có thể rỗng
  "$PY" scripts/eval_ocr.py --num-test "$EVAL_SAMPLES" --adapter "$ADAPTER_DIR" \
    $spell_arg 2>&1 | tee "$eval_log"
  set -e

  cer="$(sed -n 's/^CER: //p' "$eval_log" | tail -1)"
  wer="$(sed -n 's/^WER: //p' "$eval_log" | tail -1)"
  cer_fix="$(sed -n 's/^CER (spell-fix): //p' "$eval_log" | tail -1)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$idx" "$N" "$rev" "${cer:-NA}" "${wer:-NA}" "${cer_fix:-NA}" >> "$RESULTS"
  echo "[OK] Mốc $idx: CER=${cer:-NA} WER=${wer:-NA} CER(spell-fix)=${cer_fix:-NA}"
  echo

  prev_rev="$rev"
done

echo "=========== Xong cả $TOTAL mốc ==========="
column -t -s $'\t' "$RESULTS" 2>/dev/null || cat "$RESULTS"
echo
echo "Adapter mốc cuối: $ADAPTER_DIR"
echo "Push cuối lên   : $HUB_REPO@$prev_rev"
