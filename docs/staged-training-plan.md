# Kế hoạch: Staged training 5 mốc (5k → 59k) + push Hub định kỳ

Trạng thái: **đã triển khai**. Tài liệu này là nguồn tham chiếu cho thay đổi code tương ứng.

## 1. Mục tiêu

- Train tăng dần theo 5 mốc dữ liệu cumulative: **5k → 10k → 20k → 40k → 59k**.
- Mỗi mốc: train tiếp từ adapter mốc trước (không train lại từ 0), **đo CER/WER** trên cùng một
  tập eval cố định, rồi **push lên HuggingFace Hub dưới một revision riêng** (`stage-5k`, ...).
- Kết quả cuối: (a) adapter production = mốc 59k, (b) đường CER-vs-lượng-data để biết mốc nào
  đã bão hoà, (c) giữ được cả 5 phiên bản adapter trên Hub để so sánh.

## 2. Vấn đề của code trước khi sửa

| # | Vấn đề | Chỗ | Ảnh hưởng |
|---|---|---|---|
| 1 | `ds.select(range(N))` lấy N dòng **đầu** theo thứ tự parquet, không shuffle | `dataset.py` | 5 mốc lệch phân bố người viết → kết luận sai |
| 2 | `build_lora_model()` luôn `get_peft_model()` tạo adapter mới | `load.py` | Mốc 2 train lại từ 0 rồi **ghi đè** adapter mốc 1 |
| 3 | `push_to_hub()` không có `revision` | `trainer.py` | 5 lần push đè lên `main`, chỉ giữ bản cuối |
| 4 | Eval chỉ chạy khi `max_samples is None` | `dataset.py` | Cả 5 mốc đều không có eval → không vẽ được đường cong |
| 5 | Không có push định kỳ | `trainer.py` | Chỉ push 1 lần ở cuối run |
| 6 | Không có script điều phối mốc | — | Phải gọi `train.py` tay 5 lần, dễ sai |

## 3. Thay đổi

### 3.1 `configs/configs.py`

| Thông số | Cũ | Mới | Lý do |
|---|---|---|---|
| `MAX_PIXELS` | 768·28² (602k) | **1280·28² (1.0M)** | Dataset là ảnh crop **dòng**; độ phân giải là nút cổ chai của dấu thanh |
| `MAX_SEQ_LEN` | 1024 | **1536** | Chừa chỗ cho 1280 image-token + text |
| `BATCH_SIZE` | 2 | **4** | Per-device chỉ để vừa VRAM (A100 40GB) |
| `GRADIENT_ACCUMULATION_STEPS` | 8 | **4** | Giữ effective batch = 16 → đủ ~8.4k step cho 5 mốc |
| `LORA_R / LORA_ALPHA` | 32 / 64 | **64 / 128** | 59k dòng đủ sức chứa r=64 |
| `LEARNING_RATE` | 2e-5 | **1e-4** | LoRA bf16 + không KL chịu được lr cao hơn |
| `KL_REGULARIZATION` | True | **False** | KL kéo phân phối đáp án về base model — chính cái đang đọc sai dấu. Cờ `--kl` để bật lại |
| `DATA_SHUFFLE_SEED` | — | 42 | Shuffle cố định trước khi cắt N → 5 mốc lồng nhau **và** đại diện |
| `EVAL_SAMPLES` | — | 300 | Số dòng eval cố định, lấy từ `TEST_SPLIT` |
| `EVAL_SHUFFLE_SEED` | — | 42 | Chọn eval subset tái lập được |
| `STAGE_SAMPLES` | — | (5k, 10k, 20k, 40k, 59247) | 5 mốc cumulative |

Ghi chú: `select(range(N))` sau `shuffle(seed)` vẫn cho tập **lồng nhau** (5k ⊂ 10k) vì
permutation là cố định — đúng ý đồ cumulative.

### 3.2 `src/datasets/dataset.py`

- `load_dataset_with_fallback(config, split=None)` — nhận split, mặc định `TRAIN_SPLIT`.
- `build_train_eval_datasets()`:
  - train: `shuffle(DATA_SHUFFLE_SEED).select(range(min(N, len)))`
  - eval: **luôn** lấy `EVAL_SAMPLES` dòng từ `TEST_SPLIT` (shuffle `EVAL_SHUFFLE_SEED`)
    → mọi mốc dùng đúng một thước đo. Nếu không có test split thì fallback `train_test_split(VAL_RATIO)`.

### 3.3 `src/modeling/load.py`

`build_lora_model(config, model, init_adapter=None, adapter_revision=None)`:

- có `init_adapter` → `PeftModel.from_pretrained(model, init_adapter, revision=..., is_trainable=True)`
  (nạp **trọng số** adapter cũ, LR/step reset — khác `--resume` là khôi phục cả optimizer).
- không có → giữ nguyên `get_peft_model()` như cũ.

Lưu ý: khi `--init-adapter`, `r`/`alpha`/`target_modules` lấy theo adapter cũ, **`--lora-r` bị bỏ qua**.

### 3.4 `scripts/train.py`

Cờ mới: `--init-adapter`, `--adapter-revision`, `--hub-revision`, `--max-pixels`,
`--eval-samples`, `--kl` (bật lại KL), `--run-name` (tách thư mục checkpoint theo mốc).

### 3.5 `src/train/trainer.py`

- `train(..., hub_revision=None, run_name=None, push_every_save=False)`.
- Checkpoint tách theo mốc: `models/checkpoints/<run_name>/` → `--resume` không lẫn giữa các mốc.
- Push kèm `revision=hub_revision` cho cả model lẫn processor.
- `PushOnSaveCallback` (chỉ khi `push_every_save=True`) đẩy mỗi lần `save_steps`.
- `training_metadata.json` thêm `run_name`, `init_adapter`, `hub_revision`.

### 3.6 `scripts/eval_ocr.py`

- `--adapter-revision` để eval đúng phiên bản adapter trên Hub.
- Tập eval lấy `shuffle(EVAL_SHUFFLE_SEED).select(range(N))` — khớp với eval lúc train.

### 3.7 `scripts/train_stages.sh` (mới)

Với mỗi mốc N (đọc từ `Configs.STAGE_SAMPLES`):

1. `train.py --max-samples N --init-adapter <repo>@<revision trước> --push --hub-revision stage-N`
2. `eval_ocr.py --num-test 300 --adapter <adapter local>` → in CER/WER
3. append 1 dòng vào `models/stage_results.tsv`

## 4. Cấu hình đề xuất (A100 40GB)

```
MODEL      tranhuy67896262/Qwen2.5-VL-7B-Instruct-private   (thử --no-4bit = bf16 LoRA, chất lượng tốt hơn QLoRA)
MAX_PIXELS 1280*28*28      MAX_SEQ_LEN 1536
BATCH 4  ACCUM 4  (eff 16)  LR 1e-4  LoRA r=64/α=128  --no-kl
STAGES     5000 10000 20000 40000 59247
EVAL       300 dòng test cố định, CER/WER mỗi mốc
```

Ước tính: Σ ≈ 8,400 optimizer step (~7–12 h trên A100 40GB với flash-attn, không KL).

## 5. Lệnh chạy

```bash
# smoke test trần VRAM trước khi chạy dài
python scripts/train.py --max-samples 100 --batch-size 8 --max-seq-len 1536

# cả 5 mốc, tự push + eval từng mốc
bash scripts/train_stages.sh --hub-repo <owner>/<repo> --eval-samples 300

# xem đường cong
cat models/stage_results.tsv
```

## 6. Rủi ro & cách tránh

| Rủi ro | Cách tránh |
|---|---|
| OOM khi nâng `MAX_PIXELS` lên 1.0M | Giảm `BATCH_SIZE` rồi bù bằng `accum`; **không** hạ `MAX_PIXELS` |
| Mốc nhỏ (5k) quá ít step | Giữ effective batch = 16 (không tăng) |
| Quên kiến thức giữa các mốc | Cumulative đã replay phần trước; nếu vẫn tệ, trộn thêm mẫu mốc cũ |
| Mốc 1 tệ hơn zero-shot | Kiểm tra shuffle/eval trước khi chạy tiếp — chưa chắc tăng data là giải pháp |
| Colab hết session giữa mốc | `--resume` với `models/checkpoints/<run_name>/`; adapter đã push lên Hub nên không mất |
| Push sai revision | `--hub-revision stage-<N>`; `main` là `_latest_` chỉ khi không truyền revision |
