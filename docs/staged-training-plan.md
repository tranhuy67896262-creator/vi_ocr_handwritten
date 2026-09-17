# Kế hoạch: Staged training sequential (5k → 60k) + push Hub định kỳ

Trạng thái: **đang chạy**. Mốc 5k xong (CER 0.0643), mốc 15k đang train.
Thực tế chạy = **sequential lát mới-only** (`--start-samples/--max-samples`), KHÔNG cumulative
như bản nháp cũ. Tài liệu này là nguồn tham chiếu cho thay đổi code tương ứng.

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

## 4. Cấu hình thực tế đang chạy (7B bf16 LoRA, máy GPU trực tiếp)

```
MODEL      tranhuy67896262/Qwen2.5-VL-7B-Instruct-private
DATASET    tranhuy67896262/Viet-Handwriting-OCR-v2-local (59247 train + 1000 test)
MODE       LoRA bf16 (--no-4bit), R=32/alpha=64, base đóng băng (trainable ~1.13%)
BATCH 4  ACCUM 4  (eff 16)  LR 2e-5  epochs 1  KL bật (mặc định)
STAGES     5k → 15k → 25k → 35k → 45k → 60k  (lát mới-only, sequential)
EVAL       100 dòng test, CER/WER so với base + mốc trước
XUẤT BẢN   mỗi mốc: merge → GGUF Q6_K + mmproj → Modelfile dual-FROM → ollama create
```

Mỗi mốc chạy bằng `scripts/stage_train.sh` (uv + .venv + nohup, xem §5).
Ollama bản mới bỏ `ADAPTER` → Modelfile phải dual-FROM (2 dòng FROM: text + mmproj).

## 5. Lệnh chạy

```bash
# 1 lần duy nhất trên máy mới: script tự dựng .venv + cài deps (uv, torch cu128, requirements)
export HF_TOKEN=<token owner tranhuy67896262>
chmod +x scripts/stage_train.sh

# mốc 5k (mốc đầu, chạy tay vì lúc đó chưa có script — không --init-adapter):
python scripts/train.py \
  --dataset tranhuy67896262/Viet-Handwriting-OCR-v2-local \
  --model tranhuy67896262/Qwen2.5-VL-7B-Instruct-private \
  --max-samples 5000 --no-4bit --batch-size 4 \
  --gradient-accumulation-steps 4 --lr 2e-5 --epochs 1 \
  --push --push-every-save --hub-repo tranhuy67896262/qwen25vl-7b-vi-hwr-lora \
  --hub-revision stage-5k --run-name run-5k

# từng mốc tiếp theo (start = mốc trước, max = mốc này, init/hub-revision theo mốc):
./scripts/stage_train.sh 5000  15000 stage-5k  stage-15k run-15k
./scripts/stage_train.sh 15000 25000 stage-15k stage-25k run-25k
./scripts/stage_train.sh 25000 35000 stage-25k stage-35k run-35k
./scripts/stage_train.sh 35000 45000 stage-35k stage-45k run-45k
./scripts/stage_train.sh 45000 59247 stage-45k stage-60k run-60k

# xem tiến độ
tail -f train-run-15k.log

# eval mốc mới (so với base + mốc trước)
python scripts/eval_ocr.py --num-test 100 --no-adapter --model <base-private>
python scripts/eval_ocr.py --num-test 100 --adapter tranhuy67896262/qwen25vl-7b-vi-hwr-lora \
  --adapter-revision stage-15k --model <base-private>

# xuất bản mốc: merge → GGUF → Ollama
python scripts/export_merged.py --adapter models/adapter-stage-15k --output models/qwen25vl-7b-vi-hwr-stage-15k-merged
./scripts/export_gguf.sh models/qwen25vl-7b-vi-hwr-stage-15k-merged models/gguf-stage-15k
cd models/gguf-stage-15k && ollama create qwen25vl-7b-vi-hwr-15k -f Modelfile
```

## 7. Nhật ký chạy thực tế

| Mốc | Lát data | Steps | CER | WER | Ghi chú |
|---|---|---|---|---|---|
| base (chưa train) | — | — | 0.1533 | 0.2721 | đo trên 100 mẫu test |
| stage-5k | 0→5000 | 313 | 0.0643 | 0.1622 | −58% CER. GGUF Q6_K + mmproj → `tranhuythang9999/qwen25vl-7b-vi-hwr-5k` trên Ollama Hub. Test 10 ảnh: 4/10 khớp tuyệt đối, còn lại sai nhỏ |
| stage-15k | 5000→15000 | 938 | … | … | đang train (log: collator + LoRA 1.13% OK) |
| stage-25k | 15000→25000 | … | … | … | chờ |
| stage-35k | 25000→35000 | … | … | … | chờ |
| stage-45k | 35000→45000 | … | … | … | chờ |
| stage-60k | 45000→59247 | … | … | … | chờ (hết data) |

Quy tắc lên mốc: CER giảm tiếp → lên; đứng yên/tăng → dừng, xem xét cumulative hoặc giảm lr.

### Chi tiết mốc stage-5k (mốc đầu, chạy tay — mẫu cho các mốc sau)

```bash
python scripts/train.py \
  --dataset tranhuy67896262/Viet-Handwriting-OCR-v2-local \
  --model tranhuy67896262/Qwen2.5-VL-7B-Instruct-private \
  --max-samples 5000 --no-4bit --batch-size 4 \
  --gradient-accumulation-steps 4 --lr 2e-5 --epochs 1 \
  --push --push-every-save --hub-repo tranhuy67896262/qwen25vl-7b-vi-hwr-lora \
  --hub-revision stage-5k --run-name run-5k
# (không --init-adapter vì là mốc đầu; --start-samples mặc định 0)
```

- 313 steps, LoRA bf16 R32/alpha64 (trainable 1.13%), KL bật.
- Adapter: `models/qwen25vl-7b-vi-hwr-lora-bf16/` + nhánh Hub `stage-5k`
  (`adapter_model.safetensors` 381MB, kèm `adapter_config.json`, tokenizer…).
- Eval 100 mẫu test: base CER 0.1533/WER 0.2721 → stage-5k CER 0.0643/WER 0.1622.
- Xuất bản: `export_merged.py` (fix `peft` không hiểu `@revision` bằng
  `snapshot_download(..., revision='stage-5k', local_dir='models/adapter-stage-5k')`)
  → `export_gguf.sh` (Q6_K + mmproj + Modelfile dual-FROM)
  → `ollama create qwen25vl-7b-vi-hwr-5k` → push `tranhuythang9999/qwen25vl-7b-vi-hwr-5k`.
- Test 10 ảnh qua Ollama: 4/10 khớp tuyệt đối (`img_00/03/04` + gần đúng `07/09`
  chỉ lệch dấu câu/quotes), sai nhỏ ở chữ dễ nhầm (`rằng→sang`, `cỏ mọc→cổ mộc`,
  `Mibelcam→Mibeclam`), 1 ảnh sai nặng (`img_02`).

## 6. Rủi ro & cách tránh

| Rủi ro | Cách tránh |
|---|---|
| OOM khi nâng `MAX_PIXELS` lên 1.0M | Giảm `BATCH_SIZE` rồi bù bằng `accum`; **không** hạ `MAX_PIXELS` |
| Mốc nhỏ (5k) quá ít step | Giữ effective batch = 16 (không tăng) |
| Quên kiến thức giữa các mốc | Cumulative đã replay phần trước; nếu vẫn tệ, trộn thêm mẫu mốc cũ |
| Mốc 1 tệ hơn zero-shot | Kiểm tra shuffle/eval trước khi chạy tiếp — chưa chắc tăng data là giải pháp |
| Colab hết session giữa mốc | `--resume` với `models/checkpoints/<run_name>/`; adapter đã push lên Hub nên không mất |
| Push sai revision | `--hub-revision stage-<N>`; `main` là `_latest_` chỉ khi không truyền revision |
