"""Vòng train + lưu adapter + metadata."""
import json
import math
from datetime import datetime
from pathlib import Path

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments

from src.datasets.collator import DataCollatorForQwenVL
from src.train.kl_trainer import KLLoRATrainer
from src.utils.logging import setup_file_logging

# Optimizer theo precision (OCP: thêm mode mới chỉ cần mở rộng mapping).
OPTIMIZER_BY_4BIT = {True: "paged_adamw_8bit", False: "adamw_torch"}


class PushOnSaveCallback(TrainerCallback):
    """Push adapter lên Hub mỗi lần Trainer lưu checkpoint (run dài, mốc dài).

    Bật bằng ``push_every_save=True`` — chặn mất kết quả khi Colab hết session giữa run.
    """

    def __init__(self, repo_id, token=None, revision=None):
        self.repo_id = repo_id
        self.token = token or None
        self.revision = revision

    def on_save(self, args, state, control, **kwargs):
        """Đẩy adapter hiện tại lên Hub ở cuối mỗi lần save."""
        model = kwargs.get("model")
        if model is None or not state.is_world_process_zero:
            return control
        model.push_to_hub(self.repo_id, token=self.token, revision=self.revision)
        suffix = f"@{self.revision}" if self.revision else ""
        print(f"Đã push checkpoint (step {state.global_step}) -> {self.repo_id}{suffix}")
        return control


def ensure_hub_branch(config, hub_repo_id, hub_revision):
    """Tạo branch trên Hub nếu chưa có (push_to_hub commit thẳng vào revision đó)."""
    if not hub_revision:
        return
    try:
        from huggingface_hub import HfApi
        HfApi().create_branch(
            hub_repo_id, branch=hub_revision, repo_type="model",
            exist_ok=True, token=config.HF_TOKEN or None,
        )
    except Exception as err:  # noqa: BLE001
        print(f"  [WARN] Không tạo được branch '{hub_revision}': {err}")


def get_training_args(config, output_dir, use_4bit, num_train_steps=None):
    """Dựng TrainingArguments từ config."""
    use_bf16 = torch.cuda.is_bf16_supported()
    args = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": config.BATCH_SIZE,
        "per_device_eval_batch_size": config.BATCH_SIZE,
        "gradient_accumulation_steps": config.GRADIENT_ACCUMULATION_STEPS,
        "num_train_epochs": config.NUM_EPOCHS,
        "learning_rate": config.LEARNING_RATE,
        "lr_scheduler_type": config.LR_SCHEDULER,
        "max_grad_norm": 1.0,
        "logging_steps": config.LOGGING_STEPS,
        "save_steps": config.SAVE_STEPS,
        "save_total_limit": 2,
        "bf16": use_bf16,
        "fp16": not use_bf16,
        "optim": OPTIMIZER_BY_4BIT[bool(use_4bit)],
        "gradient_checkpointing": config.GRADIENT_CHECKPOINTING,
        "dataloader_num_workers": config.DATALOADER_NUM_WORKERS,
        "dataloader_pin_memory": True,
        "dataloader_persistent_workers": config.DATALOADER_NUM_WORKERS > 0,
        "remove_unused_columns": False,
        "seed": config.SEED,
        "report_to": ["none"],
        "save_strategy": "steps",
    }
    if num_train_steps:
        args["warmup_steps"] = max(1, int(config.WARMUP_RATIO * num_train_steps))
    if config.GRADIENT_CHECKPOINTING:
        args["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    return TrainingArguments(**args)


def _latest_checkpoint(checkpoints_dir):
    """Checkpoint `checkpoint-N` lớn nhất, hoặc None nếu chưa có."""
    ckpts = []
    for p in Path(checkpoints_dir).glob("checkpoint-*"):
        if not p.is_dir():
            continue
        try:
            ckpts.append((int(p.name.split("-")[-1]), str(p)))
        except ValueError:
            continue
    return max(ckpts)[1] if ckpts else None


def train(config, model, processor, train_ds, eval_ds=None, push=False, hub_repo_id="",
          use_4bit=None, resume=False, save_steps=None, hub_revision=None,
          run_name=None, push_every_save=False, init_adapter=None):
    """Train LoRA, lưu adapter + metadata (push Hub nếu cần).

    ``run_name`` TÁCH thư mục checkpoint theo mốc (``models/checkpoints/<run_name>``)
    để ``--resume`` giữa các mốc staged training không lẫn vào nhau.
    ``hub_revision`` commit kết quả vào một branch riêng (vd ``stage-5k``) thay vì đè
    ``main`` — giữ được cả 5 adapter của 5 mốc.
    """
    if use_4bit is None:
        use_4bit = config.USE_4BIT
    if save_steps is not None:
        config.SAVE_STEPS = save_steps
    run_name = run_name or "default"
    log_path = config.MODELS_DIR / "training.log"
    setup_file_logging(log_path)

    collator = DataCollatorForQwenVL(
        processor,
        max_length=config.MAX_SEQ_LEN,
        min_pixels=config.MIN_PIXELS,
        max_pixels=config.MAX_PIXELS,
    )
    eff_batch = config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS
    num_train_steps = None
    if train_ds is not None:
        steps_per_epoch = math.ceil(len(train_ds) / eff_batch)
        num_train_steps = steps_per_epoch * config.NUM_EPOCHS
    ckpt_dir = config.MODELS_DIR / "checkpoints" / run_name
    args = get_training_args(config, ckpt_dir, use_4bit, num_train_steps)

    trainer_cls = KLLoRATrainer if config.KL_REGULARIZATION else Trainer
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "data_collator": collator,
    }
    if trainer_cls is KLLoRATrainer:
        trainer_kwargs["kl_coef"] = config.KL_COEFFICIENT
    if push and push_every_save and hub_repo_id:
        ensure_hub_branch(config, hub_repo_id, hub_revision)
        trainer_kwargs["callbacks"] = [
            PushOnSaveCallback(hub_repo_id, config.HF_TOKEN, hub_revision)
        ]
    trainer = trainer_cls(**trainer_kwargs)
    resume_path = None
    if resume:
        resume_path = _latest_checkpoint(ckpt_dir)
        if resume_path:
            print(f"Tiếp tục từ checkpoint: {resume_path}")
        else:
            print("Không thấy checkpoint — train từ đầu.")
    trainer.train(resume_from_checkpoint=resume_path)

    config.ADAPTER_DIR.mkdir(exist_ok=True)
    model.save_pretrained(config.ADAPTER_DIR)
    processor.save_pretrained(config.ADAPTER_DIR)
    print(f"Đã lưu LoRA adapter vào: {config.ADAPTER_DIR}")

    _save_training_metadata(config, train_ds, eval_ds, trainer, use_4bit,
                            run_name=run_name, init_adapter=init_adapter,
                            hub_revision=hub_revision)
    _append_log_summary(config, log_path, train_ds, eval_ds, trainer, run_name=run_name)

    if push:
        if not hub_repo_id:
            raise ValueError(
                "Chưa có tên repo Hub để push. "
                "Truyền --hub-repo <owner>/<repo> (repo sẽ được tạo mới nếu chưa tồn tại)."
            )
        ensure_hub_branch(config, hub_repo_id, hub_revision)
        model.push_to_hub(hub_repo_id, token=config.HF_TOKEN, revision=hub_revision)
        processor.push_to_hub(hub_repo_id, token=config.HF_TOKEN, revision=hub_revision)
        suffix = f" (revision '{hub_revision}')" if hub_revision else ""
        print(f"Đã push adapter lên Hub: {hub_repo_id}{suffix}")

    return config.ADAPTER_DIR


def _save_training_metadata(config, train_ds, eval_ds, trainer, use_4bit,
                            run_name="default", init_adapter=None, hub_revision=None):
    """Ghi lại số liệu training (đã train bao nhiêu data, cấu hình gì) vào JSON."""
    train_samples = len(train_ds) if train_ds is not None else 0
    eval_samples = len(eval_ds) if eval_ds is not None else 0
    global_step = getattr(getattr(trainer, "state", None), "global_step", 0)

    metadata = {
        "model": config.MODEL_NAME,
        "dataset": config.DATASET_NAME,
        "system_prompt": config.SYSTEM_PROMPT,
        "train_samples": train_samples,
        "eval_samples": eval_samples,
        "train_steps": global_step,
        "run_name": run_name,
        "init_adapter": init_adapter or "",
        "hub_revision": hub_revision or "",
        "max_pixels": config.MAX_PIXELS,
        "num_epochs": config.NUM_EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "gradient_accumulation_steps": config.GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": config.LEARNING_RATE,
        "lr_scheduler": config.LR_SCHEDULER,
        "max_seq_len": config.MAX_SEQ_LEN,
        "lora_r": config.LORA_R,
        "lora_alpha": config.LORA_ALPHA,
        "lora_dropout": config.LORA_DROPOUT,
        "use_4bit": use_4bit,
        "seed": config.SEED,
        "kl_regularization": config.KL_REGULARIZATION,
        "kl_coefficient": config.KL_COEFFICIENT,
        "adapter_dir": str(config.ADAPTER_DIR),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }

    meta_path = config.ADAPTER_DIR / "training_metadata.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã ghi metadata vào: {meta_path}")
    print(f"  -> Đã train {train_samples} mẫu x {config.NUM_EPOCHS} epoch "
          f"(effective batch {config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS})")
    return metadata


def _append_log_summary(config, log_path, train_ds, eval_ds, trainer, run_name="default"):
    """Append 1 dòng tổng kết mỗi lần train vào file .log (giữ lịch sử nhiều run)."""
    train_samples = len(train_ds) if train_ds is not None else 0
    eval_samples = len(eval_ds) if eval_ds is not None else 0
    global_step = getattr(getattr(trainer, "state", None), "global_step", 0)

    line = (
        f"[{datetime.now().isoformat(timespec='seconds')}] run={run_name} "
        f"train={train_samples} eval={eval_samples} steps={global_step} "
        f"epochs={config.NUM_EPOCHS} lr={config.LEARNING_RATE} "
        f"lora_r={config.LORA_R} lora_alpha={config.LORA_ALPHA} "
        f"dataset={config.DATASET_NAME} model={config.MODEL_NAME} "
        f"adapter={config.ADAPTER_DIR}\n"
    )
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line)
    print(f"Đã ghi tổng kết vào: {log_path}")
