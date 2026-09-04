import json
import math
from datetime import datetime

import torch
from transformers import Trainer, TrainingArguments

from src.data.collator import DataCollatorForQwenVL
from src.train.kl_trainer import KLLoRATrainer
from src.utils.logging import setup_file_logging


def get_training_args(config, output_dir, use_4bit, num_train_steps=None):
    use_bf16 = torch.cuda.is_bf16_supported()
    args = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=config.NUM_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        lr_scheduler_type=config.LR_SCHEDULER,
        max_grad_norm=1.0,
        logging_steps=config.LOGGING_STEPS,
        save_steps=config.SAVE_STEPS,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=not use_bf16,
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        gradient_checkpointing=config.GRADIENT_CHECKPOINTING,
        remove_unused_columns=False,
        seed=config.SEED,
        report_to=["none"],
        save_strategy="steps",
    )
    if num_train_steps:
        args["warmup_steps"] = max(1, int(config.WARMUP_RATIO * num_train_steps))
    if config.GRADIENT_CHECKPOINTING:
        args["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    return TrainingArguments(**args)


def train(config, model, processor, train_ds, eval_ds=None, push=False, hub_repo_id="", use_4bit=None):
    if use_4bit is None:
        use_4bit = config.USE_4BIT
    log_path = config.MODELS_DIR / "training.log"
    setup_file_logging(log_path)

    collator = DataCollatorForQwenVL(processor)
    eff_batch = config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS
    num_train_steps = None
    if train_ds is not None:
        steps_per_epoch = math.ceil(len(train_ds) / eff_batch)
        num_train_steps = steps_per_epoch * config.NUM_EPOCHS
    args = get_training_args(config, config.MODELS_DIR / "checkpoints", use_4bit, num_train_steps)

    trainer_cls = KLLoRATrainer if config.KL_REGULARIZATION else Trainer
    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )
    if trainer_cls is KLLoRATrainer:
        trainer_kwargs["kl_coef"] = config.KL_COEFFICIENT
    trainer = trainer_cls(**trainer_kwargs)
    trainer.train()

    config.ADAPTER_DIR.mkdir(exist_ok=True)
    model.save_pretrained(config.ADAPTER_DIR)
    processor.save_pretrained(config.ADAPTER_DIR)
    print(f"Đã lưu LoRA adapter vào: {config.ADAPTER_DIR}")

    _save_training_metadata(config, train_ds, eval_ds, trainer)
    _append_log_summary(config, log_path, train_ds, eval_ds, trainer)

    if push:
        if not hub_repo_id:
            raise ValueError(
                "Chưa có tên repo Hub để push. Truyền --hub-repo <owner>/<repo> (repo sẽ được tạo mới nếu chưa tồn tại)."
            )
        model.push_to_hub(hub_repo_id, token=config.HF_TOKEN)
        processor.push_to_hub(hub_repo_id, token=config.HF_TOKEN)
        print(f"Đã push adapter lên Hub: {hub_repo_id}")

    return config.ADAPTER_DIR


def _save_training_metadata(config, train_ds, eval_ds, trainer):
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
        "num_epochs": config.NUM_EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "gradient_accumulation_steps": config.GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": config.LEARNING_RATE,
        "lr_scheduler": config.LR_SCHEDULER,
        "max_seq_len": config.MAX_SEQ_LEN,
        "lora_r": config.LORA_R,
        "lora_alpha": config.LORA_ALPHA,
        "lora_dropout": config.LORA_DROPOUT,
        "use_4bit": config.USE_4BIT,
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


def _append_log_summary(config, log_path, train_ds, eval_ds, trainer):
    """Append 1 dòng tổng kết mỗi lần train vào file .log (giữ lịch sử nhiều run)."""
    train_samples = len(train_ds) if train_ds is not None else 0
    eval_samples = len(eval_ds) if eval_ds is not None else 0
    global_step = getattr(getattr(trainer, "state", None), "global_step", 0)

    line = (
        f"[{datetime.now().isoformat(timespec='seconds')}] "
        f"train={train_samples} eval={eval_samples} steps={global_step} "
        f"epochs={config.NUM_EPOCHS} lr={config.LEARNING_RATE} "
        f"lora_r={config.LORA_R} lora_alpha={config.LORA_ALPHA} "
        f"dataset={config.DATASET_NAME} model={config.MODEL_NAME} "
        f"adapter={config.ADAPTER_DIR}\n"
    )
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line)
    print(f"Đã ghi tổng kết vào: {log_path}")