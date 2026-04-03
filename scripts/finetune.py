"""
QLoRA fine-tuning Mistral-7B-Instruct.

4-bit NF4 kvantizacija + LoRA adapteri.
Dry run: trenira 2 koraka na 4 primera (CPU).
"""

import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from config_loader import (
    BASE_MODEL_ID, DRY_RUN_MODEL_ID, TOKENIZER_ID,
    SPLITS_DIR, CHECKPOINTS_DIR, MODEL_DIR,
    QLORA, TRAINING, DRY_RUN,
)


def load_jsonl(path: Path) -> Dataset:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return Dataset.from_list(records)


def run():
    print("=" * 60)
    print(f"FINE-TUNING {'(DRY RUN)' if DRY_RUN else ''}")
    print("=" * 60)

    # GPU check
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print(f"\nGPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB)")
    elif not DRY_RUN:
        print("\nERROR: Nema GPU-a. Pokreni sa dry_run: true u config.yaml za CPU test.")
        sys.exit(1)
    else:
        print("\n[DRY RUN] Nema GPU-a — koristim CPU (sporo ali dovoljno za test).")

    # Load data
    train_path = SPLITS_DIR / "train.jsonl"
    val_path = SPLITS_DIR / "val.jsonl"
    if not train_path.exists():
        print("ERROR: Pokreni prepare_dataset.py prvo.")
        sys.exit(1)

    train_ds = load_jsonl(train_path)
    val_ds = load_jsonl(val_path)

    if DRY_RUN:
        train_ds = train_ds.select(range(min(4, len(train_ds))))
        val_ds = val_ds.select(range(min(2, len(val_ds))))

    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Tokenizer (Mistral tokenizer čak i u dry run — ima chat template)
    model_id = DRY_RUN_MODEL_ID
    print(f"\nModel: {model_id}" + (f" (tokenizer: {TOKENIZER_ID})" if DRY_RUN else ""))
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # Model
    if use_gpu:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        # CPU: load in float32, no quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )

    # Resize embeddings ako tokenizer i model imaju različit vocab size
    if len(tokenizer) != model.config.vocab_size:
        print(f"  Resizing embeddings: {model.config.vocab_size} → {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    print(f"  Parametri: {model.num_parameters():,}")

    # LoRA
    lora_config = LoraConfig(
        r=QLORA["rank"],
        lora_alpha=QLORA["alpha"],
        lora_dropout=QLORA["dropout"],
        target_modules=QLORA["target_modules"],
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"  Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # Training config
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # Dry run overrides
    num_epochs = 1 if DRY_RUN else TRAINING["num_epochs"]
    max_steps = 2 if DRY_RUN else -1
    batch_size = 1 if DRY_RUN else TRAINING["batch_size"]
    grad_accum = 1 if DRY_RUN else TRAINING["gradient_accumulation_steps"]
    eval_steps = 1 if DRY_RUN else TRAINING.get("eval_steps", 10)
    seq_len = 512 if DRY_RUN else TRAINING["max_seq_length"]

    training_args = SFTConfig(
        output_dir=str(CHECKPOINTS_DIR),
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=TRAINING["learning_rate"],
        lr_scheduler_type=TRAINING["lr_scheduler"],
        warmup_ratio=TRAINING["warmup_ratio"],
        weight_decay=TRAINING["weight_decay"],
        bf16=(use_gpu and TRAINING.get("bf16", True)),
        fp16=False,
        logging_steps=1 if DRY_RUN else 5,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=3 if DRY_RUN else 5,
        load_best_model_at_end=not DRY_RUN,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_length=seq_len,
        report_to="none",
        seed=TRAINING["seed"],
        gradient_checkpointing=use_gpu,
        gradient_checkpointing_kwargs={"use_reentrant": False} if use_gpu else None,
        optim="paged_adamw_8bit" if use_gpu else "adamw_torch",
    )

    # Trainer
    callbacks = [] if DRY_RUN else [
        EarlyStoppingCallback(early_stopping_patience=TRAINING["early_stopping_patience"])
    ]

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    # Train
    print(f"\n{'='*60}")
    print(f"TRAINING START {'(DRY RUN — 2 steps)' if DRY_RUN else ''}")
    print(f"{'='*60}")
    if not DRY_RUN:
        eff_batch = batch_size * grad_accum
        print(f"  Epochs: {num_epochs}")
        print(f"  Effective batch: {eff_batch}")
        print(f"  LR: {TRAINING['learning_rate']}")
        print(f"  LoRA: r={QLORA['rank']}, α={QLORA['alpha']}")
        print(f"  Max seq: {seq_len}")
        print(f"  Early stopping: {TRAINING['early_stopping_patience']} eval steps")

    trainer.train()

    # Save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    # Training log
    log_path = MODEL_DIR / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    print(f"\nModel saved: {MODEL_DIR}")
    print(f"Log: {log_path}")

    if DRY_RUN:
        print("\n[DRY RUN] Pipeline radi. Postavi dry_run: false za pravi trening.")

    print("\nDone.")


if __name__ == "__main__":
    run()
