"""
Inference na test setu: baseline i/ili fine-tuned Mistral 7B.

Korišćenje:
  python scripts/inference.py baseline
  python scripts/inference.py finetuned
  python scripts/inference.py both
"""

import json
import sys
import time
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from config_loader import (
    BASE_MODEL_ID, DRY_RUN_MODEL_ID, TOKENIZER_ID,
    MODEL_DIR, SPLITS_DIR, RESULTS_DIR,
    SYSTEM_PROMPT, GENERATION, DRY_RUN,
)


def load_test_tasks() -> list[dict]:
    path = SPLITS_DIR / "test_raw.jsonl"
    if not path.exists():
        print("ERROR: Pokreni prepare_dataset.py prvo.")
        sys.exit(1)
    tasks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    return tasks


def build_user_prompt(task: dict) -> str:
    meta_parts = []
    if task.get("oblast"):
        meta_parts.append(f"oblast={task['oblast']}")
    if task.get("razred"):
        meta_parts.append(f"razred={task['razred']}")
    if task.get("nivo"):
        meta_parts.append(f"nivo={task['nivo']}")
    if task.get("kategorija"):
        meta_parts.append(f"kategorija={task['kategorija']}")
    meta = "; ".join(meta_parts) if meta_parts else "Competition math"
    return (
        f"Task metadata: {meta}\n\n"
        f"Problem statement (in Serbian):\n{task['tekst_zadatka']}\n\n"
        f"Provide a CONCISE, RIGOROUS solution in SERBIAN (Latin script)."
    )


def load_model(model_type: str, tokenizer):
    use_gpu = torch.cuda.is_available()

    if not use_gpu and not DRY_RUN:
        print("ERROR: Nema GPU-a. Postavi dry_run: true za CPU test.")
        sys.exit(1)

    model_id = DRY_RUN_MODEL_ID
    bnb_config = None
    dtype = torch.float32
    device_map = None

    if use_gpu:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        dtype = torch.bfloat16
        device_map = "auto"

    print(f"\nLoading: {model_type} ({model_id}, {'GPU' if use_gpu else 'CPU'})...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    # Resize embeddings ako tokenizer i model imaju različit vocab size
    if len(tokenizer) != base_model.config.vocab_size:
        print(f"  Resizing embeddings: {base_model.config.vocab_size} → {len(tokenizer)}")
        base_model.resize_token_embeddings(len(tokenizer))

    if model_type == "finetuned":
        if not MODEL_DIR.exists():
            print(f"ERROR: Fine-tuned model nije pronadjen: {MODEL_DIR}")
            print("Pokreni finetune.py prvo.")
            sys.exit(1)
        base_model = PeftModel.from_pretrained(base_model, str(MODEL_DIR))
        print("  LoRA adapteri učitani.")

    base_model.eval()
    return base_model


def generate_solution(model, tokenizer, task: dict) -> str:
    messages = [
        {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{build_user_prompt(task)}"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    max_tokens = 64 if DRY_RUN else GENERATION["max_new_tokens"]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=GENERATION["temperature"],
            top_p=GENERATION["top_p"],
            do_sample=True,
            repetition_penalty=GENERATION["repetition_penalty"],
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def run_inference(model_type: str):
    print("=" * 60)
    print(f"INFERENCE: {model_type.upper()} {'(DRY RUN)' if DRY_RUN else ''}")
    print("=" * 60)

    tasks = load_test_tasks()
    if DRY_RUN:
        tasks = tasks[:3]
    print(f"Zadataka: {len(tasks)}")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = load_model(model_type, tokenizer)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"predictions_{model_type}.jsonl"
    results = []
    start = time.time()

    for i, task in enumerate(tasks, 1):
        t0 = time.time()
        solution = generate_solution(model, tokenizer, task)
        dt = time.time() - t0

        result = {
            "id": task["id"],
            "kategorija": task.get("kategorija", ""),
            "razred": task.get("razred", ""),
            "nivo": task.get("nivo", ""),
            "oblast": task.get("oblast", ""),
            "tekst_zadatka": task["tekst_zadatka"],
            "resenje_gt": task.get("resenje", "") or task.get("completion", ""),
            "resenje_model": solution,
            "model_type": model_type,
            "generation_time_sec": round(dt, 2),
        }
        results.append(result)

        remaining = (time.time() - start) / i * (len(tasks) - i)
        print(f"  [{i}/{len(tasks)}] {task['id']} ({dt:.1f}s, ~{remaining/60:.1f}min left)")

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{len(results)} predikcija → {output_path}")
    print(f"Ukupno: {(time.time()-start)/60:.1f} min")


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_type", choices=["baseline", "finetuned", "both"])
    args = parser.parse_args()

    if args.model_type == "both":
        run_inference("baseline")
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        run_inference("finetuned")
    else:
        run_inference(args.model_type)


if __name__ == "__main__":
    run()
