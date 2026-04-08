"""
Priprema dataseta: učitava 348 zadataka, deli na Train/Val/Test,
konvertuje u chat messages format za SFTTrainer.
"""

import json
from pathlib import Path
from collections import defaultdict

from config_loader import (
    RAW_DATA_DIR, SPLITS_DIR,
    TRAIN_FILES, VAL_FILES, TEST_FILES,
    SYSTEM_PROMPT, DRY_RUN, IS_INSTRUCT_MODEL,
)


def load_tasks(filenames: list[str]) -> list[dict]:
    tasks = []
    for fname in filenames:
        fpath = RAW_DATA_DIR / fname
        if not fpath.exists():
            print(f"  WARNING: {fpath} not found, skipping")
            continue
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        tasks.extend(data)
        print(f"  {fname}: {len(data)} zadataka")
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


def task_to_training_format(task: dict) -> dict:
    solution = task.get("resenje", "") or task.get("completion", "")
    meta = {
        "id": task.get("id", ""),
        "oblast": task.get("oblast", ""),
        "nivo": task.get("nivo", ""),
        "kategorija": task.get("kategorija", ""),
        "razred": task.get("razred", ""),
    }

    if IS_INSTRUCT_MODEL:
        # Instruct modeli: chat messages format za SFTTrainer
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(task)},
                {"role": "assistant", "content": solution},
            ],
            **meta,
        }
    else:
        # Base modeli (Mathstral, itd.): plain text format
        text = (
            f"{SYSTEM_PROMPT}\n\n"
            f"{build_user_prompt(task)}\n\n"
            f"Rešenje:\n{solution}"
        )
        return {"text": text, **meta}


def save_jsonl(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records)} → {path}")


def print_stats(tasks: list[dict], name: str):
    by_oblast = defaultdict(int)
    by_nivo = defaultdict(int)
    for t in tasks:
        by_oblast[t.get("oblast", "?")] += 1
        by_nivo[t.get("nivo", "?")] += 1
    print(f"  {name}: oblast={dict(by_oblast)}, nivo={dict(by_nivo)}")


def run():
    print("=" * 60)
    print("PREPARE DATASET")
    print("=" * 60)

    print("\nTrain:")
    train = load_tasks(TRAIN_FILES)
    print("\nVal:")
    val = load_tasks(VAL_FILES)
    print("\nTest:")
    test = load_tasks(TEST_FILES)

    # Deduplicate (known: 2022-Okruzno-B-4-1 appears in okruzno 2023)
    train_ids = {t["id"] for t in train}
    overlap = {t["id"] for t in val} & train_ids
    if overlap:
        print(f"\n  WARNING: {len(overlap)} duplikat(a) u Train∩Val: {overlap}")
        print("  Uklanjam iz validacije...")
        val = [t for t in val if t["id"] not in overlap]

    total = len(train) + len(val) + len(test)
    print(f"\nUkupno: {total} (Train={len(train)}, Val={len(val)}, Test={len(test)})")
    print_stats(train, "Train")
    print_stats(val, "Val")
    print_stats(test, "Test")

    # Convert and save
    print("\nSaving splits...")
    save_jsonl([task_to_training_format(t) for t in train], SPLITS_DIR / "train.jsonl")
    save_jsonl([task_to_training_format(t) for t in val], SPLITS_DIR / "val.jsonl")
    save_jsonl([task_to_training_format(t) for t in test], SPLITS_DIR / "test.jsonl")

    # Raw copies (for evaluation)
    save_jsonl(train, SPLITS_DIR / "train_raw.jsonl")
    save_jsonl(val, SPLITS_DIR / "val_raw.jsonl")
    save_jsonl(test, SPLITS_DIR / "test_raw.jsonl")

    # Solution length analysis
    print("\nDužina rešenja (karakteri):")
    for name, tasks in [("Train", train), ("Val", val), ("Test", test)]:
        lens = [len(t.get("resenje", "") or t.get("completion", "")) for t in tasks]
        print(f"  {name}: min={min(lens)}, avg={sum(lens)//len(lens)}, max={max(lens)}")

    if DRY_RUN:
        print("\n[DRY RUN] Dataset spreman za test.")

    print("\nDone.")


if __name__ == "__main__":
    run()
