"""
Pipeline orkestrator — pokreće sve korake redom.

Korišćenje:
  python scripts/run_pipeline.py              # prepare + finetune + inference
  python scripts/run_pipeline.py --full       # + evaluate + compare
  python scripts/run_pipeline.py --step 2     # samo finetune
"""

import argparse
import sys
import time


def run_step(step_num: int, name: str, func):
    print(f"\n{'#'*60}")
    print(f"# STEP {step_num}: {name}")
    print(f"{'#'*60}\n")
    start = time.time()
    func()
    elapsed = time.time() - start
    print(f"\n[Step {step_num} done in {elapsed/60:.1f} min]")


def main():
    parser = argparse.ArgumentParser(description="DMS Fine-Tuning Pipeline")
    parser.add_argument("--full", action="store_true",
                        help="Run all steps including evaluation (needs AWS Bedrock)")
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4, 5],
                        help="Run only a specific step")
    args = parser.parse_args()

    steps = {
        1: ("Prepare Dataset", lambda: __import__("prepare_dataset").run()),
        2: ("Fine-Tune (QLoRA)", lambda: __import__("finetune").run()),
        3: ("Inference (baseline + finetuned)", lambda: (
            __import__("inference").run_inference("baseline"),
            __import__("torch").cuda.empty_cache() if __import__("torch").cuda.is_available() else None,
            __import__("inference").run_inference("finetuned"),
        )),
        4: ("Evaluate (Claude Judge)", lambda: (
            __import__("evaluate").run_evaluation("baseline"),
            __import__("evaluate").run_evaluation("finetuned"),
        )),
        5: ("Compare Results", lambda: __import__("compare").main()),
    }

    if args.step:
        num = args.step
        name, func = steps[num]
        run_step(num, name, func)
        return

    # Default: steps 1-3 (no AWS needed)
    run_steps = [1, 2, 3]
    if args.full:
        run_steps = [1, 2, 3, 4, 5]

    total_start = time.time()
    for num in run_steps:
        name, func = steps[num]
        run_step(num, name, func)

    total = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE — {total/60:.1f} min total")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
