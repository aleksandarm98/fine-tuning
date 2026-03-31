"""
Multi-Judge evaluacija sa Claude 3.7 Sonnet via AWS Bedrock.

Tri nezavisna sudije:
  A: Tačnost konačnog odgovora (binary 0/1)
  B: Logička koherentnost (0.00–1.00)
  C: Kvalitet objašnjenja (0.00–1.00)

Korišćenje:
  python scripts/evaluate.py baseline
  python scripts/evaluate.py finetuned
  python scripts/evaluate.py both
"""

import json
import re
import sys
import time
import argparse
from pathlib import Path

import boto3

from config_loader import RESULTS_DIR, EVALUATION, DRY_RUN


# -- Judge prompts -----------------------------------------------

JUDGE_A_PROMPT = """You are an EXTREMELY STRICT mathematical answer checker.

Task: Determine if the candidate's final numerical answer is EXACTLY CORRECT.

CRITICAL RULES:
- Return "1" ONLY if answer is EXACTLY correct (or algebraically equivalent)
- Return "0" if answer is wrong, missing, unclear, or ambiguous
- Accept equivalent forms: 0.5 = 1/2 = 50%, √4 = 2
- If answer format is unclear or incomplete → return "0"

Output: Return ONLY the digit "0" or "1", nothing else.
"""

JUDGE_B_PROMPT = """You are an EXTREMELY STRICT mathematical logic validator.

Task: Evaluate the LOGICAL VALIDITY of solution steps with ZERO TOLERANCE for errors.

Ignore whether final answer is correct. Focus ONLY on logical validity of reasoning.

RED FLAGS (automatic score ≤0.3):
- Circular reasoning, unjustified claims, invalid algebra
- Misapplied theorems, logical leaps, undefined notation

Score Scale:
- 1.0: Flawless logic  |  0.8-0.9: Very good, trivial gaps
- 0.6-0.7: Mostly sound |  0.4-0.5: Has logical errors
- 0.2-0.3: Major errors  |  0.0-0.1: Incoherent

Output: Return ONLY a decimal number [0.00, 1.00] with 2 decimals.
"""

JUDGE_C_PROMPT = """You are an EXTREMELY STRICT mathematics competition grader.

Task: Rate EXPLANATION QUALITY as if grading at DMS competition.

AUTOMATIC PENALTIES:
- NOT in Serbian (Latin) → MAX score 0.2
- English-only solution → MAX score 0.1
- Just final answer, no reasoning → MAX score 0.2

Score Scale:
- 0.9-1.0: Perfect competition solution (clear, rigorous, Serbian)
- 0.7-0.8: Good, minor improvements possible
- 0.5-0.6: Acceptable but lacking rigor
- 0.3-0.4: Poor quality or wrong language
- 0.0-0.2: Incoherent or entirely wrong language

Output: Return ONLY a decimal number [0.00, 1.00] with 2 decimals.
"""


def extract_final_answer(text: str) -> str:
    if not text:
        return ""
    for pattern in [
        r"Konačan odgovor\s*:?\s*(.+?)(?:\n|$)",
        r"Odgovor\s*:?\s*(.+?)(?:\n|$)",
        r"Dakle,?\s+(.+?)(?:\n|$)",
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).strip().strip('*"\'.').strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else ""


def call_bedrock(client, system_prompt: str, user_prompt: str) -> str:
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 20,
        "temperature": 0.0,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    })
    response = client.invoke_model(
        modelId=EVALUATION["judge_model"], body=body,
    )
    return json.loads(response["body"].read())["content"][0]["text"].strip()


def evaluate_single(client, pred: dict) -> dict:
    task = pred["tekst_zadatka"]
    ref = pred["resenje_gt"]
    cand = pred["resenje_model"]
    scores = {}

    # Judge A
    try:
        user = (
            f"Problem:\n{task}\n\n"
            f"Reference final answer:\n{extract_final_answer(ref)}\n\n"
            f"Candidate's final answer:\n{extract_final_answer(cand)}\n\n"
            f'Are these answers equivalent? Return "1" if YES, "0" if NO.'
        )
        r = call_bedrock(client, JUDGE_A_PROMPT, user)
        scores["judge_a_factual"] = 1.0 if "1" in r else 0.0
    except Exception as e:
        print(f"    Judge A error: {e}")
        scores["judge_a_factual"] = 0.0

    # Judge B
    try:
        user = (
            f"Problem:\n{task}\n\nReference solution:\n{ref}\n\n"
            f"Candidate solution:\n{cand}\n\n"
            f"Rate LOGICAL VALIDITY [0.00-1.00]. Be STRICT."
        )
        r = call_bedrock(client, JUDGE_B_PROMPT, user)
        m = re.search(r"([01](?:\.\d+)?)", r)
        scores["judge_b_logic"] = max(0.0, min(1.0, float(m.group(1)))) if m else 0.0
    except Exception as e:
        print(f"    Judge B error: {e}")
        scores["judge_b_logic"] = 0.0

    # Judge C
    try:
        user = (
            f"Problem:\n{task}\n\nCandidate solution:\n{cand}\n\n"
            f"Rate EXPLANATION QUALITY [0.00-1.00]. Must be in Serbian."
        )
        r = call_bedrock(client, JUDGE_C_PROMPT, user)
        m = re.search(r"([01](?:\.\d+)?)", r)
        scores["judge_c_quality"] = max(0.0, min(1.0, float(m.group(1)))) if m else 0.0
    except Exception as e:
        print(f"    Judge C error: {e}")
        scores["judge_c_quality"] = 0.0

    # Aggregation
    w = EVALUATION["weights"]
    scores["aggregate_weighted"] = (
        w["factual"] * scores["judge_a_factual"]
        + w["logic"] * scores["judge_b_logic"]
        + w["quality"] * scores["judge_c_quality"]
    )
    return scores


def run_evaluation(model_type: str):
    print("=" * 60)
    print(f"EVALUATE: {model_type.upper()}")
    print("=" * 60)

    pred_path = RESULTS_DIR / f"predictions_{model_type}.jsonl"
    if not pred_path.exists():
        print(f"ERROR: {pred_path} ne postoji. Pokreni inference.py prvo.")
        sys.exit(1)

    preds = []
    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                preds.append(json.loads(line))

    if DRY_RUN:
        preds = preds[:3]
    print(f"Predikcija: {len(preds)}")

    # Bedrock client
    print(f"\nAWS Bedrock ({EVALUATION['aws_region']})...")
    session = boto3.Session(
        profile_name=EVALUATION["aws_profile"],
        region_name=EVALUATION["aws_region"],
    )
    client = session.client("bedrock-runtime")

    try:
        call_bedrock(client, "Return 'OK'.", "Test.")
        print("  Konekcija OK.")
    except Exception as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    # Resume support
    out_path = RESULTS_DIR / f"evaluated_{model_type}.jsonl"
    done = {}
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[r["id"]] = r
        print(f"  Resume: {len(done)} već evaluirano")

    remaining = [p for p in preds if p["id"] not in done]
    print(f"  Preostalo: {len(remaining)}")

    for i, pred in enumerate(remaining, 1):
        print(f"  [{i}/{len(remaining)}] {pred['id']}...", end=" ")
        scores = evaluate_single(client, pred)
        print(
            f"A={scores['judge_a_factual']:.0f} "
            f"B={scores['judge_b_logic']:.2f} "
            f"C={scores['judge_c_quality']:.2f} "
            f"→ {scores['aggregate_weighted']:.2f}"
        )
        done[pred["id"]] = {**pred, **scores}

        # Save incrementally
        with open(out_path, "w", encoding="utf-8") as f:
            for r in done.values():
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        time.sleep(0.5)

    # Summary
    all_r = list(done.values())
    n = len(all_r)
    print(f"\nSummary ({model_type}, N={n}):")
    print(f"  A (Tačnost):  {sum(r['judge_a_factual'] for r in all_r)/n*100:.1f}%")
    print(f"  B (Logika):   {sum(r['judge_b_logic'] for r in all_r)/n*100:.1f}%")
    print(f"  C (Kvalitet): {sum(r['judge_c_quality'] for r in all_r)/n*100:.1f}%")
    print(f"  Agregatni:    {sum(r['aggregate_weighted'] for r in all_r)/n*100:.1f}%")


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_type", choices=["baseline", "finetuned", "both"])
    args = parser.parse_args()

    if args.model_type == "both":
        run_evaluation("baseline")
        run_evaluation("finetuned")
    else:
        run_evaluation(args.model_type)


if __name__ == "__main__":
    run()
