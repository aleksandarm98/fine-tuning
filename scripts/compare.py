"""
Step 5: Compare baseline vs fine-tuned results.

Generates comprehensive comparison tables for the conference paper:
  - Overall scores
  - By oblast (Algebra, Geometrija, Kombinatorika, Teorija brojeva, Analiza)
  - By nivo (opštinsko, okružno, državno)
  - By kategorija (A, B)
  - Statistical significance (paired bootstrap test)
  - Diagnostic patterns

Output: results/comparison_report.txt + results/comparison_tables.csv
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev

import numpy as np

from config_loader import RESULTS_DIR, EVALUATION


def load_evaluated(model_type: str) -> dict[str, dict]:
    """Load evaluated results indexed by task ID."""
    path = RESULTS_DIR / f"evaluated_{model_type}.jsonl"
    if not path.exists():
        print(f"ERROR: {path} not found. Run 04_evaluate.py first.")
        sys.exit(1)

    results = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                results[r["id"]] = r
    return results


def paired_bootstrap_test(
    scores_a: list[float],
    scores_b: list[float],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict:
    """
    Paired bootstrap test for significance.
    H0: mean(scores_b) <= mean(scores_a) (fine-tuned is not better)
    Returns p-value and confidence interval for improvement.
    """
    rng = np.random.RandomState(seed)
    diffs = np.array(scores_b) - np.array(scores_a)
    observed_diff = np.mean(diffs)

    bootstrap_diffs = []
    n = len(diffs)
    for _ in range(n_bootstrap):
        sample = rng.choice(diffs, size=n, replace=True)
        bootstrap_diffs.append(np.mean(sample))

    bootstrap_diffs = np.array(bootstrap_diffs)
    p_value = np.mean(bootstrap_diffs <= 0)

    ci_low = np.percentile(bootstrap_diffs, 2.5)
    ci_high = np.percentile(bootstrap_diffs, 97.5)

    return {
        "observed_diff": observed_diff,
        "p_value": p_value,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
    }


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def format_diff(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value * 100:.1f}pp"


def main():
    print("=" * 70)
    print("COMPARISON: Baseline vs Fine-Tuned")
    print("=" * 70)

    # Load results
    baseline = load_evaluated("baseline")
    finetuned = load_evaluated("finetuned")

    # Align by task ID
    common_ids = sorted(set(baseline.keys()) & set(finetuned.keys()))
    print(f"\nCommon tasks: {len(common_ids)}")

    if len(common_ids) == 0:
        print("ERROR: No common tasks found between baseline and fine-tuned results.")
        sys.exit(1)

    missing_baseline = set(finetuned.keys()) - set(baseline.keys())
    missing_finetuned = set(baseline.keys()) - set(finetuned.keys())
    if missing_baseline:
        print(f"  Warning: {len(missing_baseline)} tasks in fine-tuned but not baseline")
    if missing_finetuned:
        print(f"  Warning: {len(missing_finetuned)} tasks in baseline but not fine-tuned")

    metrics = ["judge_a_factual", "judge_b_logic", "judge_c_quality", "aggregate_weighted"]
    metric_labels = {
        "judge_a_factual": "Studija A (Tačnost)",
        "judge_b_logic": "Studija B (Logika)",
        "judge_c_quality": "Studija C (Kvalitet)",
        "aggregate_weighted": "Agregatni skor",
    }

    report_lines = []

    def report(line=""):
        report_lines.append(line)
        print(line)

    # ================================================================
    # 1. Overall comparison
    # ================================================================
    report("\n" + "=" * 70)
    report("1. OVERALL COMPARISON")
    report("=" * 70)

    header = f"{'Metrika':<25} {'Baseline':>10} {'Fine-tuned':>10} {'Δ':>10} {'p-value':>10}"
    report(header)
    report("-" * len(header))

    for m in metrics:
        b_scores = [baseline[tid][m] for tid in common_ids]
        f_scores = [finetuned[tid][m] for tid in common_ids]

        b_mean = mean(b_scores)
        f_mean = mean(f_scores)
        diff = f_mean - b_mean

        bt = paired_bootstrap_test(b_scores, f_scores)
        sig = "***" if bt["p_value"] < 0.001 else "**" if bt["p_value"] < 0.01 else "*" if bt["p_value"] < 0.05 else ""

        report(
            f"{metric_labels[m]:<25} "
            f"{format_pct(b_mean):>10} "
            f"{format_pct(f_mean):>10} "
            f"{format_diff(diff):>10} "
            f"{bt['p_value']:>8.4f}{sig:>2}"
        )

    report("\n  * p<0.05, ** p<0.01, *** p<0.001 (paired bootstrap, 10000 iterations)")

    # ================================================================
    # 2. By Oblast
    # ================================================================
    report("\n" + "=" * 70)
    report("2. BY OBLAST (Domain)")
    report("=" * 70)

    oblasti = defaultdict(list)
    for tid in common_ids:
        oblast = baseline[tid].get("oblast", "?")
        if oblast:
            oblast = oblast[0].upper() + oblast[1:]  # Normalize
        oblasti[oblast].append(tid)

    for m in metrics:
        report(f"\n  {metric_labels[m]}:")
        header = f"    {'Oblast':<20} {'N':>4} {'Baseline':>10} {'Fine-tuned':>10} {'Δ':>10}"
        report(header)
        report("    " + "-" * (len(header) - 4))

        for oblast in sorted(oblasti.keys()):
            tids = oblasti[oblast]
            b_mean = mean(baseline[tid][m] for tid in tids)
            f_mean = mean(finetuned[tid][m] for tid in tids)
            diff = f_mean - b_mean
            report(
                f"    {oblast:<20} {len(tids):>4} "
                f"{format_pct(b_mean):>10} {format_pct(f_mean):>10} {format_diff(diff):>10}"
            )

    # ================================================================
    # 3. By Nivo
    # ================================================================
    report("\n" + "=" * 70)
    report("3. BY NIVO (Competition Level)")
    report("=" * 70)

    nivoi = defaultdict(list)
    for tid in common_ids:
        nivo = baseline[tid].get("nivo", "?")
        nivoi[nivo].append(tid)

    for m in metrics:
        report(f"\n  {metric_labels[m]}:")
        header = f"    {'Nivo':<20} {'N':>4} {'Baseline':>10} {'Fine-tuned':>10} {'Δ':>10}"
        report(header)
        report("    " + "-" * (len(header) - 4))

        for nivo in ["opstinsko", "opštinsko", "okruzno", "drzavno"]:
            if nivo not in nivoi:
                continue
            tids = nivoi[nivo]
            b_mean = mean(baseline[tid][m] for tid in tids)
            f_mean = mean(finetuned[tid][m] for tid in tids)
            diff = f_mean - b_mean
            report(
                f"    {nivo:<20} {len(tids):>4} "
                f"{format_pct(b_mean):>10} {format_pct(f_mean):>10} {format_diff(diff):>10}"
            )

    # ================================================================
    # 4. By Kategorija
    # ================================================================
    report("\n" + "=" * 70)
    report("4. BY KATEGORIJA (Difficulty)")
    report("=" * 70)

    kategorije = defaultdict(list)
    for tid in common_ids:
        kat = baseline[tid].get("kategorija", "?")
        kategorije[kat].append(tid)

    m = "aggregate_weighted"
    header = f"    {'Kategorija':<20} {'N':>4} {'Baseline':>10} {'Fine-tuned':>10} {'Δ':>10}"
    report(header)
    report("    " + "-" * (len(header) - 4))

    for kat in sorted(kategorije.keys()):
        tids = kategorije[kat]
        b_mean = mean(baseline[tid][m] for tid in tids)
        f_mean = mean(finetuned[tid][m] for tid in tids)
        diff = f_mean - b_mean
        report(
            f"    {kat:<20} {len(tids):>4} "
            f"{format_pct(b_mean):>10} {format_pct(f_mean):>10} {format_diff(diff):>10}"
        )

    # ================================================================
    # 5. Diagnostic patterns
    # ================================================================
    report("\n" + "=" * 70)
    report("5. DIAGNOSTIC PATTERNS")
    report("=" * 70)

    improved = 0
    degraded = 0
    unchanged = 0

    for tid in common_ids:
        b = baseline[tid]["aggregate_weighted"]
        f = finetuned[tid]["aggregate_weighted"]
        if f > b + 0.01:
            improved += 1
        elif f < b - 0.01:
            degraded += 1
        else:
            unchanged += 1

    report(f"\n  Tasks where fine-tuned is BETTER:   {improved} ({improved/len(common_ids)*100:.1f}%)")
    report(f"  Tasks where fine-tuned is WORSE:    {degraded} ({degraded/len(common_ids)*100:.1f}%)")
    report(f"  Tasks roughly UNCHANGED:            {unchanged} ({unchanged/len(common_ids)*100:.1f}%)")

    # Newly correct (A: 0→1)
    newly_correct = sum(
        1 for tid in common_ids
        if baseline[tid]["judge_a_factual"] == 0 and finetuned[tid]["judge_a_factual"] == 1
    )
    newly_wrong = sum(
        1 for tid in common_ids
        if baseline[tid]["judge_a_factual"] == 1 and finetuned[tid]["judge_a_factual"] == 0
    )
    report(f"\n  Newly CORRECT answers (A: 0→1): {newly_correct}")
    report(f"  Newly WRONG answers (A: 1→0):   {newly_wrong}")
    report(f"  Net gain in correct answers:     {newly_correct - newly_wrong}")

    # ================================================================
    # Save report
    # ================================================================
    report_path = RESULTS_DIR / "comparison_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\nReport saved to: {report_path}")

    # Save CSV for Excel
    csv_path = RESULTS_DIR / "comparison_per_task.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        header_cols = [
            "id", "oblast", "nivo", "kategorija", "razred",
            "baseline_A", "baseline_B", "baseline_C", "baseline_agg",
            "finetuned_A", "finetuned_B", "finetuned_C", "finetuned_agg",
            "diff_agg",
        ]
        f.write(",".join(header_cols) + "\n")

        for tid in common_ids:
            b = baseline[tid]
            ft = finetuned[tid]
            row = [
                tid,
                b.get("oblast", ""),
                b.get("nivo", ""),
                b.get("kategorija", ""),
                str(b.get("razred", "")),
                f"{b['judge_a_factual']:.2f}",
                f"{b['judge_b_logic']:.2f}",
                f"{b['judge_c_quality']:.2f}",
                f"{b['aggregate_weighted']:.4f}",
                f"{ft['judge_a_factual']:.2f}",
                f"{ft['judge_b_logic']:.2f}",
                f"{ft['judge_c_quality']:.2f}",
                f"{ft['aggregate_weighted']:.4f}",
                f"{ft['aggregate_weighted'] - b['aggregate_weighted']:.4f}",
            ]
            f.write(",".join(row) + "\n")

    print(f"Per-task CSV: {csv_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
