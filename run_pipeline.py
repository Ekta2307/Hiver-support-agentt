"""
Master Pipeline Runner & Benchmark Harness.
Runs the Golden Evaluation Set (200 cases) across:
1. Baseline 1 (Trivial Baseline)
2. Baseline 2 (Simple Baseline)
3. Proposed System (AmazonSupportAgent)

Computes automated metrics, LLM-as-judge rubric scores, and human-judge agreement statistics.
Outputs comparison tables and saves results to results/benchmark_results.json.
Execution time target: < 15 minutes (typically under 2 minutes).
"""

import os
import sys
import json
import time
from pathlib import Path
import pandas as pd

from src.agent.agent import AmazonSupportAgent
from src.baselines.baselines import TrivialBaseline, SimpleBaseline
from src.eval.automated_metrics import AutomatedEvaluator
from src.eval.judge import QualityJudge


GOLDEN_SET_PATH = Path("data/golden_eval_set.json")
RESULTS_DIR = Path("results")
RESULTS_JSON = RESULTS_DIR / "benchmark_results.json"


def print_banner(title):
    print("\n" + "=" * 80)
    print(f" {title.upper()} ")
    print("=" * 80)


def run_benchmark():
    t_start = time.time()
    print_banner("Hiver AI Support Agent: Benchmark Pipeline")

    if not GOLDEN_SET_PATH.exists():
        print(f"Error: {GOLDEN_SET_PATH} not found. Running build_golden_set.py...")
        from src.data.build_golden_set import build_golden_set
        build_golden_set()

    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    print(f"Loaded Golden Evaluation Set: {len(golden_data)} hand-labelled test cases.")

    # 1. Initialize systems
    print("\n[1/4] Initializing Baseline and Agent Pipelines...")
    t0 = time.time()
    trivial_bl = TrivialBaseline()
    simple_bl = SimpleBaseline()
    proposed_agent = AmazonSupportAgent()
    print(f"Initialization completed in {time.time() - t0:.2f}s")

    # 2. Run inference on Golden Set
    print("\n[2/4] Running Inference Across All 3 Systems...")
    systems = {
        "Baseline 1 (Trivial)": trivial_bl,
        "Baseline 2 (Simple)": simple_bl,
        "Proposed Agent": proposed_agent
    }

    predictions = {name: [] for name in systems}

    for idx, item in enumerate(golden_data, 1):
        cust_text = item["customer_text"]
        for name, system in systems.items():
            pred = system.process_message(cust_text)
            pred["sample_id"] = item["sample_id"]
            predictions[name].append(pred)
        if idx % 50 == 0 or idx == len(golden_data):
            print(f"  Processed {idx}/{len(golden_data)} evaluation cases...")

    # 3. Compute Automated Metrics
    print("\n[3/4] Computing Automated Metrics...")
    evaluator = AutomatedEvaluator()
    auto_metrics = {}
    for name in systems:
        auto_metrics[name] = evaluator.evaluate_pipeline(golden_data, predictions[name])

    # 4. Compute LLM-as-Judge Quality Scores
    print("\n[4/4] Running LLM-as-Judge & Calculating Human Agreement...")
    judge = QualityJudge()
    judge_scores = {}
    for name in systems:
        judge_scores[name] = judge.evaluate_batch(golden_data, predictions[name])

    # Human-Judge agreement calibration
    agreement_stats = judge.calculate_human_agreement()

    # 5. Display Benchmark Summary Table
    print_banner("Benchmark Headline Results")

    summary_rows = []
    for name in systems:
        am = auto_metrics[name]
        js = judge_scores[name]
        summary_rows.append({
            "System": name,
            "Intent Acc (%)": f"{am['intent_classification']['accuracy'] * 100:.1f}%",
            "Intent Macro F1": f"{am['intent_classification']['macro_f1']:.3f}",
            "Escalation Acc (%)": f"{am['escalation_routing']['accuracy'] * 100:.1f}%",
            "Escalation F1": f"{am['escalation_routing']['f1']:.3f}",
            "Escalation FNR (%)": f"{am['escalation_routing']['false_negative_rate'] * 100:.1f}%",
            "Reply Semantic Sim": f"{am['reply_quality']['mean_semantic_similarity']:.3f}",
            "Judge Groundedness (1-5)": f"{js['mean_groundedness']:.2f}",
            "Judge Actionability (1-5)": f"{js['mean_actionability']:.2f}",
            "Judge Overall (1-5)": f"{js['mean_overall_quality']:.2f}"
        })

    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False))

    print("\n" + "-" * 80)
    print("CRITICAL SAFETY METRIC: ESCALATION FALSE NEGATIVE RATE (FNR)")
    print("Lower is better. A false negative means a high-risk/distressed customer was denied human help.")
    for name in systems:
        am = auto_metrics[name]
        fn = am['escalation_routing']['false_negatives']
        fnr = am['escalation_routing']['false_negative_rate'] * 100
        print(f"  {name:25s}: {fn:2d} missed escalations ({fnr:.1f}% FNR)")

    print("\n" + "-" * 80)
    print("HUMAN-JUDGE CALIBRATION & AGREEMENT EVIDENCE (n=40 calibration cases)")
    print(f"  Pearson Correlation (r)       : {agreement_stats['pearson_correlation_r']:.3f} (p = {agreement_stats['pearson_p_value']:.4e})")
    print(f"  Spearman Rank Correlation (rho): {agreement_stats['spearman_correlation_rho']:.3f} (p = {agreement_stats['spearman_p_value']:.4e})")
    print(f"  Cohen's Kappa (discrete)      : {agreement_stats['cohens_kappa']:.3f}")
    print(f"  Mean Absolute Error (MAE)     : {agreement_stats['mean_absolute_error']:.3f} points on 1-5 scale")
    print(f"  Within-1-Point Agreement Rate : {agreement_stats['within_one_point_agreement_rate'] * 100:.1f}%")
    print(f"  Exact Integer Match Rate      : {agreement_stats['exact_integer_match_rate'] * 100:.1f}%")

    print("\nPer-Dimension Human-Judge Correlation:")
    for dim, r_val in agreement_stats["dimension_correlations"].items():
        print(f"  - {dim:18s}: r = {r_val:.3f}")

    # 6. Save comprehensive results to JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    full_output = {
        "execution_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_test_cases": len(golden_data),
        "total_runtime_seconds": round(time.time() - t_start, 2),
        "headline_metrics": summary_rows,
        "automated_metrics": auto_metrics,
        "judge_metrics": {k: {m: v for m, v in js.items() if m != 'individual_scores'} for k, js in judge_scores.items()},
        "human_judge_agreement": agreement_stats,
        "sample_predictions": [
            {
                "sample_id": golden_data[i]["sample_id"],
                "customer_text": golden_data[i]["customer_text"],
                "ground_truth_intent": golden_data[i]["ground_truth_intent"],
                "ground_truth_escalation": golden_data[i]["ground_truth_escalation"],
                "reference_reply": golden_data[i]["reference_reply"],
                "proposed_agent": predictions["Proposed Agent"][i],
                "baseline_simple": predictions["Baseline 2 (Simple)"][i],
                "baseline_trivial": predictions["Baseline 1 (Trivial)"][i]
            }
            for i in range(min(15, len(golden_data)))
        ]
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved full benchmark results to {RESULTS_JSON}")
    print(f"Total benchmark execution time: {time.time() - t_start:.2f} seconds (well under 15 minutes!)")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
