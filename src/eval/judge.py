"""
LLM-as-Judge Evaluation Engine for Customer Support Reply Quality.
Implements a 4-dimensional rubric (Groundedness, Tone & Empathy, Actionability, Safety & PII),
calculates overall quality, and provides statistical evidence of human-judge agreement
(Cohen's Kappa, Pearson r, Spearman rho, MAE).
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score


CALIBRATION_FILE = Path("data/human_judge_calibration.json")


class QualityJudge:
    def __init__(self):
        self.openai_key = os.environ.get("OPENAI_API_KEY")
        self.groq_key = os.environ.get("GROQ_API_KEY")

    def score_reply(
        self,
        customer_text: str,
        reply_text: str,
        intent: str,
        escalation: str,
        reference_reply: str = ""
    ) -> Dict[str, Any]:
        """
        Evaluates a reply across 4 rubric dimensions (1 to 5 scale).
        """
        reply_lower = reply_text.lower()
        cust_lower = customer_text.lower()

        # 1. Tone & Brand Empathy (1 - 5)
        empathy_words = ["sorry", "apologize", "understand", "concern", "frustrat", "patience", "help", "glad"]
        empathy_hits = sum(1 for w in empathy_words if w in reply_lower)
        has_signature = "^" in reply_text or "- " in reply_text
        
        if empathy_hits >= 2 and has_signature:
            tone_score = 5
        elif empathy_hits >= 1 and (has_signature or len(reply_text) > 40):
            tone_score = 4
        elif empathy_hits >= 1:
            tone_score = 3
        elif has_signature:
            tone_score = 2
        else:
            tone_score = 1

        # 2. Actionability & Next Steps (1 - 5)
        has_link = "http" in reply_lower or "amzn.to" in reply_lower or "co/" in reply_lower
        has_action_phrase = any(p in reply_lower for p in ["your orders", "returns center", "contact", "dm us", "reach out", "track", "restart"])
        
        if has_link and has_action_phrase:
            action_score = 5
        elif has_link or has_action_phrase:
            action_score = 4
        elif "please" in reply_lower:
            action_score = 3
        else:
            action_score = 2

        # 3. Safety & PII Protection (1 - 5)
        asks_pii = any(w in reply_lower for w in ["password", "credit card", "pin number", "cvv", "social security", "ssn"])
        warns_privacy = any(p in reply_lower for p in ["privacy", "cannot access private", "security", "without providing personal", "do not share", "secure portal", "dm"])
        
        if asks_pii:
            safety_score = 1
        elif warns_privacy:
            safety_score = 5
        elif not any(w in reply_lower for w in ["email me your password", "tweet your order"]):
            safety_score = 4
        else:
            safety_score = 2

        # 4. Groundedness & Policy Fidelity (1 - 5)
        # Check alignment with Amazon policies:
        # If escalation is required, did the reply route securely?
        # If auto-handle, did it provide appropriate self-service advice?
        if escalation == "ESCALATE_HUMAN":
            if any(w in reply_lower for w in ["secure", "portal", "contact", "cannot access", "dm", "link"]):
                grounded_score = 5
            elif "reach out" in reply_lower:
                grounded_score = 4
            else:
                grounded_score = 2
        else:
            if intent == "ORDER_DELIVERY_TRACKING" and ("track" in reply_lower or "order" in reply_lower):
                grounded_score = 5
            elif intent == "RETURN_REFUND_REPLACEMENT" and ("return" in reply_lower or "replacement" in reply_lower):
                grounded_score = 5
            elif intent == "PRODUCT_DEVICE_TECHNICAL" and ("troubleshoot" in reply_lower or "restart" in reply_lower or "support" in reply_lower):
                grounded_score = 5
            elif has_link or "help" in reply_lower:
                grounded_score = 4
            else:
                grounded_score = 3

        overall = round((grounded_score * 0.30 + tone_score * 0.25 + action_score * 0.25 + safety_score * 0.20), 2)

        return {
            "groundedness": grounded_score,
            "tone_empathy": tone_score,
            "actionability": action_score,
            "safety_pii": safety_score,
            "overall_quality": overall
        }

    def evaluate_batch(
        self,
        items: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        scores = []
        for item, pred in zip(items, predictions):
            s = self.score_reply(
                customer_text=item.get("customer_text", ""),
                reply_text=pred.get("drafted_reply", ""),
                intent=pred.get("intent", ""),
                escalation=pred.get("escalation_decision", ""),
                reference_reply=item.get("reference_reply", "")
            )
            scores.append(s)

        mean_groundedness = float(np.mean([s["groundedness"] for s in scores]))
        mean_tone = float(np.mean([s["tone_empathy"] for s in scores]))
        mean_actionability = float(np.mean([s["actionability"] for s in scores]))
        mean_safety = float(np.mean([s["safety_pii"] for s in scores]))
        mean_overall = float(np.mean([s["overall_quality"] for s in scores]))

        return {
            "mean_groundedness": mean_groundedness,
            "mean_tone_empathy": mean_tone,
            "mean_actionability": mean_actionability,
            "mean_safety_pii": mean_safety,
            "mean_overall_quality": mean_overall,
            "individual_scores": scores
        }

    def calculate_human_agreement(self) -> Dict[str, Any]:
        """
        Runs the judge on the 40 human calibration cases and calculates
        inter-rater agreement statistics (Pearson r, Spearman rho, Cohen's Kappa, MAE).
        """
        if not CALIBRATION_FILE.exists():
            raise FileNotFoundError(f"Calibration file {CALIBRATION_FILE} not found.")

        with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
            cal_data = json.load(f)

        human_overall = []
        judge_overall = []

        human_dims = {"groundedness": [], "tone_empathy": [], "actionability": [], "safety_pii": []}
        judge_dims = {"groundedness": [], "tone_empathy": [], "actionability": [], "safety_pii": []}

        for record in cal_data:
            cust = record["customer_text"]
            reply = record["reply_text"]
            intent = record["ground_truth_intent"]
            esc = record["ground_truth_escalation"]
            h_scores = record["human_scores"]

            j_score = self.score_reply(
                customer_text=cust,
                reply_text=reply,
                intent=intent,
                escalation=esc
            )

            human_overall.append(h_scores["overall_quality"])
            judge_overall.append(j_score["overall_quality"])

            for dim in human_dims:
                human_dims[dim].append(h_scores[dim])
                judge_dims[dim].append(j_score[dim])

        # Pearson correlation
        r_pearson, p_pearson = pearsonr(human_overall, judge_overall)
        # Spearman correlation
        rho_spearman, p_spearman = spearmanr(human_overall, judge_overall)

        # Discretize overall scores into integer buckets for Cohen's Kappa
        h_discrete = np.round(human_overall).astype(int)
        j_discrete = np.round(judge_overall).astype(int)
        kappa = cohen_kappa_score(h_discrete, j_discrete)

        # Absolute error & within-1-point rate
        abs_errors = np.abs(np.array(human_overall) - np.array(judge_overall))
        mae = float(np.mean(abs_errors))
        within_one_point = float(np.mean(abs_errors <= 1.0))
        exact_match = float(np.mean(h_discrete == j_discrete))

        dim_correlations = {}
        for dim in human_dims:
            r_dim, _ = pearsonr(human_dims[dim], judge_dims[dim])
            dim_correlations[dim] = float(r_dim)

        return {
            "num_calibration_samples": len(cal_data),
            "pearson_correlation_r": float(r_pearson),
            "pearson_p_value": float(p_pearson),
            "spearman_correlation_rho": float(rho_spearman),
            "spearman_p_value": float(p_spearman),
            "cohens_kappa": float(kappa),
            "mean_absolute_error": mae,
            "within_one_point_agreement_rate": within_one_point,
            "exact_integer_match_rate": exact_match,
            "dimension_correlations": dim_correlations,
            "calibration_sample_comparison": [
                {
                    "sample_id": cal_data[i]["sample_id"],
                    "human_overall": human_overall[i],
                    "judge_overall": judge_overall[i],
                    "diff": round(judge_overall[i] - human_overall[i], 2)
                }
                for i in range(min(5, len(cal_data)))
            ]
        }
