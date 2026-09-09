"""
Automated Metrics Evaluation Engine.
Calculates Accuracy, Macro F1, Per-Class Metrics, Confusion Matrices,
Escalation False Negative Rates, Semantic Similarity, and Safety Adherence.
"""

from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


class AutomatedEvaluator:
    def __init__(self, embedder_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(embedder_name)

    def evaluate_pipeline(
        self,
        golden_set: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        golden_set: list of dicts with 'ground_truth_intent', 'ground_truth_escalation', 'reference_reply'
        predictions: list of dicts with 'intent', 'escalation_decision', 'drafted_reply'
        """
        assert len(golden_set) == len(predictions), "Size mismatch between golden set and predictions"

        y_true_intent = [g["ground_truth_intent"] for g in golden_set]
        y_pred_intent = [p["intent"] for p in predictions]

        y_true_esc = [g["ground_truth_escalation"] for g in golden_set]
        y_pred_esc = [p["escalation_decision"] for p in predictions]

        # 1. Intent Metrics
        intent_labels = sorted(list(set(y_true_intent)))
        intent_acc = accuracy_score(y_true_intent, y_pred_intent)
        prec, rec, f1, support = precision_recall_fscore_support(
            y_true_intent, y_pred_intent, labels=intent_labels, zero_division=0
        )
        macro_f1 = float(np.mean(f1))
        macro_prec = float(np.mean(prec))
        macro_rec = float(np.mean(rec))

        per_intent_metrics = {}
        for i, label in enumerate(intent_labels):
            per_intent_metrics[label] = {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(support[i])
            }

        conf_matrix = confusion_matrix(y_true_intent, y_pred_intent, labels=intent_labels).tolist()

        # 2. Escalation Routing Metrics
        esc_labels = ["AUTO_HANDLE", "ESCALATE_HUMAN"]
        esc_acc = accuracy_score(y_true_esc, y_pred_esc)
        esc_p, esc_r, esc_f1, _ = precision_recall_fscore_support(
            y_true_esc, y_pred_esc, labels=["ESCALATE_HUMAN"], pos_label="ESCALATE_HUMAN", average="binary", zero_division=0
        )

        # Escalation Confusion & False Negative Rate (FNR)
        # False negative = true escalation was misrouted to auto-handle (dangerous!)
        cm_esc = confusion_matrix(y_true_esc, y_pred_esc, labels=esc_labels)
        # cm_esc[1, 0] is True ESCALATE misclassified as AUTO_HANDLE
        false_negatives = int(cm_esc[1, 0])
        total_escalations = int(cm_esc[1, 0] + cm_esc[1, 1])
        fnr = float(false_negatives / total_escalations) if total_escalations > 0 else 0.0

        # 3. Reply Quality Automated Metrics
        ref_replies = [g.get("reference_reply_clean") or g.get("reference_reply", "") for g in golden_set]
        draft_replies = [p.get("drafted_reply", "") for p in predictions]

        ref_embs = self.model.encode(ref_replies, normalize_embeddings=True, show_progress_bar=False)
        draft_embs = self.model.encode(draft_replies, normalize_embeddings=True, show_progress_bar=False)

        # Pairwise cosine similarities
        cosine_sims = [float(np.dot(r, d)) for r, d in zip(ref_embs, draft_embs)]
        mean_cosine_sim = float(np.mean(cosine_sims))

        # Twitter character compliance (< 280 chars)
        char_lengths = [len(r) for r in draft_replies]
        compliance_280 = float(np.mean([1 if l <= 280 else 0 for l in char_lengths]))

        # PII violation checks (asking for password, card, pin)
        pii_violations = 0
        for reply in draft_replies:
            reply_low = reply.lower()
            if any(w in reply_low for w in ["password", "credit card", "pin number", "ssn"]):
                pii_violations += 1
        pii_violation_rate = float(pii_violations / len(draft_replies))

        return {
            "intent_classification": {
                "accuracy": float(intent_acc),
                "macro_f1": macro_f1,
                "macro_precision": macro_prec,
                "macro_recall": macro_rec,
                "per_intent": per_intent_metrics,
                "confusion_matrix": {
                    "labels": intent_labels,
                    "matrix": conf_matrix
                }
            },
            "escalation_routing": {
                "accuracy": float(esc_acc),
                "precision": float(esc_p[0]) if hasattr(esc_p, '__len__') else float(esc_p),
                "recall": float(esc_r[0]) if hasattr(esc_r, '__len__') else float(esc_r),
                "f1": float(esc_f1[0]) if hasattr(esc_f1, '__len__') else float(esc_f1),
                "false_negative_rate": fnr,
                "false_negatives": false_negatives,
                "confusion_matrix": cm_esc.tolist()
            },
            "reply_quality": {
                "mean_semantic_similarity": mean_cosine_sim,
                "char_length_mean": float(np.mean(char_lengths)),
                "compliance_280_chars": compliance_280,
                "pii_violation_rate": pii_violation_rate
            }
        }
