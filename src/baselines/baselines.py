"""
Baselines for comparative evaluation:
1. TrivialBaseline: Majority-class intent + static canned template + constant escalation.
2. SimpleBaseline: TF-IDF + Logistic Regression intent + 1-NN historical verbatim copy-paste + keyword heuristic escalation.
"""

import re
import json
from pathlib import Path
from typing import Dict, Any, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


KB_FILE = Path("data/amazon_knowledge_base.jsonl")


class TrivialBaseline:
    """
    A trivial baseline that:
    - Always predicts the majority intent ('ORDER_DELIVERY_TRACKING').
    - Returns a static, generic canned response.
    - Always escalates to human agents.
    """
    def __init__(self):
        self.majority_intent = "ORDER_DELIVERY_TRACKING"
        self.canned_reply = "Thanks for reaching out! Please send us a direct message with your details so we can assist you. ^CS"

    def process_message(self, customer_text: str, username: str = "") -> Dict[str, Any]:
        return {
            "intent": self.majority_intent,
            "confidence": 1.0,
            "escalation_decision": "ESCALATE_HUMAN",
            "escalation_reason_category": "TRIVIAL_DEFAULT_POLICY",
            "escalation_reason_detail": "Trivial baseline applies a blanket escalation rule to all inbound traffic.",
            "drafted_reply": self.canned_reply,
            "retrieved_exemplars": []
        }


class SimpleBaseline:
    """
    A simple baseline that:
    - Uses TF-IDF + Logistic Regression on synthetic exemplar pairs for intent classification.
    - Uses 1-NN TF-IDF retrieval to directly copy-paste the raw historical tweet without adaptation.
    - Uses a simple keyword-matching heuristic for escalation decisions.
    """
    def __init__(self, max_docs: int = 2000):
        self.documents = []
        self._load_data(max_docs)
        self._fit_models()

        self.escalation_keywords = [
            "refund", "cancel", "manager", "supervisor", "fraud", "stole", "stolen",
            "lawyer", "police", "charged", "hacked", "court", "complaint", "call me", "phone", "unauthorized"
        ]

    def _load_data(self, max_docs: int):
        if not KB_FILE.exists():
            raise FileNotFoundError(f"Knowledge base file {KB_FILE} not found.")
        with open(KB_FILE, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= max_docs:
                    break
                self.documents.append(json.loads(line))

    def _fit_models(self):
        # Build TF-IDF search index over knowledge base
        texts = [doc["customer_text_clean"] for doc in self.documents]
        self.tfidf_vectorizer = TfidfVectorizer(max_features=2500, stop_words="english")
        self.doc_vectors = self.tfidf_vectorizer.fit_transform(texts)

        # Train simple keyword-based intent classifier using exemplars
        from src.agent.intent_classifier import INTENT_EXEMPLARS
        train_texts = []
        train_labels = []
        for intent, examples in INTENT_EXEMPLARS.items():
            for ex in examples:
                train_texts.append(ex)
                train_labels.append(intent)

        self.intent_vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
        X = self.intent_vectorizer.fit_transform(train_texts)
        self.intent_clf = LogisticRegression(C=1.0, max_iter=200)
        self.intent_clf.fit(X, train_labels)

    def process_message(self, customer_text: str, username: str = "") -> Dict[str, Any]:
        # 1. Intent via Logistic Regression
        X_test = self.intent_vectorizer.transform([customer_text])
        probs = self.intent_clf.predict_proba(X_test)[0]
        classes = self.intent_clf.classes_
        intent = classes[probs.argmax()]
        confidence = float(probs.max())

        # 2. Retrieval via 1-NN TF-IDF (copy-paste raw tweet)
        q_vec = self.tfidf_vectorizer.transform([customer_text])
        scores = (self.doc_vectors * q_vec.T).toarray().ravel()
        best_idx = int(scores.argmax())
        best_doc = self.documents[best_idx]
        raw_retrieved_reply = best_doc.get("reply_text", "")

        # 3. Keyword-heuristic escalation
        text_lower = customer_text.lower()
        matched_kw = [kw for kw in self.escalation_keywords if kw in text_lower]
        if matched_kw:
            decision = "ESCALATE_HUMAN"
            reason_cat = "KEYWORD_HEURISTIC_TRIGGERED"
            reason_detail = f"Matched escalation keywords: {', '.join(matched_kw)}"
        else:
            decision = "AUTO_HANDLE"
            reason_cat = "NO_ESCALATION_KEYWORDS"
            reason_detail = "No high-risk keywords detected in customer message."

        return {
            "intent": intent,
            "confidence": confidence,
            "escalation_decision": decision,
            "escalation_reason_category": reason_cat,
            "escalation_reason_detail": reason_detail,
            "drafted_reply": raw_retrieved_reply,
            "retrieved_exemplars": [{
                "pair_id": best_doc.get("pair_id"),
                "similarity": float(scores[best_idx]),
                "historical_customer": best_doc.get("customer_text_clean", ""),
                "historical_reply": best_doc.get("reply_text_clean", "")
            }]
        }
