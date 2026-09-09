"""
Unified Amazon Customer Support Agent.
End-to-end pipeline combining intent classification, dense retrieval over historical resolutions,
calibrated escalation routing with stated reason, and grounded response synthesis.
"""

from typing import Dict, Any, Optional
from src.agent.intent_classifier import IntentClassifier
from src.agent.retriever import HistoricalRetriever
from src.agent.escalation_engine import EscalationEngine
from src.agent.response_generator import ResponseGenerator


class AmazonSupportAgent:
    def __init__(
        self,
        classifier: Optional[IntentClassifier] = None,
        retriever: Optional[HistoricalRetriever] = None,
        escalation_engine: Optional[EscalationEngine] = None,
        generator: Optional[ResponseGenerator] = None
    ):
        self.classifier = classifier or IntentClassifier()
        self.retriever = retriever or HistoricalRetriever()
        self.escalation_engine = escalation_engine or EscalationEngine()
        self.generator = generator or ResponseGenerator()

    def process_message(self, customer_text: str, username: str = "") -> Dict[str, Any]:
        """
        Processes an incoming customer message and produces:
        - predicted intent & confidence score
        - retrieved historical exemplar resolutions
        - escalation decision & stated reason
        - drafted grounded response
        """
        # Step 1: Intent Classification
        intent, confidence, intent_scores = self.classifier.predict(customer_text)

        # Step 2: Dense Retrieval of Historical Resolutions
        exemplars = self.retriever.search(customer_text, top_k=3)

        # Step 3: Escalation Decision & Stated Reason
        escalation, reason_cat, reason_detail = self.escalation_engine.evaluate(
            customer_text=customer_text,
            predicted_intent=intent,
            confidence=confidence,
            intent_scores=intent_scores
        )

        # Step 4: Grounded Response Drafting
        reply = self.generator.draft_reply(
            customer_text=customer_text,
            intent=intent,
            escalation=escalation,
            escalation_reason=reason_cat,
            retrieved_exemplars=exemplars,
            username=username
        )

        return {
            "intent": intent,
            "confidence": float(confidence),
            "intent_scores": intent_scores,
            "escalation_decision": escalation,
            "escalation_reason_category": reason_cat,
            "escalation_reason_detail": reason_detail,
            "drafted_reply": reply,
            "retrieved_exemplars": [
                {
                    "pair_id": ex["pair_id"],
                    "similarity": ex.get("similarity_score", 0.0),
                    "historical_customer": ex.get("customer_text_clean", ""),
                    "historical_reply": ex.get("reply_text_clean", "")
                }
                for ex in exemplars
            ]
        }
