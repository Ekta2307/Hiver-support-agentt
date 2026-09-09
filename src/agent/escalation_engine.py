"""
Calibrated Multi-Factor Escalation Engine for Amazon Customer Support.
Decides whether an incoming customer message should be auto-handled or escalated to a human agent,
providing a clear, structured stated reason.
"""

import re
from typing import Dict, Tuple, Any


class EscalationEngine:
    def __init__(self, confidence_threshold: float = 0.35):
        self.confidence_threshold = confidence_threshold

    def evaluate(
        self,
        customer_text: str,
        predicted_intent: str,
        confidence: float,
        intent_scores: Dict[str, float]
    ) -> Tuple[str, str, str]:
        """
        Returns:
            decision: "AUTO_HANDLE" or "ESCALATE_HUMAN"
            reason_category: Standardized enum reason
            reason_detail: Human-readable justification for the routing decision
        """
        text_lower = customer_text.lower()

        # 1. Critical Escalation: Safety / Driver Misconduct / Property Damage
        driver_misconduct = ["threw", "thrown", "fence", "lawn", "damaged property", "smashed", "stole", "driver was rude", "reckless"]
        if any(w in text_lower for w in driver_misconduct):
            return (
                "ESCALATE_HUMAN",
                "SAFETY_OR_DRIVER_MISCONDUCT",
                "Involves physical property damage, driver misconduct, or compromised delivery requiring carrier investigation."
            )

        # 2. Critical Escalation: PII, Account Lockout, Security, or Payment Mutation
        account_triggers = [
            "password", "login", "locked", "sign in", "unauthorized", "hacked",
            "charged twice", "double charged", "bank account", "credit card", "close my account", "cancel prime"
        ]
        if predicted_intent == "ACCOUNT_BILLING_SECURITY" or any(w in text_lower for w in account_triggers):
            return (
                "ESCALATE_HUMAN",
                "PII_OR_ACCOUNT_MUTATION_REQUIRED",
                "Requires private authentication, sensitive account lookup, or payment database mutation not permitted on public Twitter."
            )

        # 3. Critical Escalation: Financial Dispute / Compensation Demands
        financial_triggers = ["refund me", "where is my money", "compensate", "credit my card", "money back", "deducted", "lost £", "lost $"]
        if any(w in text_lower for w in financial_triggers):
            return (
                "ESCALATE_HUMAN",
                "FINANCIAL_DISPUTE_OR_REFUND_ACTION",
                "Customer demands financial reimbursement, refund status intervention, or monetary compensation."
            )

        # 4. Critical Escalation: Severe Frustration / Repeat Contact / Legal or Manager Demands
        frustration_triggers = [
            "talk to manager", "supervisor", "worst service", "hung up", "lawsuit", "consumer court",
            "3 times", "third time", "days ago and nothing", "still waiting", "useless", "pathetic",
            "lying", "fraud", "scam", "unacceptable"
        ]
        if predicted_intent == "SERVICE_COMPLAINT_FEEDBACK" or any(w in text_lower for w in frustration_triggers):
            return (
                "ESCALATE_HUMAN",
                "REPEATED_FAILURE_OR_HIGH_FRUSTRATION",
                "High customer distress, repeat unresolved contacts, or demand for supervisory escalation."
            )

        # 5. Missing / Stolen Package with "Delivered" claim
        stolen_package_triggers = ["says delivered but", "marked delivered", "stolen", "opened and empty", "empty box"]
        if any(w in text_lower for w in stolen_package_triggers):
            return (
                "ESCALATE_HUMAN",
                "PACKAGE_STOLEN_OR_CARRIER_INCIDENT",
                "Disputed delivery where carrier reports delivered but customer reports missing or tampered contents."
            )

        # 6. Low Confidence / Ambiguity Guardrail
        if confidence < self.confidence_threshold:
            return (
                "ESCALATE_HUMAN",
                "LOW_CONFIDENCE_AMBIGUOUS_QUERY",
                f"Classification confidence ({confidence:.2f}) is below threshold ({self.confidence_threshold:.2f}). Safe fallback to human triage."
            )

        # 7. Safe Auto-Handle: Basic Technical / App Troubleshooting
        if predicted_intent == "PRODUCT_DEVICE_TECHNICAL":
            return (
                "AUTO_HANDLE",
                "BASIC_DEVICE_TROUBLESHOOTING",
                "Standard device or streaming troubleshooting can be resolved with self-service restart or settings guidance."
            )

        # 8. Safe Auto-Handle: Standard Return Policy Guidance
        if predicted_intent == "RETURN_REFUND_REPLACEMENT" and any(w in text_lower for w in ["how to return", "return window", "drop off", "return policy"]):
            return (
                "AUTO_HANDLE",
                "STANDARD_RETURN_POLICY_INFORMATION",
                "Routine return inquiry addressable via standard Online Returns Center instructions."
            )

        # 9. Safe Auto-Handle: Tracking Status Guidance
        if predicted_intent == "ORDER_DELIVERY_TRACKING":
            return (
                "AUTO_HANDLE",
                "SELF_SERVICE_TRACKING_GUIDANCE",
                "Routine shipment whereabouts resolvable by guiding customer to 'Your Orders' tracking updates."
            )

        # 10. Safe Auto-Handle: General Policy / Catalog Inquiries
        if predicted_intent == "GENERAL_INQUIRY_POLICY":
            return (
                "AUTO_HANDLE",
                "GENERAL_POLICY_INFORMATION",
                "General policy, warranty, or feature availability resolvable via public Amazon guidelines."
            )

        # Default fallback: safe auto-handle if no escalation criteria triggered
        return (
            "AUTO_HANDLE",
            "SELF_SERVICE_RESOLVABLE",
            "Query matches standard support workflows and does not require account mutation or escalation."
        )
