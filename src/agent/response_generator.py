"""
Grounded Response Generator for Amazon Customer Support.
Synthesizes customer service replies grounded in historical AmazonHelp resolution exemplars,
strictly adhering to Amazon brand guidelines (empathy, privacy guardrails, actionable links, ^AI signature).
"""

import os
import re
from typing import List, Dict, Any, Optional


OFFICIAL_LINKS = {
    "TRACKING": "https://amzn.to/track-package",
    "RETURNS": "https://amzn.to/returns-center",
    "CONTACT_SECURE": "https://amzn.to/contact-us",
    "DEVICE_HELP": "https://amzn.to/device-support",
    "GENERAL_HELP": "https://amzn.to/amazon-help"
}


class ResponseGenerator:
    def __init__(self):
        self.openai_key = os.environ.get("OPENAI_API_KEY")
        self.gemini_key = os.environ.get("GEMINI_API_KEY")
        self.groq_key = os.environ.get("GROQ_API_KEY")

    def draft_reply(
        self,
        customer_text: str,
        intent: str,
        escalation: str,
        escalation_reason: str,
        retrieved_exemplars: List[Dict[str, Any]],
        username: str = ""
    ) -> str:
        """
        Drafts a grounded reply. Uses LLM API if key is available,
        otherwise uses high-fidelity exemplar-grounded RAG synthesis.
        """
        if (self.openai_key or self.gemini_key or self.groq_key):
            try:
                llm_reply = self._generate_with_llm(customer_text, intent, escalation, retrieved_exemplars, username)
                if llm_reply:
                    return llm_reply
            except Exception:
                pass # Fall back to exemplar grounding

        return self._generate_exemplar_grounded(customer_text, intent, escalation, escalation_reason, retrieved_exemplars, username)

    def _generate_exemplar_grounded(
        self,
        customer_text: str,
        intent: str,
        escalation: str,
        escalation_reason: str,
        retrieved_exemplars: List[Dict[str, Any]],
        username: str
    ) -> str:
        user_prefix = f"@{username} " if username else ""
        top_exemplar = retrieved_exemplars[0] if retrieved_exemplars else None
        
        # Empathy openings tailored by intent
        empathy_map = {
            "ORDER_DELIVERY_TRACKING": "I'm sorry to hear your package hasn't arrived as expected.",
            "RETURN_REFUND_REPLACEMENT": "I'm sorry your item arrived in less than perfect condition.",
            "ACCOUNT_BILLING_SECURITY": "I understand your concern regarding this account charge.",
            "PRODUCT_DEVICE_TECHNICAL": "I'm sorry you're having trouble with your device.",
            "SERVICE_COMPLAINT_FEEDBACK": "I apologize for the frustrating experience you've had with our service.",
            "GENERAL_INQUIRY_POLICY": "Thanks for reaching out to us!"
        }

        opening = empathy_map.get(intent, "Thanks for reaching out to Amazon Support.")

        # Grounding from historical exemplar
        exemplar_advice = ""
        if top_exemplar:
            ex_reply = top_exemplar.get("reply_text_clean", "")
            # Extract useful advice snippet from exemplar if present
            if "restart" in ex_reply.lower():
                exemplar_advice = "As a quick check, please try power-cycling the device and verifying your connection."
            elif "24-48 hours" in ex_reply.lower() or "allow" in ex_reply.lower():
                exemplar_advice = "Please allow 24-48 business hours for tracking updates to populate."
            elif "courier" in ex_reply.lower() or "carrier" in ex_reply.lower():
                exemplar_advice = "Couriers occasionally scan items slightly ahead of physical delivery."

        # Escalation routing actions
        if escalation == "ESCALATE_HUMAN":
            action = (
                "For your privacy and security, we cannot access private account details over Twitter. "
                "Please connect directly with our specialized support team via our secure portal so we can investigate right away: "
                f"{OFFICIAL_LINKS['CONTACT_SECURE']}"
            )
        else:
            # Auto-handle resolution actions
            if intent == "ORDER_DELIVERY_TRACKING":
                action = (
                    f"{exemplar_advice} You can review real-time delivery status and driver notes directly under 'Your Orders': "
                    f"{OFFICIAL_LINKS['TRACKING']}"
                )
            elif intent == "RETURN_REFUND_REPLACEMENT":
                action = (
                    "You can easily initiate a replacement or return label through our Online Returns Center: "
                    f"{OFFICIAL_LINKS['RETURNS']}"
                )
            elif intent == "PRODUCT_DEVICE_TECHNICAL":
                action = (
                    f"{exemplar_advice} For step-by-step device troubleshooting guides, please visit: "
                    f"{OFFICIAL_LINKS['DEVICE_HELP']}"
                )
            elif intent == "GENERAL_INQUIRY_POLICY":
                action = (
                    "You can find full details on our policies, delivery options, and benefits in our Help Directory: "
                    f"{OFFICIAL_LINKS['GENERAL_HELP']}"
                )
            else:
                action = (
                    f"You can find self-service options and FAQs in our Customer Service portal: "
                    f"{OFFICIAL_LINKS['GENERAL_HELP']}"
                )

        closing = "^AI"
        reply = f"{user_prefix}{opening} {action} {closing}".strip()
        # Clean double spaces
        reply = re.sub(r"\s+", " ", reply)
        return reply

    def _generate_with_llm(
        self,
        customer_text: str,
        intent: str,
        escalation: str,
        retrieved_exemplars: List[Dict[str, Any]],
        username: str
    ) -> Optional[str]:
        # Pluggable LLM caller when external API key is provided
        prompt = (
            f"You are @AmazonHelp customer support on Twitter. Draft a professional reply to this tweet:\n"
            f"Customer Tweet: {customer_text}\n"
            f"Intent: {intent}\n"
            f"Escalation Decision: {escalation}\n\n"
            f"Grounding Historical Exemplars:\n"
        )
        for i, ex in enumerate(retrieved_exemplars[:2], 1):
            prompt += f"Exemplar {i}: Customer: {ex.get('customer_text_clean', '')} | Amazon Reply: {ex.get('reply_text_clean', '')}\n"
        prompt += (
            "\nRules:\n"
            "1. Be empathetic and professional in Amazon's tone.\n"
            "2. Never ask for or accept PII (passwords, card info) on Twitter.\n"
            "3. If ESCALATE_HUMAN, direct them to contact securely via https://amzn.to/contact-us.\n"
            "4. If AUTO_HANDLE, give clear self-service guidance and link.\n"
            "5. Keep under 280 characters.\n"
            "6. End with signature ^AI."
        )
        # We check Groq / OpenAI if configured
        if self.groq_key:
            import httpx
            res = httpx.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.groq_key}"},
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 120,
                    "temperature": 0.2
                },
                timeout=5.0
            )
            if res.status_code == 200:
                return res.json()["choices"][0]["message"]["content"].strip()
        return None
