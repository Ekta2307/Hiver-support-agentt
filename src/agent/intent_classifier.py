"""
Semantic Intent Classifier for Amazon Customer Support.
Classifies customer messages into 6 empirical intents using dense semantic embeddings
(all-MiniLM-L6-v2) and calibrated similarity scoring.
"""

import re
import numpy as np
from typing import Dict, Tuple, Any


INTENT_LABELS = [
    "ORDER_DELIVERY_TRACKING",
    "RETURN_REFUND_REPLACEMENT",
    "ACCOUNT_BILLING_SECURITY",
    "PRODUCT_DEVICE_TECHNICAL",
    "SERVICE_COMPLAINT_FEEDBACK",
    "GENERAL_INQUIRY_POLICY"
]

# Canonical intent prototypes derived from training data
INTENT_EXEMPLARS = {
    "ORDER_DELIVERY_TRACKING": [
        "Where is my package? It was supposed to be delivered today but tracking hasn't updated.",
        "Carrier says delivered to front porch but I was home all day and received nothing.",
        "My shipment is delayed for over a week at the local courier facility. What is the ETA?",
        "Tracking status says out for delivery since morning, when will it arrive?",
        "Package marked as handed to resident but no one knocked on my door.",
        "Order status stuck on preparing for dispatch for 5 days."
    ],
    "RETURN_REFUND_REPLACEMENT": [
        "I received the item damaged and broken in pieces. I want a replacement.",
        "How do I return this defective product and get my money back?",
        "I sent back my return package last week but I still haven't received my refund.",
        "Wrong item delivered in my box. I ordered headphones and got a cable.",
        "Can I exchange this shirt for a different size?",
        "The box was ripped and two items were missing from my order."
    ],
    "ACCOUNT_BILLING_SECURITY": [
        "I was charged twice on my credit card for an order I canceled.",
        "My Amazon account is locked and I cannot sign in or reset my password.",
        "Unauthorized Prime subscription fee charged to my bank account without permission.",
        "I keep getting 2-step verification codes but I am not trying to log in. Was I hacked?",
        "Please cancel my Amazon payments account and delete my card details.",
        "Why did my payment decline when my card has sufficient balance?"
    ],
    "PRODUCT_DEVICE_TECHNICAL": [
        "My Amazon Echo Dot is unresponsive and showing a spinning red ring.",
        "Kindle Paperwhite screen is frozen and will not wake up from sleep mode.",
        "Fire TV Stick keeps buffering and throwing audio sync error on Prime Video.",
        "Alexa app is crashing every time I try to connect a new smart plug.",
        "How do I deregister my Kindle device and transfer books to a new owner?",
        "Prime Video error code 5004 on my Samsung Smart TV."
    ],
    "SERVICE_COMPLAINT_FEEDBACK": [
        "Your delivery driver threw my fragile package over a 6-foot gate and broke it!",
        "Customer service agent was extremely rude and hung up the phone on me.",
        "Worst customer experience of my life, lied to three times by support reps.",
        "I want to speak with a senior supervisor or manager immediately.",
        "Terrible service, I will never buy from Amazon again and filing a complaint.",
        "Delivery driver left package in the rain on the sidewalk near the street."
    ],
    "GENERAL_INQUIRY_POLICY": [
        "What is the return window policy for holiday gift purchases?",
        "Do you offer Amazon Prime student discounts for international students?",
        "Can I use an Amazon gift card to pay for a Prime membership subscription?",
        "When is Prime Wardrobe or same day delivery coming to my zip code?",
        "How do manufacturer warranties work for third-party sellers on Amazon?",
        "Is international shipping available to Australia for electronics?"
    ]
}

# Domain keyword boosts for high-precision boundary resolution
KEYWORD_BOOSTS = {
    "ACCOUNT_BILLING_SECURITY": ["charged", "password", "login", "locked", "hacked", "unauthorized", "billing", "credit card", "bank account", "prime fee", "subscription fee", "twice"],
    "RETURN_REFUND_REPLACEMENT": ["return", "refund", "replace", "replacement", "broken", "damaged", "defective", "wrong item", "missing item", "send back", "exchange"],
    "ORDER_DELIVERY_TRACKING": ["delivery", "delivered", "tracking", "track", "carrier", "courier", "package", "shipment", "shipped", "arrived", "late", "eta", "dispatch"],
    "PRODUCT_DEVICE_TECHNICAL": ["echo", "alexa", "kindle", "fire tv", "firestick", "app", "sync", "buffer", "bluetooth", "wifi", "reboot", "frozen", "screen"],
    "SERVICE_COMPLAINT_FEEDBACK": ["worst", "rude", "hung up", "manager", "supervisor", "threw", "thrown", "terrible", "disgusting", "horrible", "unacceptable", "lawsuit", "complaint"],
    "GENERAL_INQUIRY_POLICY": ["policy", "discount", "gift card", "warranty", "when will", "is it possible", "how do i", "eligible", "international shipping"]
}


class IntentClassifier:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._encode_exemplars()

    def _encode_exemplars(self):
        self.exemplar_embeddings = {}
        for intent, texts in INTENT_EXEMPLARS.items():
            embs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            # Store normalized mean centroid
            centroid = np.mean(embs, axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            self.exemplar_embeddings[intent] = centroid

    def clean_text(self, text: str) -> str:
        text = re.sub(r"@[A-Za-z0-9_]+", "", text)
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def predict(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        cleaned = self.clean_text(text)
        if not cleaned:
            return "GENERAL_INQUIRY_POLICY", 0.0, {label: 1.0 / len(INTENT_LABELS) for label in INTENT_LABELS}

        query_emb = self.model.encode([cleaned], normalize_embeddings=True, show_progress_bar=False)[0]

        raw_scores = {}
        cleaned_lower = cleaned.lower()

        for intent in INTENT_LABELS:
            centroid = self.exemplar_embeddings[intent]
            cos_sim = float(np.dot(query_emb, centroid))

            # Apply domain keyword boost
            boost = 0.0
            keywords = KEYWORD_BOOSTS.get(intent, [])
            for kw in keywords:
                if re.search(r"\b" + re.escape(kw) + r"\b", cleaned_lower):
                    boost += 0.06

            raw_scores[intent] = cos_sim + boost

        # Softmax normalization with temperature
        temp = 0.15
        exp_scores = {k: np.exp(v / temp) for k, v in raw_scores.items()}
        sum_exp = sum(exp_scores.values())
        probs = {k: float(v / sum_exp) for k, v in exp_scores.items()}

        best_intent = max(probs, key=probs.get)
        confidence = probs[best_intent]

        return best_intent, confidence, probs
