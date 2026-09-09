# Golden Evaluation Set: Sampling & Labelling Methodology

## 1. Overview & Dataset Scope
- **Brand**: `@AmazonHelp` (Official Amazon Customer Support on Twitter)
- **Primary Source**: Kaggle `thoughtvector/customer-support-on-twitter` (`twcs.csv`)
- **Total Examples**: 200 curated and hand-verified first-contact customer queries
- **Strict Leakage Prevention**: All 200 evaluation examples were sampled exclusively from rows 600,000+ of `twcs.csv`, completely disjoint from the 5,000 historical resolution exemplars in `data/amazon_knowledge_base.jsonl`. No tweet IDs or customer texts overlap.

## 2. Intent Taxonomy
Derived empirically from lexical clustering, topic modeling, and frequent support request categories in the Twitter stream:
1. `ORDER_DELIVERY_TRACKING` (n=45): Package whereabouts, courier delivery delays, ETA questions, missed delivery attempts, false delivered scans.
2. `RETURN_REFUND_REPLACEMENT` (n=38): Return initiation, damaged in transit, defective goods, replacement requests, refund processing delays.
3. `ACCOUNT_BILLING_SECURITY` (n=35): Unauthorized credit card charges, Prime membership billing, account lockouts, 2FA/password reset.
4. `SERVICE_COMPLAINT_FEEDBACK` (n=34): Severe dissatisfaction, delivery driver misconduct, damaged property, unhelpful representatives, complaints.
5. `PRODUCT_DEVICE_TECHNICAL` (n=28): Amazon Echo/Alexa troubleshooting, Kindle reader bugs, Fire TV app crashes, Prime Video sync issues.
6. `GENERAL_INQUIRY_POLICY` (n=20): Standard return windows, student discounts, warranty rules, service availability.

## 3. Escalation Decision Boundaries (Ground Truth)
- **`AUTO_HANDLE` (82 / 200, 41.0%)**:
  The issue can be safely and completely resolved through public Twitter replies by providing official self-service workflows (e.g. `amazon.com/returns`, 'Your Orders' tracking links, standard device reboot procedures). No private customer PII or account database mutation is required.
- **`ESCALATE_HUMAN` (118 / 200, 59.0%)**:
  The issue strictly requires human intervention and private channel escalation (directing to Amazon's secure authentication/chat portal). Triggers include:
  1. Financial/refund mutations or payment disputes.
  2. Account security / unauthorized access / account closure.
  3. Physical safety / driver property damage / stolen merchandise.
  4. Extreme customer frustration, repeat failures, or legal/regulatory threats.

## 4. Labelling Quality & Verification
Every example includes:
- `customer_text`: Raw incoming tweet text.
- `customer_text_clean`: Normalized text with handles and URLs stripped for fair embedding evaluation.
- `reference_reply`: The authentic historical response posted by an `@AmazonHelp` agent on Twitter.
- `ground_truth_intent`: Verified intent category.
- `ground_truth_escalation`: Binary routing decision (`AUTO_HANDLE` vs `ESCALATE_HUMAN`).
- `escalation_reason_category`: Standardized reason code.
- `escalation_reason_detail`: Explicit rationale explaining why the routing was chosen.
