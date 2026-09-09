# Hiver SDE Intern Take-Home Assignment: Amazon Customer Support AI Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Benchmark Runtime](https://img.shields.io/badge/Benchmark_Runtime-<2_mins-brightgreen.svg)]()
[![Evaluated On](https://img.shields.io/badge/Golden_Eval_Set-200_cases-orange.svg)]()

> An autonomous, safety-calibrated AI customer support agent for **@AmazonHelp** built on real-world Twitter customer service conversations (`thoughtvector/customer-support-on-twitter`).
> The agent classifies inbound customer intent, retrieves historical resolution exemplars, decides escalation routing (`AUTO_HANDLE` vs. `ESCALATE_HUMAN`) with structured reasons, and drafts grounded, empathetic responses adhering to brand guidelines.

---

## ⚡ Reproduce Headline Results in Under 2 Minutes

To reproduce the benchmark comparison across Baseline 1, Baseline 2, and the Proposed Agent on the 200-example Golden Evaluation Set:

```bash
# 1. Clone the repository
git clone https://github.com/Ekta2307/Hiver-support-agentt.git
cd Hiver-support-agentt

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the master evaluation pipeline
python run_pipeline.py
```

*Expected output: Full benchmark metrics table, critical safety FNR comparison, and human-judge statistical correlation in **under 90 seconds** on standard CPU.*

---

## 📊 Headline Benchmark Results

Evaluated on the **Golden Evaluation Set** ($N = 200$ hand-annotated, stratified cases held-out from the historical knowledge base):

| System | Intent Accuracy | Intent Macro F1 | Escalation Accuracy | Escalation F1 | Escalation FNR (Missed Escalations) | Reply Semantic Similarity | Judge Groundedness (1-5) | Judge Actionability (1-5) | Judge Overall (1-5) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1 (Trivial)** | 22.5% | 0.061 | 59.0% | 0.742 | 0.0% (0 / 118) | 0.360 | 2.00 | 3.00 | 2.65 |
| **Baseline 2 (Simple)** | 60.0% | 0.578 | 60.5% | 0.586 | 52.5% (62 / 118) | 0.338 | 3.31 | 3.33 | 3.34 |
| **Proposed Agent** | **70.5%** | **0.638** | **75.0%** | **0.813** | **7.6% (9 / 118)** | 0.318 | **4.98** | **4.93** | **4.77** |

### Critical Safety Finding: Escalation False Negative Rate (FNR)
- **Baseline 2 (Simple)** missed **62 out of 118 true escalations (52.5% FNR)**, leaving customers with stolen goods, double billing, or account security issues abandoned.
- **Proposed Agent** reduced missed escalations to **9 cases (7.6% FNR)**, achieving high customer safety while still automating 41% of benign routine inquiries.

### Human-Judge Agreement Evidence (n=40 Calibration Set)
- **Pearson Correlation ($r$):** **0.641** ($p = 8.23 \times 10^{-6}$)
- **Spearman Rank Correlation ($\rho$):** **0.606** ($p = 3.42 \times 10^{-5}$)
- **Safety / PII Correlation:** **$r = 1.000$**
- **Actionability Correlation:** **$r = 0.804$**
- **Mean Absolute Error (MAE):** **0.925 points** on a 1–5 scale

---

## 🎮 Interactive Live Demo

Test arbitrary customer messages through either the CLI or Web UI:

### CLI Demo
```bash
python demo.py
```
*Allows selecting real pre-loaded customer scenarios or typing custom tweets.*

### Web UI Dashboard
```bash
python demo.py --web
```
*Launches an interactive web app at `http://localhost:8000` showing real-time intent confidence, escalation badges, stated reasons, drafted responses, and matched historical exemplars.*

---

## 🏗️ Repository Architecture

```
Hiver-support-agentt/
├── README.md                           # Quickstart, headline results, and architectural overview
├── REPORT.md                           # Standalone 6-page comprehensive technical report
├── requirements.txt                    # Project dependencies
├── run_pipeline.py                     # Master benchmark script (<90s runtime)
├── demo.py                             # Interactive CLI & Web UI demo
│
├── data/
│   ├── golden_eval_set.json            # 200 hand-labelled evaluation cases with full metadata
│   ├── golden_eval_set.csv             # CSV version of the golden set
│   ├── golden_set_sampling_note.md     # Sampling & labelling methodology documentation
│   ├── human_judge_calibration.json    # 40 hand-scored cases for LLM-as-judge correlation
│   ├── amazon_knowledge_base.jsonl     # 5,000 indexed historical Amazon resolution exemplars
│   └── kb_embeddings.npy               # Cached dense vector embeddings for instantaneous loading
│
├── src/
│   ├── agent/
│   │   ├── intent_classifier.py        # Dense semantic intent classifier (all-MiniLM-L6-v2)
│   │   ├── retriever.py                # Top-k historical exemplar retriever
│   │   ├── escalation_engine.py        # Calibrated multi-factor routing engine + stated reasons
│   │   ├── response_generator.py       # Grounded RAG reply synthesizer with brand guardrails
│   │   └── agent.py                    # Unified orchestrator pipeline
│   │
│   ├── baselines/
│   │   └── baselines.py                # TrivialBaseline (majority) & SimpleBaseline (TF-IDF + 1-NN)
│   │
│   ├── eval/
│   │   ├── automated_metrics.py        # Accuracy, F1, FNR, Cosine Sim, Length, PII checks
│   │   └── judge.py                    # Multi-rubric judge + human calibration correlation
│   │
│   └── data/
│       ├── build_knowledge_base.py     # Ingests twcs.csv and extracts AmazonHelp exemplars
│       ├── build_golden_set.py         # Stratified sampling of 200 held-out evaluation cases
│       └── build_calibration_set.py   # Constructs 40-case human calibration set
│
└── results/
    └── benchmark_results.json          # Machine-readable output of all benchmark runs
```

---

## 📖 Intent Taxonomy & Escalation Boundaries

### 1. Intent Taxonomy (Derived from Data)
1. `ORDER_DELIVERY_TRACKING` (n=45): Package whereabouts, tracking delays, missed deliveries, false delivery scans.
2. `RETURN_REFUND_REPLACEMENT` (n=38): Return requests, damaged goods, missing items, replacement orders, refund status.
3. `ACCOUNT_BILLING_SECURITY` (n=35): Unauthorized credit card charges, Prime subscription fees, account lockouts, password reset.
4. `SERVICE_COMPLAINT_FEEDBACK` (n=34): Severe dissatisfaction, delivery driver misconduct, damaged property, unhelpful phone reps.
5. `PRODUCT_DEVICE_TECHNICAL` (n=28): Amazon Echo/Alexa setup, Kindle reader bugs, Fire TV app buffering, Prime Video playback.
6. `GENERAL_INQUIRY_POLICY` (n=20): Return window policies, student discounts, international delivery guidelines, warranties.

### 2. Escalation Decision Matrix
| Escalation Decision | Primary Stated Reason | Typical Triggers |
| :--- | :--- | :--- |
| `ESCALATE_HUMAN` | `PII_OR_ACCOUNT_MUTATION_REQUIRED` | Password reset, unauthorized charge, account closure request |
| `ESCALATE_HUMAN` | `FINANCIAL_DISPUTE_OR_REFUND_ACTION` | Monetary compensation demand, double charge dispute, lost high-value item |
| `ESCALATE_HUMAN` | `SAFETY_OR_DRIVER_MISCONDUCT` | Package thrown over gate, damaged property, courier misconduct |
| `ESCALATE_HUMAN` | `REPEATED_FAILURE_OR_HIGH_FRUSTRATION` | Customer tried 3+ times, representative hung up, legal/consumer court threat |
| `AUTO_HANDLE` | `SELF_SERVICE_TRACKING_GUIDANCE` | Routine "where is my order" addressable via 'Your Orders' tracking link |
| `AUTO_HANDLE` | `STANDARD_RETURN_POLICY_INFORMATION` | Routine return inquiries addressable via Online Returns Center |
| `AUTO_HANDLE` | `BASIC_DEVICE_TROUBLESHOOTING` | Standard advice to power-cycle device, check network, or reinstall app |

---

## 🔍 Failure Analysis Summary (Top 5 Failure Modes)

1. **Multi-Intent Collision:** Customer combines a refund request with a Prime subscription cancellation. Handled correctly for escalation, but single-label intent forced to pick one.
2. **Sentiment Drowning:** Intense customer grievance language ("worst service ever") overshadows the functional intent (e.g., installation delay or cashback).
3. **Implicit Follow-Ups (Single-Turn Blindspot):** In multi-turn threads, customer replies ("Yes, I already called the carrier") trigger self-service heuristics because prior turns were unavailable.
4. **Structured Multi-Part Complaints:** Numbered lists of courier violations bypass simple keyword checks by using neutral phrasing.
5. **Over-Escalation on Benign Queries:** Queries mentioning "card" or "Prime" in informational contexts are escalated to prioritize safety over automation rate.

*Full details and real examples in [REPORT.md](REPORT.md).*

---

## 💡 "What is Misleading About My Headline Number?"

- **Single-Turn Horizon:** Evaluates first-contact tweets; accuracy drops on conversational follow-up turns without dialogue context.
- **Asymmetric Loss Concealed by Accuracy:** A 75% accuracy metric treats a $3.50 human queue overhead (False Positive) the same as an abandoned fraud victim (False Negative).
- **Semantic Similarity Metric Gap:** Modernized, secure canonical shortlinks diverge lexically from noisy 2017 historical tweets despite being superior.
- **Selection Bias:** English and length filtering excludes single-word tweets ("Help!") and screenshot-only messages common on live Twitter streams.

---

## 🛠️ Citations & Acknowledgments

- **Primary Dataset:** Kaggle *Customer Support on Twitter* (`thoughtvector/customer-support-on-twitter`), mirrored on Hugging Face (`SunidhiSriram/twcs`).
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` by Hugging Face.
- **Evaluation Methodology:** Multi-rubric LLM-as-judge with Pearson/Spearman inter-rater correlation and Cohen's Kappa calibration.
