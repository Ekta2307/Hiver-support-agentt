"""
Interactive Demo for Amazon Customer Support AI Agent.
Allows testing arbitrary customer messages via CLI or Web UI.
Run CLI mode: python demo.py
Run Web UI mode: python demo.py --web
"""

import sys
import json
import argparse
from src.agent.agent import AmazonSupportAgent


PRESET_QUERIES = [
    ("Order Tracking (Auto-Handle)", "Where is my package? The tracking shows it was dispatched 3 days ago but has not moved."),
    ("Damaged Package / Refund (Escalate)", "My package arrived completely ripped open and the headphones are missing! I want a full refund immediately!"),
    ("Account / Unauthorized Charge (Escalate)", "I see a charge of $119 for Prime on my credit card that I never authorized. Please cancel my account!"),
    ("Echo Technical Issue (Auto-Handle)", "My Amazon Echo Dot has a spinning cyan ring and won't respond to my voice commands. How do I fix it?"),
    ("Driver Misconduct Complaint (Escalate)", "Your driver threw my fragile package over an 8-foot fence onto the concrete and broke it. Worst service ever!"),
    ("Policy Inquiry (Auto-Handle)", "What is the return window policy for holiday gift purchases made in November?")
]


def run_cli():
    print("=" * 70)
    print(" AMAZON SUPPORT AI AGENT - INTERACTIVE CLI DEMO ")
    print("=" * 70)
    print("Initializing agent models (this takes ~3-5 seconds)...")
    agent = AmazonSupportAgent()
    print("Ready!\n")

    print("Choose an option:")
    print("  [1-6] Test a pre-loaded real Twitter customer scenario")
    print("  [7]   Type your own customer tweet")
    print("  [q]   Quit")

    while True:
        choice = input("\nSelect option [1-7 or q]: ").strip()
        if choice.lower() == "q":
            print("Exiting demo.")
            break

        if choice in ["1", "2", "3", "4", "5", "6"]:
            idx = int(choice) - 1
            scenario_name, text = PRESET_QUERIES[idx]
            print(f"\n---> Scenario: {scenario_name}")
            print(f"Customer Tweet: \"{text}\"")
        elif choice == "7":
            text = input("\nEnter customer tweet text: ").strip()
            if not text:
                continue
        else:
            print("Invalid selection.")
            continue

        print("\nProcessing message...")
        result = agent.process_message(text)

        print("\n" + "-" * 60)
        print(f"PREDICTED INTENT      : {result['intent']} (Confidence: {result['confidence'] * 100:.1f}%)")
        print(f"ESCALATION DECISION   : {result['escalation_decision']}")
        print(f"STATED REASON CATEGORY: {result['escalation_reason_category']}")
        print(f"STATED REASON DETAIL  : {result['escalation_reason_detail']}")
        print("-" * 60)
        print(f"DRAFTED REPLY         :\n\"{result['drafted_reply']}\"")
        print("-" * 60)
        if result["retrieved_exemplars"]:
            top_ex = result["retrieved_exemplars"][0]
            print(f"TOP RETRIEVED HISTORICAL EXEMPLAR (Similarity: {top_ex['similarity']:.3f}):")
            print(f"  Customer: \"{top_ex['historical_customer']}\"")
            print(f"  Amazon  : \"{top_ex['historical_reply']}\"")
        print("-" * 60)


def run_web():
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse

    app = FastAPI(title="Amazon Support Agent Demo")
    agent = AmazonSupportAgent()

    HTML_TEMPLATE = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Amazon Customer Support AI Agent</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 30px; }
            .container { max-width: 900px; margin: 0 auto; background: #1e293b; padding: 30px; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
            h1 { color: #38bdf8; margin-top: 0; }
            textarea { width: 100%; height: 90px; background: #0f172a; color: #fff; border: 1px solid #475569; border-radius: 8px; padding: 12px; font-size: 15px; box-sizing: border-box; }
            button { background: #0284c7; color: white; border: none; padding: 12px 24px; border-radius: 6px; font-size: 15px; cursor: pointer; font-weight: bold; margin-top: 10px; }
            button:hover { background: #0369a1; }
            .card { background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 20px; margin-top: 20px; }
            .badge { display: inline-block; padding: 4px 10px; border-radius: 4px; font-size: 13px; font-weight: bold; }
            .badge-auto { background: #15803d; color: #dcfce7; }
            .badge-esc { background: #b91c1c; color: #fee2e2; }
            .badge-intent { background: #1d4ed8; color: #dbeafe; }
            .field-label { color: #94a3b8; font-size: 13px; text-transform: uppercase; font-weight: bold; margin-bottom: 4px; }
            .field-value { font-size: 16px; margin-bottom: 16px; }
            .reply-box { background: #1e1b4b; border-left: 4px solid #818cf8; padding: 14px; border-radius: 4px; font-size: 15px; line-height: 1.5; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>@AmazonHelp Support AI Agent</h1>
            <p style="color: #94a3b8;">Classifies customer intent, retrieves historical resolutions, decides escalation with stated reasons, and drafts grounded replies.</p>
            
            <form method="post" action="/query">
                <div class="field-label">Customer Inbound Tweet</div>
                <textarea name="text" placeholder="Type a customer message here...">Where is my package? Tracking says delivered but I received nothing.</textarea>
                <button type="submit">Process Customer Message</button>
            </form>
        </div>
    </body>
    </html>
    """

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTML_TEMPLATE

    @app.post("/query", response_class=HTMLResponse)
    async def process(request: Request):
        form_data = await request.form()
        text = form_data.get("text", "")
        res = agent.process_message(text)

        esc_badge = f'<span class="badge badge-esc">ESCALATE TO HUMAN</span>' if res["escalation_decision"] == "ESCALATE_HUMAN" else f'<span class="badge badge-auto">AUTO-HANDLE</span>'
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Amazon Customer Support AI Agent</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 30px; }}
                .container {{ max-width: 900px; margin: 0 auto; background: #1e293b; padding: 30px; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }}
                h1 {{ color: #38bdf8; margin-top: 0; }}
                textarea {{ width: 100%; height: 90px; background: #0f172a; color: #fff; border: 1px solid #475569; border-radius: 8px; padding: 12px; font-size: 15px; box-sizing: border-box; }}
                button {{ background: #0284c7; color: white; border: none; padding: 12px 24px; border-radius: 6px; font-size: 15px; cursor: pointer; font-weight: bold; margin-top: 10px; }}
                .card {{ background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 20px; margin-top: 20px; }}
                .badge {{ display: inline-block; padding: 4px 10px; border-radius: 4px; font-size: 13px; font-weight: bold; margin-right: 8px; }}
                .badge-auto {{ background: #15803d; color: #dcfce7; }}
                .badge-esc {{ background: #b91c1c; color: #fee2e2; }}
                .badge-intent {{ background: #1d4ed8; color: #dbeafe; }}
                .field-label {{ color: #94a3b8; font-size: 13px; text-transform: uppercase; font-weight: bold; margin-top: 14px; margin-bottom: 4px; }}
                .field-value {{ font-size: 15px; }}
                .reply-box {{ background: #1e1b4b; border-left: 4px solid #818cf8; padding: 14px; border-radius: 4px; font-size: 15px; line-height: 1.5; color: #e0e7ff; margin-top: 6px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>@AmazonHelp Support AI Agent</h1>
                <form method="post" action="/query">
                    <div class="field-label">Customer Inbound Tweet</div>
                    <textarea name="text">{text}</textarea>
                    <button type="submit">Process Customer Message</button>
                    <a href="/" style="color: #94a3b8; margin-left: 15px; text-decoration: none;">Reset</a>
                </form>

                <div class="card">
                    <div style="margin-bottom: 15px;">
                        <span class="badge badge-intent">{res["intent"]} ({res["confidence"]*100:.1f}%)</span>
                        {esc_badge}
                    </div>

                    <div class="field-label">Stated Escalation Reason</div>
                    <div class="field-value"><strong>{res["escalation_reason_category"]}</strong>: {res["escalation_reason_detail"]}</div>

                    <div class="field-label">Drafted Grounded Response</div>
                    <div class="reply-box">{res["drafted_reply"]}</div>

                    <div class="field-label">Historical Resolution Grounding</div>
                    <div class="field-value" style="color: #94a3b8; font-size: 13px;">
                        Matched Exemplar (Similarity: {res["retrieved_exemplars"][0]["similarity"]:.3f}):<br>
                        <em>"{res["retrieved_exemplars"][0]["historical_reply"]}"</em>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        return html

    print("Starting Web Server at http://localhost:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000)


def main():
    parser = argparse.ArgumentParser(description="Amazon Support Agent Demo")
    parser.add_argument("--web", action="store_true", help="Launch interactive Web UI server")
    args = parser.parse_args()

    if args.web:
        run_web()
    else:
        run_cli()


if __name__ == "__main__":
    main()
