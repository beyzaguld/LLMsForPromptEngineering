"""
Projenin özünü gösteren ilk gerçek test.
Aynı prompt, 3 farklı model — çıktılar arasındaki farkı görün.

Kullanim: python3 benchmark_test.py
"""

import json
from openai import OpenAI

API_KEY = "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"  # <-- key'inizi buraya yazin

client = OpenAI(api_key=API_KEY, base_url="https://openrouter.ai/api/v1")

# ── Prompt (henüz optimize edilmemiş, kasıtlı olarak eksik) ──────────────────
PROMPT = """You are a software bug analyzer.
Given a bug report, extract the key information.
Return the result as JSON."""

# ── Benchmark: 3 test case ────────────────────────────────────────────────────
TEST_CASES = [
    {
        "id": "tc_001",
        "input": (
            "The login button on the checkout page crashes the app on iOS 17.2 "
            "when the user has more than 5 items in their cart. Happens 100% of the time."
        ),
        "expected": {
            "severity": "high",
            "affected_component": "checkout/login_button",
            "affected_platform": "iOS 17.2",
            "condition": "cart_items > 5",
            "reproducibility": "100%"
        }
    },
    {
        "id": "tc_002",
        "input": (
            "Sometimes the dashboard fails to load charts when the user switches "
            "between dark and light mode rapidly. Only seen on Chrome 120+. "
            "Occurs roughly 30% of the time."
        ),
        "expected": {
            "severity": "medium",
            "affected_component": "dashboard/charts",
            "affected_platform": "Chrome 120+",
            "condition": "rapid theme switch",
            "reproducibility": "30%"
        }
    },
    {
        "id": "tc_003",
        "input": (
            "Export to PDF button does nothing on Firefox when the report "
            "contains more than 50 rows. No error message shown. Reproducible every time."
        ),
        "expected": {
            "severity": "high",
            "affected_component": "report/export_pdf",
            "affected_platform": "Firefox",
            "condition": "rows > 50",
            "reproducibility": "100%"
        }
    }
]

MODELS = [
    "qwen/qwen3-30b-a3b",
    "qwen/qwen-2.5-7b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
]

REQUIRED_KEYS = {"severity", "affected_component", "affected_platform", "condition", "reproducibility"}

def evaluate(response_text: str, expected: dict) -> dict:
    """Response'u parse edip expected ile karşılaştır."""
    # JSON bloğunu bul
    text = response_text.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError:
        return {"valid_json": False, "missing_keys": list(REQUIRED_KEYS), "score": 0}

    missing = [k for k in REQUIRED_KEYS if k not in parsed]
    score = (len(REQUIRED_KEYS) - len(missing)) / len(REQUIRED_KEYS)
    return {"valid_json": True, "missing_keys": missing, "score": round(score, 2)}

# ── Ana döngü ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  BENCHMARK: Structured JSON Output (Bug Report Extractor)")
print("="*60)
print(f"\nPrompt:\n{PROMPT}\n")
print("─"*60)

results = {}

for model in MODELS:
    print(f"\nModel: {model}")
    model_scores = []

    for tc in TEST_CASES:
        messages = [
            {"role": "system", "content": PROMPT},
            {"role": "user",   "content": f"Bug report:\n{tc['input']}"}
        ]
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=300,
                temperature=0
            )
            raw = resp.choices[0].message.content
            eval_result = evaluate(raw, tc["expected"])
            model_scores.append(eval_result["score"])

            status = "✓" if eval_result["score"] == 1.0 else "~" if eval_result["score"] > 0 else "✗"
            print(f"  {status} TC {tc['id']}  score={eval_result['score']}  "
                  f"{'valid JSON' if eval_result['valid_json'] else 'INVALID JSON'}  "
                  f"missing={eval_result['missing_keys'] or 'none'}")
            if eval_result["score"] < 1.0:
                print(f"    Raw output: {raw[:200].strip()}")
        except Exception as e:
            print(f"  ✗ TC {tc['id']}  HATA: {e}")
            model_scores.append(0)

    avg = round(sum(model_scores) / len(model_scores), 2)
    results[model] = avg
    print(f"  → Pass rate: {avg*100:.0f}%")

print("\n" + "="*60)
print("  ÖZET")
print("="*60)
for model, score in results.items():
    bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
    print(f"  {bar} {score*100:.0f}%  {model.split('/')[-1]}")

print("\nModeller arasındaki fark ne kadar büyükse,")
print("prompt optimizasyonuna o kadar çok ihtiyaç var.")
print("="*60 + "\n")
