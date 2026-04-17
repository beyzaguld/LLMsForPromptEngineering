"""
Ayni benchmark, iyilestirilmis prompt ile.
Projenin optimizer'inin yapacagi sey budur — manuel olarak yapiyoruz.

Kullanim: python3 benchmark_improved.py
"""

import json
from openai import OpenAI

API_KEY = "" 

client = OpenAI(api_key=API_KEY, base_url="https://openrouter.ai/api/v1")

# ── V1: Orijinal (kotu) prompt ────────────────────────────────────────────────
PROMPT_V1 = """You are a software bug analyzer.
Given a bug report, extract the key information.
Return the result as JSON."""

# ── V2: Optimize edilmis prompt ────────────────────────────────────────────────
PROMPT_V2 = """You are a software bug analyzer. Extract information from bug reports.

You MUST return ONLY a valid JSON object with EXACTLY these fields (no extra fields, no explanation):
{
  "severity": "high" | "medium" | "low",
  "affected_component": "<module/feature path, e.g. checkout/login_button>",
  "affected_platform": "<OS, browser, or device with version>",
  "condition": "<the specific condition that triggers the bug>",
  "reproducibility": "<percentage or always/sometimes/rarely>"
}

Rules:
- Do NOT include any text before or after the JSON.
- Do NOT wrap the JSON in markdown code blocks.
- severity is "high" if the app crashes or a core feature is broken, "medium" for partial failures, "low" for cosmetic issues.
- affected_component uses slash-separated path format.

Example input: "The save button in the editor crashes the app on Android 12 every time."
Example output:
{"severity": "high", "affected_component": "editor/save_button", "affected_platform": "Android 12", "condition": "clicking save button", "reproducibility": "100%"}"""

# ── Test cases ─────────────────────────────────────────────────────────────────
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

def evaluate(response_text, expected):
    if response_text is None:
        return {"valid_json": False, "missing_keys": list(REQUIRED_KEYS), "score": 0}
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

def run_benchmark(prompt, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    all_scores = {}
    for model in MODELS:
        print(f"\n  Model: {model.split('/')[-1]}")
        model_scores = []
        for tc in TEST_CASES:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Bug report:\n{tc['input']}"}
            ]
            try:
                resp = client.chat.completions.create(
                    model=model, messages=messages,
                    max_tokens=300, temperature=0
                )
                raw = resp.choices[0].message.content if resp.choices[0].message.content else None
                result = evaluate(raw, tc["expected"])
                model_scores.append(result["score"])
                status = "✓" if result["score"] == 1.0 else "~" if result["score"] > 0 else "✗"
                print(f"    {status} {tc['id']}  score={result['score']}  "
                      f"missing={result['missing_keys'] or 'none'}")
            except Exception as e:
                model_scores.append(0)
                print(f"    ✗ {tc['id']}  HATA: {e}")
        avg = round(sum(model_scores) / len(model_scores), 2)
        all_scores[model] = avg
        print(f"    → Pass rate: {avg*100:.0f}%")
    return all_scores

# ── Calistir ──────────────────────────────────────────────────────────────────
scores_v1 = run_benchmark(PROMPT_V1, "PROMPT V1 — Orijinal (optimize edilmemis)")
scores_v2 = run_benchmark(PROMPT_V2, "PROMPT V2 — Optimize edilmis")

print(f"\n{'='*60}")
print("  KARSILASTIRMA: V1 vs V2")
print(f"{'='*60}")
print(f"  {'Model':<30} {'V1':>6} {'V2':>6} {'Artis':>8}")
print(f"  {'-'*52}")
for model in MODELS:
    name  = model.split("/")[-1]
    v1    = scores_v1[model]
    v2    = scores_v2[model]
    delta = v2 - v1
    arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
    print(f"  {name:<30} {v1*100:>5.0f}% {v2*100:>5.0f}%  {arrow} {abs(delta)*100:+.0f}pp")
print(f"{'='*60}")
print("\nProjenizin optimizer'i bu iyilestirmeyi otomatik yapacak.")
print("Hedef: tum modellerde V2 seviyesine veya daha yukseğe ulasmak.\n")
