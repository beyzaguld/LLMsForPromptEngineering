"""
Projenin kalbi: Feedback-driven prompt optimizer loop.

Calisma mantigi:
  1. Mevcut prompt'u tum modellere gonder
  2. Sonuclari degerlendir → hangi model, hangi test case'de basarisiz?
  3. Basarisizliklari optimizer LLM'e gonder
  4. Optimizer yeni bir prompt onerisi dondurur
  5. 1'e don — ta ki tum modeller gece veya max iterasyona ulasana kadar

Kullanim:
  pip install openai python-dotenv
  python3 optimizer_loop.py
"""

import json
import os
import copy
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(api_key=API_KEY, base_url="https://openrouter.ai/api/v1")

# ══════════════════════════════════════════════════════════════════════════════
# YAPILANDIRMA
# ══════════════════════════════════════════════════════════════════════════════

# Optimize edilecek modeller (target models)
TARGET_MODELS = [
    "qwen/qwen3-30b-a3b",
    "qwen/qwen-2.5-7b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
]

# Prompt'u optimize eden guclu model (optimizer model)
# Qwen3 "thinking" modunda calistigi icin optimizer olarak kullanmiyoruz
OPTIMIZER_MODEL = "meta-llama/llama-3.3-70b-instruct"

MAX_ITERATIONS   = 8      # Maksimum deneme sayisi
PASS_THRESHOLD   = 1.0    # 1.0 = %100 (tum modeller tum test case'leri gecmeli)
REQUIRED_KEYS    = ["severity", "affected_component", "affected_platform",
                    "condition", "reproducibility"]

# ══════════════════════════════════════════════════════════════════════════════
# BASLANGIC PROMPTU (zayif, optimize edilmemis)
# ══════════════════════════════════════════════════════════════════════════════

INITIAL_PROMPT = """You are a software bug analyzer.
Given a bug report, extract the key information.
Return the result as JSON."""

# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK (egitim seti — optimizer bunlari gorecek)
# ══════════════════════════════════════════════════════════════════════════════

TRAIN_CASES = [
    {
        "id": "tc_001",
        "input": "The login button on the checkout page crashes the app on iOS 17.2 "
                 "when the user has more than 5 items in their cart. Happens 100% of the time.",
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
        "input": "Sometimes the dashboard fails to load charts when the user switches "
                 "between dark and light mode rapidly. Only seen on Chrome 120+. "
                 "Occurs roughly 30% of the time.",
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
        "input": "Export to PDF button does nothing on Firefox when the report "
                 "contains more than 50 rows. No error message shown. Reproducible every time.",
        "expected": {
            "severity": "high",
            "affected_component": "report/export_pdf",
            "affected_platform": "Firefox",
            "condition": "rows > 50",
            "reproducibility": "100%"
        }
    },
    {
        "id": "tc_004",
        "input": "Profile picture upload silently fails on Safari 16 when the image "
                 "file is larger than 5MB. No error shown to the user. Always reproducible.",
        "expected": {
            "severity": "medium",
            "affected_component": "profile/picture_upload",
            "affected_platform": "Safari 16",
            "condition": "file_size > 5MB",
            "reproducibility": "100%"
        }
    },
]

# Validation seti — optimizer HICBIR ZAMAN bunlari gormez (generalisation olcumu icin)
VALIDATION_CASES = [
    {
        "id": "val_001",
        "input": "Notification badge count resets to zero when the app is backgrounded "
                 "on Android 13. Happens about 50% of the time.",
        "expected": {
            "severity": "medium",
            "affected_component": "notifications/badge_count",
            "affected_platform": "Android 13",
            "condition": "app backgrounded",
            "reproducibility": "50%"
        }
    },
    {
        "id": "val_002",
        "input": "Search autocomplete dropdown blocks the submit button on mobile viewports "
                 "below 375px width. Reproducible every time on all browsers.",
        "expected": {
            "severity": "high",
            "affected_component": "search/autocomplete_dropdown",
            "affected_platform": "mobile viewport < 375px",
            "condition": "autocomplete visible",
            "reproducibility": "100%"
        }
    },
]

# ══════════════════════════════════════════════════════════════════════════════
# YARDIMCI FONKSIYONLAR
# ══════════════════════════════════════════════════════════════════════════════

def strip_thinking(text: str) -> str:
    """Qwen3 ve benzeri modellerin <think>...</think> bloklarini temizle."""
    import re
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned.strip()


def call_llm(model: str, system: str, user: str, max_tokens=600) -> str | None:
    """Bir modele sistem + kullanici mesaji gonder, cevabi dondur."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user}
            ],
            max_tokens=max_tokens,
            temperature=0
        )
        content = resp.choices[0].message.content
        if content:
            content = strip_thinking(content)
        return content
    except Exception as e:
        print(f"      [API HATA] {model}: {e}")
        return None


def parse_json(text: str) -> dict | None:
    """LLM ciktisindaki JSON'u parse et."""
    if text is None:
        return None
    t = text.strip()
    # Markdown kod blogu varsa temizle
    if "```" in t:
        parts = t.split("```")
        for p in parts:
            p2 = p.strip()
            if p2.startswith("json"):
                p2 = p2[4:]
            try:
                return json.loads(p2.strip())
            except Exception:
                continue
    try:
        return json.loads(t)
    except Exception:
        return None


def evaluate_response(raw: str, expected: dict) -> dict:
    """
    Bir modelin cevabini degerlendir.
    Returns: { valid_json, missing_keys, score (0-1), parsed }
    """
    parsed = parse_json(raw)
    if parsed is None:
        return {"valid_json": False, "missing_keys": REQUIRED_KEYS[:], "score": 0.0, "parsed": None}
    missing = [k for k in REQUIRED_KEYS if k not in parsed]
    score   = (len(REQUIRED_KEYS) - len(missing)) / len(REQUIRED_KEYS)
    return {"valid_json": True, "missing_keys": missing, "score": round(score, 2), "parsed": parsed}


def run_all_models(prompt: str, test_cases: list) -> dict:
    """
    Tum target modelleri verilen prompt + test case'lerle calistir.
    Returns: { model -> { tc_id -> {raw, eval_result} } }
    """
    results = {}
    for model in TARGET_MODELS:
        results[model] = {}
        for tc in test_cases:
            raw = call_llm(model, prompt, f"Bug report:\n{tc['input']}")
            results[model][tc["id"]] = {
                "raw":  raw,
                "eval": evaluate_response(raw, tc["expected"]),
                "expected": tc["expected"]
            }
    return results


def compute_pass_rates(results: dict) -> dict:
    """Her model icin ortalama score hesapla."""
    rates = {}
    for model, tc_results in results.items():
        scores = [v["eval"]["score"] for v in tc_results.values()]
        rates[model] = round(sum(scores) / len(scores), 3)
    return rates


def overall_pass_rate(rates: dict) -> float:
    return round(sum(rates.values()) / len(rates), 3)


def build_failure_report(results: dict, rates: dict) -> str:
    """
    Optimizer'in anlayacagi formatta bir basarisizlik raporu olustur.
    Bu rapor meta-prompt'a eklenir.
    """
    lines = ["=== FAILURE REPORT ===\n"]
    for model, tc_results in results.items():
        model_rate = rates[model]
        lines.append(f"Model: {model}  (pass rate: {model_rate*100:.0f}%)")
        for tc_id, data in tc_results.items():
            ev = data["eval"]
            if ev["score"] < 1.0:
                lines.append(f"  FAIL  {tc_id}")
                if not ev["valid_json"]:
                    lines.append(f"    → Output was NOT valid JSON.")
                    lines.append(f"    → Raw output (first 200 chars): {str(data['raw'])[:200]}")
                else:
                    lines.append(f"    → Missing keys: {ev['missing_keys']}")
                    lines.append(f"    → Model returned: {json.dumps(ev['parsed'], ensure_ascii=False)}")
                    lines.append(f"    → Expected had:   {json.dumps(data['expected'], ensure_ascii=False)}")
        lines.append("")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# META-PROMPT (optimizer LLM'e verilen talimat)
# ══════════════════════════════════════════════════════════════════════════════

META_PROMPT = """You are an expert prompt engineer. Your task is to improve a system prompt so that it works correctly on ALL provided LLM models.

You will be given:
1. The CURRENT system prompt that is failing on some models
2. A FAILURE REPORT showing exactly which models fail, which test cases fail, and why

Your job: Rewrite the system prompt so that ALL models return a valid JSON object with EXACTLY these keys:
  severity, affected_component, affected_platform, condition, reproducibility

STRICT RULES:
- Return ONLY the improved system prompt text. No explanation, no commentary, no code blocks.
- Do NOT hardcode any expected outputs or test case answers into the prompt.
- The improved prompt must be generic — it should work on NEW bug reports, not just the ones in the failure report.
- Make the output format requirements crystal clear and unambiguous.
- If models are producing wrong key names, explicitly list the required key names.
- If models are wrapping output in prose or markdown, add explicit instructions to prevent that.
- Keep the prompt reasonably concise (under 400 words).

Remember: you are writing a system prompt for a bug report JSON extractor."""


def optimize_prompt(current_prompt: str, failure_report: str) -> str:
    """
    Optimizer LLM'e mevcut prompt + basarisizlik raporunu gonder,
    yeni bir prompt al.
    """
    user_msg = f"""CURRENT SYSTEM PROMPT:
---
{current_prompt}
---

{failure_report}

Please provide an improved system prompt that fixes these failures on all models."""

    new_prompt = call_llm(
        OPTIMIZER_MODEL,
        META_PROMPT,
        user_msg,
        max_tokens=800
    )
    return new_prompt.strip() if new_prompt else current_prompt


# ══════════════════════════════════════════════════════════════════════════════
# ANA OPTIMIZER DONGUSU
# ══════════════════════════════════════════════════════════════════════════════

def print_scores(rates: dict, label: str = ""):
    if label:
        print(f"\n  {label}")
    for model, rate in rates.items():
        bar  = "█" * int(rate * 20) + "░" * (20 - int(rate * 20))
        name = model.split("/")[-1]
        print(f"    {bar} {rate*100:5.0f}%  {name}")


def run_optimizer():
    print("\n" + "="*65)
    print("  FEEDBACK-DRIVEN PROMPT OPTIMIZER")
    print("="*65)

    current_prompt = INITIAL_PROMPT
    history = []  # Her iterasyonun sonuclarini sakla

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n{'─'*65}")
        print(f"  ITERASYON {iteration}/{MAX_ITERATIONS}")
        print(f"{'─'*65}")
        print(f"\n  Mevcut prompt ({len(current_prompt)} karakter):")
        preview = current_prompt[:200].replace('\n', ' ')
        print(f"  {preview}{'...' if len(current_prompt) > 200 else ''}")

        # 1. Tum modelleri egitim seti uzerinde calistir
        print(f"\n  [1/4] Tum modeller calistiriliyor ({len(TRAIN_CASES)} test case)...")
        results = run_all_models(current_prompt, TRAIN_CASES)
        rates   = compute_pass_rates(results)
        overall = overall_pass_rate(rates)

        print_scores(rates, f"Egitim seti sonuclari (overall: {overall*100:.0f}%):")

        # 2. Validation seti uzerinde de test et (generalisation)
        print(f"\n  [2/4] Validation seti kontrol ediliyor ({len(VALIDATION_CASES)} case)...")
        val_results = run_all_models(current_prompt, VALIDATION_CASES)
        val_rates   = compute_pass_rates(val_results)
        val_overall = overall_pass_rate(val_rates)
        print_scores(val_rates, f"Validation sonuclari  (overall: {val_overall*100:.0f}%):")

        gap = round(overall - val_overall, 3)
        print(f"\n  Generalisation gap: {gap*100:.0f}pp "
              f"({'OK' if gap < 0.15 else 'YUKSEK — overfitting riski!'})")

        # Sonuclari kaydet
        history.append({
            "iteration":   iteration,
            "prompt":      current_prompt,
            "train_rates": rates,
            "train_overall": overall,
            "val_rates":   val_rates,
            "val_overall": val_overall,
        })

        # 3. Durma kosullari kontrol et
        if overall >= PASS_THRESHOLD:
            print(f"\n  ✓ HEDEF ULASILDI! Tum modeller %{PASS_THRESHOLD*100:.0f} pass rate'e ulasti.")
            break

        if iteration == MAX_ITERATIONS:
            print(f"\n  Maksimum iterasyon sayisina ulasildi ({MAX_ITERATIONS}).")
            break

        # 4. Basarisizlik raporu olustur ve prompt'u optimize et
        print(f"\n  [3/4] Basarisizlik raporu olusturuluyor...")
        failure_report = build_failure_report(results, rates)

        print(f"  [4/4] Optimizer LLM prompt'u iyilestiriyor ({OPTIMIZER_MODEL})...")
        new_prompt = optimize_prompt(current_prompt, failure_report)

        print(f"  Yeni prompt onizleme: {new_prompt[:120].replace(chr(10),' ')}...")

        if new_prompt == current_prompt:
            print("  Optimizer hic degisiklik yapmadi. Duruyorum.")
            break

        current_prompt = new_prompt

    # ── Ozet ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  OPTIMIZASYON TAMAMLANDI — OZET")
    print(f"{'='*65}")
    print(f"\n  Toplam iterasyon: {len(history)}")
    print(f"\n  Iterasyon basina egitim pass rate:")
    for h in history:
        bar = "█" * int(h['train_overall'] * 20) + "░" * (20 - int(h['train_overall'] * 20))
        print(f"    Iter {h['iteration']}: {bar} {h['train_overall']*100:.0f}%")

    print(f"\n  FINAL PROMPT:")
    print("  " + "─"*50)
    for line in current_prompt.split('\n'):
        print(f"  {line}")
    print("  " + "─"*50)

    # Sonuclari JSON dosyasina kaydet
    os.makedirs("results", exist_ok=True)
    out_path = "results/optimization_run.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "history": history,
            "final_prompt": current_prompt
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  Detayli sonuclar kaydedildi: {out_path}\n")


if __name__ == "__main__":
    run_optimizer()
