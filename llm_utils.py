"""
Paylaşılan LLM yardımcı fonksiyonları.

cli.py ve optimizer_graph.py tarafından kullanılır.
Circular import sorununu önlemek için ayrı modüle çıkarıldı.
"""

import json
import os
import re
import sys
import time

from openai import OpenAI

from project_loader import Project, build_user_message


# ══════════════════════════════════════════════════════════════════════════════
# LLM ÇAĞIRICI
# ══════════════════════════════════════════════════════════════════════════════

def get_client(project: Project) -> OpenAI:
    api_key = os.getenv(project.api_key_env)
    if not api_key:
        sys.exit(f"HATA: {project.api_key_env} ortam degiskeni bulunamadi. .env dosyasini kontrol edin.")
    return OpenAI(api_key=api_key, base_url=project.base_url)


def strip_thinking(text: str) -> str:
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def call_llm(client: OpenAI, model: str, system: str, user_content, max_tokens=2000,
             retries: int = 3) -> str | None:
    for attempt in range(retries + 1):
        try:
            messages = [{"role": "system", "content": system}]
            messages.append({"role": "user", "content": user_content})
            resp = client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens, temperature=0
            )
            content = resp.choices[0].message.content
            result = strip_thinking(content) if content else None
            if result is not None:
                return result
            # None döndü, retry yap
            if attempt < retries:
                time.sleep(4 * (attempt + 1))
        except Exception as e:
            # Free modeller gecici 429 (rate-limit) dondurebilir; daha uzun bekle
            is_rate_limit = "429" in str(e) or "rate-limit" in str(e).lower()
            if attempt < retries:
                time.sleep((8 if is_rate_limit else 4) * (attempt + 1))
            else:
                print(f"      [API HATA] {model}: {e}")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════

def parse_json_response(text: str) -> dict | None:
    if text is None:
        return None
    t = text.strip()
    if "```" in t:
        for part in t.split("```"):
            p = part.strip().lstrip("json").strip()
            try:
                return json.loads(p)
            except Exception:
                continue
    try:
        return json.loads(t)
    except Exception:
        match = re.search(r'\{.*\}', t, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return None


def _normalize_value(val) -> str:
    """Karsilastirma icin degeri normalize et."""
    if val is None:
        return ""
    s = str(val).strip().lower()
    # Boolean normalization
    if s in ("true", "yes", "evet"):
        return "true"
    if s in ("false", "no", "hayir"):
        return "false"
    return s


def _parse_number(s: str):
    """Farkli locale formatlarda sayi parse etmeye calis."""
    s = re.sub(r'[^\d.,\-]', '', s.strip())
    if not s:
        return None
    last_dot   = s.rfind('.')
    last_comma = s.rfind(',')
    if last_dot == -1 and last_comma == -1:
        try:
            return float(s)
        except ValueError:
            return None
    if last_dot > last_comma:
        # Son ayrac nokta → ondalik nokta (US/standart: 450,125.00)
        try:
            return float(s.replace(',', ''))
        except ValueError:
            return None
    else:
        # Son ayrac virgul → ondalik virgul (Turkce: 450.125,00)
        try:
            return float(s.replace('.', '').replace(',', '.'))
        except ValueError:
            return None


def _value_similarity(expected_val, actual_val) -> float:
    """Iki deger arasindaki benzerlik (0.0-1.0)."""
    e = _normalize_value(expected_val)
    a = _normalize_value(actual_val)
    if e == a:
        return 1.0
    # Numerik karsilastirma: birden fazla locale formatini dene
    e_num = _parse_number(e)
    a_num = _parse_number(a)
    if e_num is not None and a_num is not None:
        if abs(e_num - a_num) < 0.005:   # 2 ondalik basamak toleransi
            return 1.0
        if e_num != 0:
            return max(0.0, 1.0 - abs(e_num - a_num) / abs(e_num))
        return 0.0
    # String benzerlik (basit: ortak karakter orani)
    if not e or not a:
        return 0.0
    common = sum(1 for c in a if c in e)
    return common / max(len(e), len(a))


def evaluate(raw: str, expected: dict, match_mode: str = "keys_only",
             similarity_threshold: float = 1.0) -> dict:
    """
    LLM cevabini expected ile karsilastirir.

    match_mode:
      - "keys_only": sadece JSON key varligini kontrol et (eski davranis)
      - "value_match": key varligi + deger benzerligini kontrol et
    similarity_threshold: value_match modunda minimum kabul edilen benzerlik (0.0-1.0)
    """
    required_keys = list(expected.keys())
    parsed = parse_json_response(raw)
    if parsed is None:
        return {"valid_json": False, "missing_keys": required_keys, "score": 0.0,
                "parsed": None, "value_scores": {}}

    missing = [k for k in required_keys if k not in parsed]

    if match_mode == "keys_only":
        score = (len(required_keys) - len(missing)) / len(required_keys)
        return {"valid_json": True, "missing_keys": missing, "score": round(score, 2),
                "parsed": parsed, "value_scores": {}}

    # value_match modu: her key icin deger benzerligini hesapla
    value_scores = {}
    total_sim = 0.0
    for key in required_keys:
        if key in missing:
            value_scores[key] = 0.0
        else:
            sim = _value_similarity(expected[key], parsed[key])
            value_scores[key] = round(sim, 3)
            total_sim += (1.0 if sim >= similarity_threshold else 0.0)

    score = total_sim / len(required_keys)
    return {"valid_json": True, "missing_keys": missing, "score": round(score, 2),
            "parsed": parsed, "value_scores": value_scores}


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_all_models(client: OpenAI, project: Project, prompt: str, cases: list) -> dict:
    """Tum modelleri verilen test case'lerle calistir."""
    match_mode = getattr(project, "match_mode", "keys_only")
    sim_threshold = getattr(project, "similarity_threshold", 1.0)
    call_delay = float(getattr(project, "call_delay", 1.0))
    results = {}
    for model in project.target_models:
        results[model] = {}
        for tc in cases:
            user_msg = build_user_message(tc["input"])
            raw      = call_llm(client, model, prompt, user_msg)
            results[model][tc["id"]] = {
                "raw":      raw,
                "eval":     evaluate(raw, tc["expected"], match_mode, sim_threshold),
                "expected": tc["expected"],
            }
            # Free modellerin dakikalik rate-limit'ine takilmamak icin kucuk gecikme
            if call_delay > 0:
                time.sleep(call_delay)
    return results


def compute_pass_rates(results: dict) -> dict:
    return {
        model: round(
            sum(v["eval"]["score"] for v in tc_res.values()) / len(tc_res), 3
        )
        for model, tc_res in results.items()
    }


def overall_rate(rates: dict) -> float:
    return round(sum(rates.values()) / len(rates), 3) if rates else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# FAILURE REPORT + OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════

def build_failure_report(results: dict, rates: dict) -> str:
    lines = ["=== FAILURE REPORT ===\n"]
    for model, tc_res in results.items():
        lines.append(f"Model: {model}  (pass rate: {rates[model]*100:.0f}%)")
        for tc_id, data in tc_res.items():
            ev = data["eval"]
            if ev["score"] < 1.0:
                lines.append(f"  FAIL  {tc_id}")
                if not ev["valid_json"]:
                    lines.append(f"    → NOT valid JSON. Raw (first 200): {str(data['raw'])[:200]}")
                else:
                    lines.append(f"    → Missing keys : {ev['missing_keys']}")
                    lines.append(f"    → Model output : {json.dumps(ev['parsed'], ensure_ascii=False)}")
                    lines.append(f"    → Expected     : {json.dumps(data['expected'], ensure_ascii=False)}")
                    if ev.get("value_scores"):
                        failed_vals = {k: v for k, v in ev["value_scores"].items() if v < 1.0}
                        if failed_vals:
                            lines.append(f"    → Value mismatches: {json.dumps(failed_vals, ensure_ascii=False)}")
        lines.append("")
    return "\n".join(lines)


META_PROMPT = """You are an expert prompt engineer. Your task is to improve a system prompt so that it works correctly on ALL provided LLM models.

You will receive:
1. The CURRENT system prompt that is failing on some models
2. A FAILURE REPORT showing exactly which models fail, which test cases fail, the model's actual output, and the expected output.

Your job: Rewrite the system prompt so that ALL models produce the correct answer VALUE in a valid JSON object with EXACTLY the keys shown in the expected outputs.

HOW TO DIAGNOSE FAILURES (reason step by step before rewriting):
- Compare each model output to the expected output and look for SYSTEMATIC patterns, not one-off mistakes.
- If a numeric answer is wrong by a consistent factor (for example roughly 1000x too small or too large), the model is almost certainly MISREADING THE NUMBER FORMAT. Tables can use different locale conventions: in some locales '.' is the thousands separator and ',' is the decimal separator, while in others it is the reverse. Decide which convention makes the expected answers correct, then add an explicit, unambiguous rule telling the model exactly how to interpret '.' and ',' in the input numbers, with a short illustrative example using MADE-UP numbers (never the test values).
- If the answer value is correct but formatted differently (extra thousands separators, currency symbols, trailing text), add an explicit output-format rule: output a plain number using '.' as the decimal point and NO thousands separators.
- If models return wrong/extra keys or wrap output in prose or markdown, state the exact required keys and forbid any text outside the JSON object.

STRICT RULES:
- Return ONLY the improved system prompt text. No explanation, no commentary, no markdown fences.
- Do NOT hardcode any expected output values or test-specific numbers into the prompt. Keep it generic so it works on NEW inputs.
- BUILD ON what already works — keep the rules that are passing and only add or sharpen what is needed to fix the failures.
- Keep the prompt concise (under 400 words)."""


def optimize_prompt(client: OpenAI, optimizer_model: str,
                    current_prompt: str, failure_report: str) -> str:
    user_msg = (
        f"CURRENT SYSTEM PROMPT:\n---\n{current_prompt}\n---\n\n"
        f"{failure_report}\n\n"
        "Please provide an improved system prompt that fixes these failures on all models."
    )
    result = call_llm(client, optimizer_model, META_PROMPT, user_msg, max_tokens=2000)
    return result.strip() if result else current_prompt


# ══════════════════════════════════════════════════════════════════════════════
# PRINT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def print_scores(rates: dict, label: str = ""):
    if label:
        print(f"\n  {label}")
    for model, rate in rates.items():
        bar  = "█" * int(rate * 20) + "░" * (20 - int(rate * 20))
        name = model.split("/")[-1]
        print(f"    {bar} {rate*100:5.0f}%  {name}")


def print_results_table(results: dict):
    for model, tc_res in results.items():
        name = model.split("/")[-1]
        print(f"\n    {name}")
        for tc_id, data in tc_res.items():
            ev     = data["eval"]
            status = "✓" if ev["score"] == 1.0 else "~" if ev["score"] > 0 else "✗"
            miss   = ev["missing_keys"] or "none"
            print(f"      {status} {tc_id}  score={ev['score']}  missing={miss}")
