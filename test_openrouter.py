"""
OpenRouter bağlantı testi
Kullanım: python test_openrouter.py
"""
import sys

try:
    from openai import OpenAI
except ImportError:
    print("openai paketi bulunamadı. Yükleniyor...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])
    from openai import OpenAI

# ── Buraya yeni API key'inizi girin ───────────────────────────────────────────
API_KEY = "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"  # <-- buraya OpenRouter key'inizi yazin
# ─────────────────────────────────────────────────────────────────────────────

client = OpenAI(
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

models = [
    "qwen/qwen3-30b-a3b",
    "qwen/qwen-2.5-7b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
]

prompt = "What is 2+2? Reply with just the number."

print("\n=== OpenRouter Bağlantı Testi ===\n")
all_ok = True
for model in models:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0
        )
        answer = resp.choices[0].message.content.strip()
        status = "✓" if "4" in answer else "?"
        print(f"  {status}  {model}\n     Cevap: {answer}\n")
    except Exception as e:
        all_ok = False
        print(f"  ✗  {model}\n     HATA: {e}\n")

if all_ok:
    print("Tüm modeller çalışıyor! Devam edebilirsiniz.")
else:
    print("Bazı modellerde hata var. API key'i kontrol edin.")
