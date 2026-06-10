"""
OpenRouter bağlantı testi
Kullanım: python test_openrouter.py

API anahtarı .env dosyasındaki OPENROUTER_API_KEY değişkeninden okunur.
"""
import os
import sys

try:
    from openai import OpenAI
except ImportError:
    print("openai paketi bulunamadı. Yükleniyor...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])
    from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    sys.exit("HATA: OPENROUTER_API_KEY bulunamadı. .env dosyasını oluşturun (bkz. README.txt).")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

models = [
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "openai/gpt-oss-20b:free",
    "openai/gpt-oss-120b:free",
]

prompt = "What is 2+2? Reply with just the number."

print("\n=== OpenRouter Bağlantı Testi ===\n")
all_ok = True
for model in models:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0
        )
        answer = (resp.choices[0].message.content or "").strip()
        status = "✓" if "4" in answer else "?"
        print(f"  {status}  {model}\n     Cevap: {answer}\n")
    except Exception as e:
        all_ok = False
        print(f"  ✗  {model}\n     HATA: {e}\n")

if all_ok:
    print("Tüm modeller çalışıyor! Devam edebilirsiniz.")
else:
    print("Bazı modellerde hata var. API key'i kontrol edin.")
