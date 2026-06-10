"""
Bir proje dizinini okuyup optimizer'in anlayacagi formata cevirir.

index.md icindeki YAML frontmatter'dan proje yapisini kesfeder:
  - prompt.md        → baslangic promptu
  - credentials.md   → model listesi, API ayarlari
  - tests/           → her alt klasor bir test case
      <id>/input.txt (veya .png, .pdf)
      <id>/expected_output.json
"""

import os
import json
import base64
import re
from pathlib import Path


# ── YAML-benzeri frontmatter parser (PyYAML gerektirmez) ─────────────────────

def parse_frontmatter(text: str) -> dict:
    """--- ... --- blogu icindeki YAML'i parse eder (sadece basit tipler)."""
    match = re.search(r'^---\s*\n(.*?)\n---', text, re.DOTALL)
    if not match:
        return {}
    raw = match.group(1)
    result = {}
    current_list_key = None

    for line in raw.splitlines():
        # Liste ogesi
        if line.strip().startswith("- ") and current_list_key:
            result[current_list_key].append(line.strip()[2:].strip())
            continue
        # Anahtar: deger
        if ":" in line and not line.startswith(" "):
            current_list_key = None
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            if val == "":               # Sonraki satirlar liste olabilir
                result[key] = []
                current_list_key = key
            elif val.startswith("["):   # Inline liste [a, b, c]
                items = [x.strip().strip("'\"") for x in val.strip("[]").split(",") if x.strip()]
                result[key] = items
            else:
                # Sayi donusum dene
                try:
                    result[key] = int(val)
                except ValueError:
                    try:
                        result[key] = float(val)
                    except ValueError:
                        result[key] = val.strip("'\"")
        # Indent'li anahtar: deger (nested — split altindakiler)
        elif ":" in line and line.startswith("  "):
            key, _, val = line.strip().partition(":")
            key = key.strip()
            val = val.strip()
            if val.startswith("["):
                items = [x.strip().strip("'\"") for x in val.strip("[]").split(",") if x.strip()]
                if "split" not in result or not isinstance(result["split"], dict):
                    result["split"] = {}
                result["split"][key] = items
    return result


# ── Input yukleme ─────────────────────────────────────────────────────────────

def load_input(input_path: Path) -> dict:
    """
    Bir test case input dosyasini yukler.
    Returns: {"type": "text"|"image"|"document", "content": str|bytes, "path": str}
    """
    suffix = input_path.suffix.lower()

    if suffix in (".txt", ".md"):
        return {"type": "text", "content": input_path.read_text(encoding="utf-8").strip()}

    elif suffix in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
        data = input_path.read_bytes()
        b64  = base64.b64encode(data).decode("utf-8")
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "gif": "image/gif", "webp": "image/webp"}.get(suffix.lstrip("."), "image/png")
        return {"type": "image", "content": b64, "mime": mime, "path": str(input_path)}

    elif suffix == ".pdf":
        try:
            import pdfplumber
            with pdfplumber.open(input_path) as pdf:
                text = "\n".join(p.extract_text() or "" for p in pdf.pages)
            return {"type": "document", "content": text.strip(), "path": str(input_path)}
        except ImportError:
            raise RuntimeError("PDF destegi icin: pip3 install pdfplumber")

    else:
        # Bilinmeyen: duz metin olarak dene
        return {"type": "text", "content": input_path.read_text(encoding="utf-8").strip()}


def build_user_message(input_data: dict) -> list:
    """
    LangChain/OpenAI API icin user mesaji olusturur.
    Text icin: string. Image icin: multimodal content list.
    """
    if input_data["type"] == "text":
        return input_data["content"]

    elif input_data["type"] == "image":
        return [
            {"type": "text",      "text": "Please analyze the following image:"},
            {"type": "image_url", "image_url": {
                "url": f"data:{input_data['mime']};base64,{input_data['content']}"
            }}
        ]

    elif input_data["type"] == "document":
        return f"[Document content]\n{input_data['content']}"

    return input_data.get("content", "")


# ── Ana proje yukleyici ────────────────────────────────────────────────────────

class Project:
    """Bir proje dizinini temsil eder."""

    def __init__(self, project_dir: str):
        self.dir = Path(project_dir).resolve()
        if not self.dir.exists():
            raise FileNotFoundError(f"Proje dizini bulunamadi: {self.dir}")

        index_path = self.dir / "index.md"
        if not index_path.exists():
            raise FileNotFoundError(f"index.md bulunamadi: {index_path}")

        self.meta = parse_frontmatter(index_path.read_text(encoding="utf-8"))
        self._load_prompt()
        self._load_credentials()
        self._load_test_cases()

    # ── Prompt ────────────────────────────────────────────────────────────────

    def _load_prompt(self):
        prompt_file = self.dir / self.meta.get("prompt", "prompt.md")
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt dosyasi bulunamadi: {prompt_file}")
        self.initial_prompt = prompt_file.read_text(encoding="utf-8").strip()
        self.prompt_file    = prompt_file

    def save_prompt(self, new_prompt: str):
        """Optimize edilmis promptu output dosyasina kaydet."""
        out_rel  = self.meta.get("output", "results/optimized_prompt.md")
        out_path = self.dir / out_rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(new_prompt, encoding="utf-8")
        print(f"\n  Optimize edilmis prompt kaydedildi: {out_path}")

    # ── Credentials ───────────────────────────────────────────────────────────

    def _load_credentials(self):
        cred_file = self.dir / self.meta.get("credentials", "credentials.md")
        if not cred_file.exists():
            raise FileNotFoundError(f"credentials.md bulunamadi: {cred_file}")
        cred = parse_frontmatter(cred_file.read_text(encoding="utf-8"))

        self.provider        = cred.get("provider", "openrouter")
        self.api_key_env     = cred.get("api_key_env", "OPENROUTER_API_KEY")
        self.base_url        = cred.get("base_url", "https://openrouter.ai/api/v1")
        self.target_models   = cred.get("target_models", [])
        self.optimizer_model = cred.get("optimizer_model", "meta-llama/llama-3.3-70b-instruct")
        self.max_iterations  = int(cred.get("max_iterations", 8))
        self.pass_threshold  = float(cred.get("pass_threshold", 1.0))
        self.match_mode      = cred.get("match_mode", "keys_only")
        self.similarity_threshold = float(cred.get("similarity_threshold", 1.0))
        self.call_delay      = float(cred.get("call_delay", 1.0))

        if not self.target_models:
            raise ValueError("credentials.md icinde en az bir target_model olmali.")

    # ── Test Cases ────────────────────────────────────────────────────────────

    def _load_test_cases(self):
        tests_dir = self.dir / self.meta.get("tests", "tests/")
        split     = self.meta.get("split", {})
        train_ids = split.get("train", [])
        val_ids   = split.get("validation", [])

        all_cases = {}
        for tc_dir in sorted(tests_dir.iterdir()):
            if not tc_dir.is_dir():
                continue
            tc_id = tc_dir.name

            # Input dosyasini bul (herhangi bir uzantida olabilir)
            input_file = None
            for fname in sorted(tc_dir.iterdir()):
                if fname.stem == "input":
                    input_file = fname
                    break
            if input_file is None:
                continue

            # Expected output
            expected_file = tc_dir / "expected_output.json"
            if not expected_file.exists():
                continue

            all_cases[tc_id] = {
                "id":       tc_id,
                "input":    load_input(input_file),
                "expected": json.loads(expected_file.read_text(encoding="utf-8")),
            }

        # Train / validation ayir
        if train_ids:
            self.train_cases = [all_cases[i] for i in train_ids if i in all_cases]
        else:
            # Split yoksa alfabetik sirada ilk %80 train, kalan val
            all_list = list(all_cases.values())
            split_at = max(1, int(len(all_list) * 0.8))
            self.train_cases = all_list[:split_at]
            val_ids = [tc["id"] for tc in all_list[split_at:]]

        self.val_cases = [all_cases[i] for i in val_ids if i in all_cases]

        if not self.train_cases:
            raise ValueError("Hicbir egitim test case'i yuklenemedi.")

    # ── Ozet ──────────────────────────────────────────────────────────────────

    def summary(self):
        print(f"\n  Proje dizini : {self.dir}")
        print(f"  Baslangic prompt ({len(self.initial_prompt)} karakter):")
        print(f"    {self.initial_prompt[:100].replace(chr(10),' ')}...")
        print(f"  Target modeller: {self.target_models}")
        print(f"  Optimizer model: {self.optimizer_model}")
        print(f"  Egitim seti    : {len(self.train_cases)} case")
        print(f"  Validation     : {len(self.val_cases)} case")
        print(f"  Max iterasyon  : {self.max_iterations}")
        print(f"  Pass threshold : {self.pass_threshold*100:.0f}%")
