"""
Prompt Optimizer CLI

Kullanim:
  python3 cli.py optimize --dir projects/bug_reporter
  python3 cli.py optimize --dir projects/tool_caller --max-iter 5
  python3 cli.py test     --dir projects/bug_reporter
  python3 cli.py list

Komutlar:
  optimize   Prompt'u LangGraph state machine ile optimize et ve kaydet
  test       Mevcut prompt'u benchmark ile calistir (optimize etmeden)
  list       Mevcut proje dizinlerini listele
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from project_loader import Project
from llm_utils import (
    get_client, run_all_models, compute_pass_rates, overall_rate,
    print_scores, print_results_table,
)

load_dotenv()


# ══════════════════════════════════════════════════════════════════════════════
# KOMUTLAR
# ══════════════════════════════════════════════════════════════════════════════

def cmd_test(args):
    """Mevcut promptu calistir, optimize etme."""
    project = Project(args.dir)
    client  = get_client(project)

    # --prompt flag'i varsa o dosyadan oku, yoksa project'in initial_prompt'unu kullan
    if args.prompt:
        prompt_path = Path(args.prompt)
        if not prompt_path.is_absolute():
            prompt_path = Path(args.dir) / prompt_path
        if not prompt_path.exists():
            print(f"HATA: Prompt dosyasi bulunamadi: {prompt_path}")
            return
        prompt = prompt_path.read_text(encoding="utf-8").strip()
        prompt_label = str(args.prompt)
    else:
        prompt = project.initial_prompt
        prompt_label = "prompt.md (baslangic)"

    print("\n" + "="*65)
    print(f"  TEST: {args.dir}")
    print("="*65)
    project.summary()
    print(f"  Kullanilan prompt  : {prompt_label}")
    print(f"  Prompt ozeti       : {prompt[:80]}...")

    print(f"\n  Egitim seti ({len(project.train_cases)} case):")
    train_res   = run_all_models(client, project, prompt, project.train_cases)
    train_rates = compute_pass_rates(train_res)
    print_results_table(train_res)
    print_scores(train_rates, f"Ozet (overall: {overall_rate(train_rates)*100:.0f}%):")

    if project.val_cases:
        print(f"\n  Validation seti ({len(project.val_cases)} case):")
        val_res   = run_all_models(client, project, prompt, project.val_cases)
        val_rates = compute_pass_rates(val_res)
        print_results_table(val_res)
        print_scores(val_rates, f"Validation ozet (overall: {overall_rate(val_rates)*100:.0f}%):")


def cmd_optimize(args):
    """Prompt'u LangGraph state machine ile optimize et ve kaydet."""
    from optimizer_graph import run_optimization

    project = Project(args.dir)
    client  = get_client(project)

    # CLI argumanlari credentials'daki degerleri override edebilir
    if args.max_iter:
        project.max_iterations = args.max_iter
    if args.threshold:
        project.pass_threshold = args.threshold

    print("\n" + "="*65)
    print(f"  OPTIMIZE (LangGraph): {args.dir}")
    print("="*65)
    project.summary()

    # LangGraph state machine'i calistir
    final_state = run_optimization(project, client)

    current_prompt = final_state["current_prompt"]
    best_prompt    = final_state.get("best_prompt", current_prompt)
    history        = final_state["history"]

    # ── Ozet ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  OPTIMIZASYON OZETI")
    print(f"{'='*65}")
    print(f"\n  Iterasyon basina pass rate (train | val | combined):")
    for h in history:
        bar = "█" * int(h['train_overall'] * 20) + "░" * (20 - int(h['train_overall'] * 20))
        star = "  ★ EN IYI" if h['iteration'] == final_state.get("best_iteration") else ""
        print(f"    Iter {h['iteration']}: {bar} "
              f"train {h['train_overall']*100:3.0f}% | val {h.get('val_overall',0)*100:3.0f}% | "
              f"combined {h.get('combined',0)*100:3.0f}%{star}")

    print(f"\n  EN IYI prompt: iter {final_state.get('best_iteration')}  "
          f"(train {final_state.get('best_train',0)*100:.0f}% | "
          f"val {final_state.get('best_val',0)*100:.0f}% | "
          f"combined {final_state.get('best_combined',0)*100:.0f}%)")

    print(f"\n  FINAL (EN IYI) PROMPT:\n  {'─'*55}")
    for line in best_prompt.split('\n'):
        print(f"  {line}")
    print(f"  {'─'*55}")

    # Kaydet — son prompt degil, EN IYI prompt kaydedilir
    project.save_prompt(best_prompt)

    # JSON log
    log_path = Path(args.dir) / "results" / "optimization_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "history": history,
            "best_iteration": final_state.get("best_iteration"),
            "best_train": final_state.get("best_train"),
            "best_val": final_state.get("best_val"),
            "best_combined": final_state.get("best_combined"),
            "best_prompt": best_prompt,
            "final_prompt": current_prompt,
        }, f, ensure_ascii=False, indent=2)
    print(f"  Log kaydedildi: {log_path}\n")


def cmd_list(args):
    """Mevcut proje dizinlerini listele."""
    base = Path(args.projects_dir)
    if not base.exists():
        print(f"Proje dizini bulunamadi: {base}")
        return
    print(f"\nProjeler ({base}):")
    for d in sorted(base.iterdir()):
        if d.is_dir() and (d / "index.md").exists():
            try:
                p = Project(str(d))
                print(f"  {d.name:<25} "
                      f"{len(p.train_cases)} train + {len(p.val_cases)} val case  |  "
                      f"{len(p.target_models)} model")
            except Exception as e:
                print(f"  {d.name:<25} [HATA: {e}]")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# ARGPARSE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Feedback-driven LLM Prompt Optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ornekler:
  python3 cli.py optimize --dir projects/bug_reporter
  python3 cli.py optimize --dir projects/tool_caller --max-iter 5
  python3 cli.py test     --dir projects/bug_reporter
  python3 cli.py list
        """
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # optimize
    p_opt = sub.add_parser("optimize", help="Prompt'u optimize et")
    p_opt.add_argument("--dir",       required=True, help="Proje dizini")
    p_opt.add_argument("--max-iter",  type=int,   default=None, help="Max iterasyon sayisi")
    p_opt.add_argument("--threshold", type=float, default=None, help="Hedef pass rate (0-1)")

    # test
    p_test = sub.add_parser("test", help="Mevcut promptu test et (optimize etmeden)")
    p_test.add_argument("--dir",    required=True, help="Proje dizini")
    p_test.add_argument("--prompt", default=None,  help="Ozel prompt dosyasi (ornek: results/optimized_prompt.md)")

    # list
    p_list = sub.add_parser("list", help="Proje dizinlerini listele")
    p_list.add_argument("--projects-dir", default="projects", help="Projeler ust klasoru")

    args = parser.parse_args()

    if args.command == "optimize":
        cmd_optimize(args)
    elif args.command == "test":
        cmd_test(args)
    elif args.command == "list":
        cmd_list(args)


if __name__ == "__main__":
    main()
