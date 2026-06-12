"""
LangGraph tabanli prompt optimizasyon state machine.

Mevcut iteratif döngüyü (cli.py cmd_optimize) graph-based bir
state machine olarak yeniden yapilandirir.

State Graph:
                    ┌─────────────────────────────────────────────┐
                    │                                             │
                    ▼                                             │
  START ──▶ run_train ──▶ run_validation ──▶ [should_continue]   │
                                                │           │     │
                                               end      continue  │
                                                │           │     │
                                                ▼           ▼     │
                                               END   analyze_failures
                                                │           │
                                           ┌────┘           ▼
                                           │       optimize_prompt
                                           │         │          │
                                           │      changed   unchanged
                                           │         │          │
                                           │         └──────────┤
                                           │                    │
                                          END ◀─────────────────┘
                                  (unchanged durumunda)

Kullanim:
    from optimizer_graph import run_optimization
    final_state = run_optimization(project, client)
"""

import operator
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END

from llm_utils import (
    run_all_models,
    compute_pass_rates,
    overall_rate,
    build_failure_report,
    optimize_prompt as _optimize_prompt_fn,
    print_scores,
)


# ══════════════════════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════════════════════

class OptimizerState(TypedDict):
    # Yapilandirma (baslangiçta set edilir, degismez)
    max_iterations: int
    pass_threshold: float

    # Degisen durum
    current_prompt: str
    iteration: int
    train_results: dict
    train_rates: dict
    train_overall: float
    val_results: dict
    val_rates: dict
    val_overall: float
    history: Annotated[list, operator.add]   # Her iterasyonda eklenir
    failure_report: str
    prompt_changed: bool

    # En iyi prompt takibi (regresyonu kaydetmemek icin)
    best_prompt: str
    best_combined: float
    best_train: float
    best_val: float
    best_iteration: int
    best_results: dict
    best_rates: dict


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def create_optimizer_graph(project, client):
    """
    Proje ve client ile kapatilmis (closure) node fonksiyonlari olusturur,
    LangGraph state machine'ini derleyip dondurur.
    """

    # ── Node: Egitim setini calistir ─────────────────────────────────────────

    def run_train(state: OptimizerState) -> dict:
        iteration = state["iteration"]
        prompt    = state["current_prompt"]
        max_iter  = state["max_iterations"]

        print(f"\n{'─'*65}")
        print(f"  ITERASYON {iteration}/{max_iter}")
        print(f"{'─'*65}")
        prev = prompt[:120].replace('\n', ' ')
        print(f"\n  Prompt: {prev}{'...' if len(prompt) > 120 else ''}")

        print(f"\n  [1/4] Modeller calistiriliyor ({len(project.train_cases)} egitim case)...")
        results = run_all_models(client, project, prompt, project.train_cases)
        rates   = compute_pass_rates(results)
        ovr     = overall_rate(rates)
        print_scores(rates, f"Egitim sonuclari (overall: {ovr*100:.0f}%):")

        return {
            "train_results": results,
            "train_rates":   rates,
            "train_overall": ovr,
        }

    # ── Node: Validation setini calistir + history kaydi ─────────────────────

    def run_validation(state: OptimizerState) -> dict:
        prompt    = state["current_prompt"]
        train_ovr = state["train_overall"]
        val_ovr   = 0.0

        if project.val_cases:
            print(f"\n  [2/4] Validation seti ({len(project.val_cases)} case)...")
            results  = run_all_models(client, project, prompt, project.val_cases)
            rates    = compute_pass_rates(results)
            val_ovr  = overall_rate(rates)
            print_scores(rates, f"Validation sonuclari (overall: {val_ovr*100:.0f}%):")

            gap = round(train_ovr - val_ovr, 3)
            print(f"\n  Generalisation gap: {gap*100:.0f}pp "
                  f"({'OK' if gap < 0.15 else 'YUKSEK — overfitting riski!'})")

        # Tum case'ler uzerinde birlesik (combined) pass rate — case sayisina gore agirlikli
        n_train  = len(project.train_cases)
        n_val    = len(project.val_cases)
        combined = round((train_ovr * n_train + val_ovr * n_val) / (n_train + n_val), 3)

        # history reducer (operator.add) ile otomatik append edilir
        history_entry = [{
            "iteration":     state["iteration"],
            "prompt":        state["current_prompt"],
            "train_overall": train_ovr,
            "val_overall":   val_ovr,
            "combined":      combined,
        }]

        updates = {
            "val_overall": val_ovr,
            "history":     history_entry,
        }

        # En iyi prompt'u sakla — boylece son iterasyon kotuyse bile en iyiyi kaydederiz
        if combined > state.get("best_combined", -1.0):
            updates.update({
                "best_prompt":    state["current_prompt"],
                "best_combined":  combined,
                "best_train":     train_ovr,
                "best_val":       val_ovr,
                "best_iteration": state["iteration"],
                "best_results":   state["train_results"],
                "best_rates":     state["train_rates"],
            })
            print(f"\n  ★ Yeni EN IYI prompt (combined {combined*100:.0f}% — "
                  f"iter {state['iteration']})")

        return updates

    # ── Conditional: Devam mi, dur mu? ───────────────────────────────────────

    def should_continue(state: OptimizerState) -> str:
        if state["train_overall"] >= state["pass_threshold"]:
            print(f"\n  ✓ HEDEF ULASILDI! Pass rate >= {state['pass_threshold']*100:.0f}%")
            return "end"
        if state["iteration"] >= state["max_iterations"]:
            print(f"\n  Maksimum iterasyona ulasildi ({state['max_iterations']}).")
            return "end"
        return "continue"

    # ── Node: Basarisizlik raporu olustur ────────────────────────────────────

    def analyze_failures(state: OptimizerState) -> dict:
        print(f"\n  [3/4] Basarisizlik raporu olusturuluyor...")
        # Greedy: her zaman EN IYI prompt'un sonuclarindan rapor uret.
        # Boylece regresyon yapan bir iterasyon optimizasyon tabanini bozmaz.
        results = state.get("best_results") or state["train_results"]
        rates   = state.get("best_rates")   or state["train_rates"]
        report = build_failure_report(results, rates)
        return {"failure_report": report}

    # ── Node: Optimizer LLM'i calistir ───────────────────────────────────────

    def optimize_prompt_node(state: OptimizerState) -> dict:
        print(f"  [4/4] Optimizer calistiriliyor ({project.optimizer_model})...")
        # Greedy hill-climbing: bir onceki (belki regrese olmus) prompt yerine
        # su ana kadarki EN IYI prompt'tan iyilestir. Temperature ile cesitlilik
        # ekleyerek ayni tabandan farkli denemeler uretiriz.
        base_prompt = state.get("best_prompt") or state["current_prompt"]
        new_prompt = _optimize_prompt_fn(
            client, project.optimizer_model,
            base_prompt, state["failure_report"],
            temperature=0.5,
        )
        preview = new_prompt[:100].replace('\n', ' ')
        print(f"  Yeni prompt: {preview}...")

        changed = new_prompt != state["current_prompt"]
        if not changed:
            print("  Optimizer degisiklik yapmadi.")

        return {
            "current_prompt": new_prompt,
            "iteration":      state["iteration"] + 1,
            "prompt_changed":  changed,
        }

    # ── Conditional: Prompt degisti mi? ──────────────────────────────────────

    def check_prompt_changed(state: OptimizerState) -> str:
        return "loop" if state.get("prompt_changed", True) else "stop"

    # ══════════════════════════════════════════════════════════════════════════
    # GRAPH ASSEMBLY
    # ══════════════════════════════════════════════════════════════════════════

    builder = StateGraph(OptimizerState)

    # Node'lari ekle
    builder.add_node("run_train",         run_train)
    builder.add_node("run_validation",    run_validation)
    builder.add_node("analyze_failures",  analyze_failures)
    builder.add_node("optimize_prompt",   optimize_prompt_node)

    # Edge'leri bagla
    builder.add_edge(START, "run_train")
    builder.add_edge("run_train", "run_validation")

    builder.add_conditional_edges("run_validation", should_continue, {
        "continue": "analyze_failures",
        "end":      END,
    })

    builder.add_edge("analyze_failures", "optimize_prompt")

    builder.add_conditional_edges("optimize_prompt", check_prompt_changed, {
        "loop": "run_train",
        "stop": END,
    })

    return builder.compile()


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def run_optimization(project, client) -> dict:
    """
    Tam optimizasyon döngüsünü LangGraph uzerinden calistirir.

    Returns:
        Final state dict: current_prompt, history, train_overall, vb.
    """
    graph = create_optimizer_graph(project, client)

    initial_state: OptimizerState = {
        "max_iterations":  project.max_iterations,
        "pass_threshold":  project.pass_threshold,
        "current_prompt":  project.initial_prompt,
        "iteration":       1,
        "train_results":   {},
        "train_rates":     {},
        "train_overall":   0.0,
        "val_results":     {},
        "val_rates":       {},
        "val_overall":     0.0,
        "history":         [],
        "failure_report":  "",
        "prompt_changed":  True,
        "best_prompt":     project.initial_prompt,
        "best_combined":   -1.0,
        "best_train":      0.0,
        "best_val":        0.0,
        "best_iteration":  0,
        "best_results":    {},
        "best_rates":      {},
    }

    final_state = graph.invoke(initial_state)
    return final_state
