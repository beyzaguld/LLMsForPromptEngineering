"""
Capture concrete before/after model outputs for the report's experiments section.

For each project it runs ONE representative test case through every target model,
once with the initial (weak) prompt and once with the optimized prompt, then
records the raw model output and the evaluator score. Results are written to
projects/<name>/results/sample_outputs.json.

Usage:  python3 tools_capture_samples.py
"""
import json
import time
from pathlib import Path

from dotenv import load_dotenv

from project_loader import Project, build_user_message
from llm_utils import get_client, call_llm, evaluate

load_dotenv()

# project -> representative test-case id to showcase
SHOWCASE = {
    "bug_reporter":        "tc_001",
    "tool_caller":         "tc_002",
    "locale_query":        "tc_001",
    "screenshot_reporter": "tc_001",
}


def find_case(project: Project, tc_id: str):
    for tc in project.train_cases + project.val_cases:
        if tc["id"] == tc_id:
            return tc
    return (project.train_cases or project.val_cases)[0]


def run_one(client, project, prompt, tc):
    match_mode = getattr(project, "match_mode", "keys_only")
    sim = getattr(project, "similarity_threshold", 1.0)
    out = {}
    for model in project.target_models:
        raw = call_llm(client, model, prompt, build_user_message(tc["input"]))
        ev = evaluate(raw, tc["expected"], match_mode, sim)
        out[model] = {"raw": raw, "score": ev["score"]}
        time.sleep(float(getattr(project, "call_delay", 2.0)))
    return out


def main():
    for name, tc_id in SHOWCASE.items():
        pdir = f"projects/{name}"
        project = Project(pdir)
        client = get_client(project)

        tc = find_case(project, tc_id)
        weak = project.initial_prompt
        opt_path = Path(pdir) / "results" / "optimized_prompt.md"
        optimized = opt_path.read_text(encoding="utf-8").strip()

        print(f"\n==== {name} / {tc['id']} ====")
        print("  [weak prompt] running ...")
        before = run_one(client, project, weak, tc)
        print("  [optimized prompt] running ...")
        after = run_one(client, project, optimized, tc)

        record = {
            "project": name,
            "test_case": tc["id"],
            "expected": tc["expected"],
            "weak_prompt": weak,
            "optimized_prompt": optimized,
            "before": before,
            "after": after,
        }
        out_path = Path(pdir) / "results" / "sample_outputs.json"
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  saved -> {out_path}")
        for m in project.target_models:
            print(f"    {m.split('/')[-1]:<40} before={before[m]['score']}  after={after[m]['score']}")


if __name__ == "__main__":
    main()
