"""
Runs Experiments 1-5 (the paper's canonical single-run configurations,
matching Section IV's numbering) end to end, in the order the paper
presents them. Imports each experiment module's own main() rather than
duplicating any logic.

This does NOT include the multi-seed robustness scripts
(experiment{1,2,3,4}_multi_seed.py) or the diagnostic re-audit scripts
(diagnose_masked_retrieval.py, reaudit_experiment5_*.py, sanity_check_
verifier.py) — those were follow-ups run during development to check
whether single-seed results were noise, and are expensive (the Exp1/2/4
multi-seed scripts alone run 15-20+ minutes each). Run them individually
if you want to reproduce that robustness checking; see DEVIATIONS_LOG.md
entries #9-#13 for what each one found. All the numbers actually cited in
final_aligned.tex come from a MIX of the single-run scripts below and the
multi-seed scripts — re-read the relevant DEVIATIONS_LOG.md entry before
assuming a single run of this file reproduces every number in the paper.

Expect this to take on the order of 20-30 minutes total on a local 7B
Ollama model — each experiment prints its own progress as it goes.

Run:
    python run_all.py
"""
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.llm_client import check_connection, MODEL_NAME

EXPERIMENTS = [
    ("Experiment 1 — SOTA Comparative Analysis",
     "experiments.experiment1_sota_comparison"),
    ("Experiment 2 — Grounding Modality Ablation",
     "experiments.experiment2_grounding_ablation"),
    ("Experiment 3 — Iterative Refinement Ablation",
     "experiments.experiment3_iterative_refinement"),
    ("Experiment 4 — Parameter Sensitivity and Overhead Analysis",
     "experiments.experiment4_parameter_sensitivity"),
    ("Experiment 5 — Diagnostic Assessment of Memory Quality",
     "experiments.experiment5_memory_quality"),
]

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def banner(title: str) -> None:
    line = "#" * 78
    print(f"\n\n{line}\n# {title}\n{line}\n")


def main() -> None:
    print(f"Checking Ollama connection (model={MODEL_NAME!r})...")
    check_connection()

    results = []
    overall_start = time.time()

    for title, module_path in EXPERIMENTS:
        banner(title)
        t0 = time.time()
        try:
            module = __import__(module_path, fromlist=["main"])
            module.main()
            status = "OK"
        except Exception as e:
            status = f"FAILED: {e}"
            print(f"\n!!! {title} raised an exception — continuing with the "
                  f"remaining experiments so one failure doesn't block the rest.")
            traceback.print_exc()
        elapsed = time.time() - t0
        results.append((title, status, elapsed))

    total_elapsed = time.time() - overall_start

    banner("SUMMARY")
    for title, status, elapsed in results:
        mins = elapsed / 60
        print(f"  [{status:<40}] {title}  ({mins:.1f} min)")
    print(f"\nTotal wall time: {total_elapsed/60:.1f} min")

    print(f"\nOutput files written to {OUTPUT_DIR}:")
    if OUTPUT_DIR.exists():
        for f in sorted(OUTPUT_DIR.glob("*.json")):
            print(f"  {f.name}")
    else:
        print("  (outputs directory not found — did any experiment run to completion?)")


if __name__ == "__main__":
    main()
