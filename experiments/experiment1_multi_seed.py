"""
Multi-seed robustness check for Experiment 1.

Level 1 and Level 2 have no masking (deterministic dataset) — repeated
ONCE each to check whether the current model's numbers hold under a
fresh run. ORIGINAL below must match whatever model MODEL_NAME is
currently set to (see core/llm_client.py) — it's the baseline this
script checks the repeat run AGAINST, not a fixed 7B reference. Update
it whenever you change MODEL_NAME, or this "repeatability check" will
just show every method as "DIFFERS" because it's comparing across two
different models rather than checking determinism of one.

Level 3 (has real masking randomness) — full 4-method comparison across
3 different masking seeds, to check whether a result is a stable
property of the current model/architecture combination or an artifact
of one particular masking draw.

Writes an incremental checkpoint to outputs/experiment1_multiseed_results.json
after EVERY stage (Level1 repeat, Level2 repeat, each Level3 seed) — a
full run can take hours with a large model, and losing everything to a
crash near the end has happened before (see DEVIATIONS_LOG.md #19).

Run:
    python -m experiments.experiment1_multi_seed
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.llm_client import check_connection, MODEL_NAME
from core.task_env import make_level3_split, all_combinations, level1_combinations
from experiments.experiment1_sota_comparison import run_level, OUTPUT_DIR

LEVEL3_SEEDS = [42, 7, 123]

# NOTE: these are the CURRENT model's (llama3.3:70b) baseline numbers,
# from the standalone experiment1_sota_comparison.py run — update this
# dict whenever MODEL_NAME changes, or the "repeatability check" below
# is meaningless (see module docstring).
ORIGINAL = {
    # Updated after the Reflexion baseline fairness fix (DEVIATIONS_LOG.md
    # #32-33) -- these are the CURRENT model'''s (llama3.3:70b) numbers with
    # the FIXED K-bounded Reflexion, from the standalone
    # experiment1_sota_comparison.py re-run. The old pre-fix values
    # (Reflexion 0.500/0.438) are superseded; using them here would make
    # this repeatability check misreport the fix itself as noise.
    "Level1": {"VLEG": 1.000, "HDR": 1.000, "Reflexion": 0.875, "ICL": 1.000},
    "Level2": {"VLEG": 1.000, "HDR": 1.000, "Reflexion": 0.750, "ICL": 0.9375},
}

out_path = OUTPUT_DIR / "experiment1_multiseed_results.json"


def save_partial(state: dict):
    out_path.write_text(json.dumps(state, indent=2))
    print(f"  [checkpoint] partial multi-seed results written to {out_path}")


def main():
    print(f"Checking Ollama connection (model={MODEL_NAME!r})...")
    check_connection()

    level1_combos = level1_combinations()
    level2_combos = all_combinations()

    state = {"model": MODEL_NAME, "level1_repeat": None, "level2_repeat": None, "level3_by_seed": []}

    print(f"\n\n{'#'*70}\n# LEVEL 1 — REPEAT RUN (determinism check)\n{'#'*70}")
    level1_repeat = run_level("Level1_repeat", level1_combos, level1_combos)
    state["level1_repeat"] = level1_repeat
    save_partial(state)

    print(f"\n\n{'#'*70}\n# LEVEL 2 — REPEAT RUN (determinism check)\n{'#'*70}")
    level2_repeat = run_level("Level2_repeat", level2_combos, level2_combos)
    state["level2_repeat"] = level2_repeat
    save_partial(state)

    print(f"\n\n{'='*70}\nLEVEL 1/2 REPEATABILITY CHECK (model={MODEL_NAME!r})\n{'='*70}")
    for level, repeat_result in [("Level1", level1_repeat), ("Level2", level2_repeat)]:
        print(f"\n{level}:")
        for method in ["VLEG", "HDR", "Reflexion", "ICL"]:
            orig = ORIGINAL[level][method] * 100
            new = repeat_result["results"][method]["accuracy"] * 100
            match = "SAME" if abs(orig - new) < 1e-6 else f"DIFFERS (was {orig:.1f}%)"
            print(f"  {method:<10} {new:>6.1f}%   {match}")

    level3_results = []
    for seed in LEVEL3_SEEDS:
        print(f"\n\n{'#'*70}\n# LEVEL 3 — SEED = {seed}\n{'#'*70}")
        train, test = make_level3_split(mask_count=4, seed=seed)
        r = run_level(f"Level3_seed{seed}", train, test)
        r["seed"] = seed
        level3_results.append(r)
        state["level3_by_seed"] = level3_results
        save_partial(state)

    print(f"\n\n{'='*80}\nLEVEL 3 MULTI-SEED SUMMARY (model={MODEL_NAME!r})\n{'='*80}")
    print(f"{'seed':>6} {'VLEG':>8} {'HDR':>8} {'Reflexion':>11} {'ICL':>8}   ICL>VLEG?")
    icl_wins = 0
    for r in level3_results:
        res = r["results"]
        v, h, rf, i = (res["VLEG"]["accuracy"]*100, res["HDR"]["accuracy"]*100,
                       res["Reflexion"]["accuracy"]*100, res["ICL"]["accuracy"]*100)
        wins = i > v
        icl_wins += int(wins)
        print(f"{r['seed']:>6} {v:>7.1f}% {h:>7.1f}% {rf:>10.1f}% {i:>7.1f}%   {wins}")

    print(f"\nICL beat VLEG in {icl_wins}/{len(LEVEL3_SEEDS)} seeds on Level 3.")

    # Pooled accuracy across seeds (n = len(SEEDS)*16 per method)
    def pooled(method):
        total_correct, total_n = 0, 0
        for r in level3_results:
            acc = r["results"][method]["accuracy"]
            n = r["n_test"]
            total_correct += round(acc * n)
            total_n += n
        return total_correct, total_n

    print(f"\nPooled Level3 accuracy across {len(LEVEL3_SEEDS)} seeds (n={len(LEVEL3_SEEDS)*16} each):")
    for method in ["VLEG", "HDR", "Reflexion", "ICL"]:
        c, n = pooled(method)
        print(f"  {method:<10} {c}/{n} = {c/n*100:.1f}%")

    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
