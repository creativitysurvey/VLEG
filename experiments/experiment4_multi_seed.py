"""
Multi-seed robustness check for Experiment 4.

Part A — Level 3 (has real randomness: which 4 combos get masked): full
K in {1,2,3,5} sweep across 3 masking seeds, to check whether the
"plateau exactly at K=3" finding (K=3 and K=5 producing IDENTICAL
iteration counts/tokens/accuracy under seed=42) replicates.

Part B — Level 2 (no masking, nothing structural to vary across
"seeds"): repeats ONLY K=3 and K=5 once more, to check whether the
"no plateau — still climbing K=3 to K=5" finding (87.5% -> 93.8% under
the first run) is a real signal or run-to-run LLM/embedding noise.

Run:
    python -m experiments.experiment4_multi_seed
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.llm_client import check_connection, MODEL_NAME
from core.task_env import make_level3_split, all_combinations
from experiments.experiment4_parameter_sensitivity import run_condition, OUTPUT_DIR, K_VALUES

LEVEL3_SEEDS = [42, 7, 123]


def main():
    print(f"Checking Ollama connection (model={MODEL_NAME!r})...")
    check_connection()

    level2_combos = all_combinations()

    out_path = OUTPUT_DIR / "experiment4_multiseed_results.json"
    state = {"level3_by_seed": [], "level2_repeat": []}

    def save_partial():
        out_path.write_text(json.dumps(state, indent=2))
        print(f"  [checkpoint] partial results written to {out_path}")

    # --- Part A: Level 3 across 3 masking seeds, full K sweep ---
    level3_results = []
    for seed in LEVEL3_SEEDS:
        print(f"\n\n{'#'*70}\n# LEVEL 3 — SEED = {seed}\n{'#'*70}")
        train, test = make_level3_split(mask_count=4, seed=seed)
        for K in K_VALUES:
            r = run_condition(f"Level3_seed{seed}", train, test, K)
            r["seed"] = seed
            level3_results.append(r)
            state["level3_by_seed"] = level3_results
            save_partial()

    print(f"\n\n{'='*90}\nLEVEL 3 MULTI-SEED SUMMARY (plateau check)\n{'='*90}")
    print(f"{'seed':>6} " + " ".join(f"K={k:<10}" for k in K_VALUES))
    for seed in LEVEL3_SEEDS:
        row = [r for r in level3_results if r["seed"] == seed]
        accs = " ".join(f"{r['accuracy']*100:>9.1f}% " for r in row)
        print(f"{seed:>6} {accs}")
    print(f"\n{'seed':>6} " + " ".join(f"K={k} tok/ep" for k in K_VALUES))
    for seed in LEVEL3_SEEDS:
        row = [r for r in level3_results if r["seed"] == seed]
        toks = " ".join(f"{r['tokens_per_episode']:>10.0f}" for r in row)
        print(f"{seed:>6} {toks}")
    # Check: does K=3 == K=5 (bank_size, accuracy) per seed, matching the plateau claim?
    print("\nPer-seed check — does K=3 exactly match K=5 (the 'true plateau' signature)?")
    for seed in LEVEL3_SEEDS:
        row = {r["K"]: r for r in level3_results if r["seed"] == seed}
        same = (row[3]["bank_size"] == row[5]["bank_size"] and
                abs(row[3]["accuracy"] - row[5]["accuracy"]) < 1e-9)
        print(f"  seed={seed}: K3 bank={row[3]['bank_size']} acc={row[3]['accuracy']*100:.1f}%  |  "
              f"K5 bank={row[5]['bank_size']} acc={row[5]['accuracy']*100:.1f}%  |  "
              f"IDENTICAL={same}")

    # --- Part B: Level 2, repeat K=3 and K=5 once more ---
    print(f"\n\n{'#'*70}\n# LEVEL 2 — REPEAT RUN (K=3, K=5 only)\n{'#'*70}")
    level2_repeat = []
    for K in [3, 5]:
        r = run_condition("Level2_repeat", level2_combos, level2_combos, K)
        level2_repeat.append(r)
        state["level2_repeat"] = level2_repeat
        save_partial()

    print(f"\n{'='*70}\nLEVEL 2 REPEATABILITY CHECK\n{'='*70}")
    print("NOTE: 'Run 1' below is the 7B reference (87.5%/93.8%) from the")
    print("original experiment4_parameter_sensitivity.py run — this script")
    print("has not yet had a matching 70B single-run baseline to compare")
    print("against, so treat this comparison as informational only, not a")
    print("same-model determinism check (unlike experiment1_multi_seed.py's")
    print("ORIGINAL dict, which IS kept in sync with the current model).")
    print("Run 1 (7B reference):  K=3 -> 87.5%   K=5 -> 93.8%")
    r3 = next(r for r in level2_repeat if r["K"] == 3)
    r5 = next(r for r in level2_repeat if r["K"] == 5)
    print(f"Run 2 (this run):      K=3 -> {r3['accuracy']*100:.1f}%   K=5 -> {r5['accuracy']*100:.1f}%")

    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
