"""
Multi-seed robustness check for Experiment 3's masked-combo result.

Runs the K=1 vs K=3 comparison across several different Level-3 masking
seeds (i.e., different random choices of which 4 combos get masked from
training) to check whether the K=1/K=3 tie on masked-combo accuracy
(both 75% under seed=42, post-bugfix) holds up or was a coincidence of
that one particular mask selection.

This re-runs full consolidation + evaluation per seed per condition —
expect this to take noticeably longer than a single run (roughly 5x the
LLM calls of one seed). Edit SEEDS below to run fewer if you want a
quicker first look.

Run:
    python -m experiments.experiment3_multi_seed
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.llm_client import check_connection, MODEL_NAME
from core.task_env import make_level3_split
from experiments.experiment3_iterative_refinement import run_condition, OUTPUT_DIR

SEEDS = [42, 7, 123, 2024, 99]  # seed=42 reproduces the earlier single-seed run


def main():
    print(f"Checking Ollama connection (model={MODEL_NAME!r})...")
    check_connection()

    all_results = []
    for seed in SEEDS:
        print(f"\n\n{'#'*70}\n# SEED = {seed}\n{'#'*70}")
        train_combos, test_combos = make_level3_split(mask_count=4, seed=seed)
        masked = [c for c in test_combos if c["masked_at_train"]]
        masked_desc = [{k: v for k, v in c.items() if k != "masked_at_train"} for c in masked]
        print(f"Masked combos for seed={seed}: {masked_desc}")

        r1 = run_condition(K=1, train_combos=train_combos, test_combos=test_combos,
                            label=f"seed{seed}_K1")
        r3 = run_condition(K=3, train_combos=train_combos, test_combos=test_combos,
                            label=f"seed{seed}_K3")

        delta_masked = None
        if r1["masked_combo_accuracy"] is not None and r3["masked_combo_accuracy"] is not None:
            delta_masked = r3["masked_combo_accuracy"] - r1["masked_combo_accuracy"]

        all_results.append({
            "seed": seed,
            "masked_combos": masked_desc,
            "k1_overall": r1["overall_accuracy"], "k1_masked": r1["masked_combo_accuracy"],
            "k3_overall": r3["overall_accuracy"], "k3_masked": r3["masked_combo_accuracy"],
            "delta_overall": r3["overall_accuracy"] - r1["overall_accuracy"],
            "delta_masked": delta_masked,
        })

    print(f"\n\n{'='*70}\nMULTI-SEED SUMMARY\n{'='*70}")
    header = f"{'seed':>6} {'K1 overall':>11} {'K3 overall':>11} {'K1 masked':>10} {'K3 masked':>10} {'delta masked':>13}"
    print(header)
    for r in all_results:
        k1m = f"{r['k1_masked']*100:.0f}%" if r["k1_masked"] is not None else "N/A"
        k3m = f"{r['k3_masked']*100:.0f}%" if r["k3_masked"] is not None else "N/A"
        dm = f"{r['delta_masked']*100:+.0f}pp" if r["delta_masked"] is not None else "N/A"
        print(f"{r['seed']:>6} {r['k1_overall']*100:>10.1f}% {r['k3_overall']*100:>10.1f}% "
              f"{k1m:>10} {k3m:>10} {dm:>13}")

    overall_deltas = [r["delta_overall"] for r in all_results]
    masked_deltas = [r["delta_masked"] for r in all_results if r["delta_masked"] is not None]

    print(f"\nMean overall accuracy delta (K3-K1): {statistics.mean(overall_deltas)*100:+.1f}pp", end="")
    if len(overall_deltas) > 1:
        print(f" (stdev {statistics.stdev(overall_deltas)*100:.1f}pp)")
    else:
        print()

    if masked_deltas:
        print(f"Mean masked-combo accuracy delta (K3-K1): {statistics.mean(masked_deltas)*100:+.1f}pp", end="")
        if len(masked_deltas) > 1:
            print(f" (stdev {statistics.stdev(masked_deltas)*100:.1f}pp)")
        else:
            print()
        n_ties = sum(1 for d in masked_deltas if d == 0.0)
        print(f"Ties (delta_masked == 0) across {len(masked_deltas)} seeds: {n_ties}")

    out_path = OUTPUT_DIR / "experiment3_multiseed_results.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
