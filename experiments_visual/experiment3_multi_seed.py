"""
Multi-seed robustness check for the real-image (CLEVR) Experiment 3
pilot. Same 5-seed list used throughout this project's symbolic
multi-seed checks, for consistency.

Writes an incremental checkpoint to
outputs_visual/experiment3_visual_multiseed_results.json after EVERY
seed completes (not just at the very end) -- this pipeline involves
real API calls that can take a while per seed, and past experience in
this project shows sessions/connections can drop mid-run.

Run:
    python -m experiments_visual.experiment3_multi_seed
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_visual.llm_client import check_connection, check_vision, MODEL_NAME
from core_visual.task_env import make_level3_split
from experiments_visual.experiment3_iterative_refinement import run_condition, OUTPUT_DIR

SEEDS = [42, 7, 123, 2024, 99]  # same seed list as experiments/experiment3_multi_seed.py

OUT_PATH = OUTPUT_DIR / "experiment3_visual_multiseed_results.json"


def save_partial(results):
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"  [checkpoint] {len(results)} seed(s) written to {OUT_PATH}")


def main():
    print(f"Checking Ollama connection (model={MODEL_NAME!r})...")
    check_connection()

    # quick vision sanity check once, using seed=42's first training image
    train0, _ = make_level3_split(mask_count=4, seed=42)
    check_vision(train0[0]["image_path"])

    results = []
    for seed in SEEDS:
        print(f"\n\n{'#'*70}\n# SEED = {seed}\n{'#'*70}")
        train, test = make_level3_split(mask_count=4, seed=seed)
        masked_desc = [{k: v for k, v in c.items() if k not in ("masked_at_train", "image_path")}
                        for c in test if c["masked_at_train"]]
        print(f"Masked combos for seed={seed}: {masked_desc}")

        r1 = run_condition(K=1, train_combos=train, test_combos=test, label=f"seed{seed}_K1")
        r3 = run_condition(K=3, train_combos=train, test_combos=test, label=f"seed{seed}_K3")

        delta_masked = None
        if r1["masked_combo_accuracy"] is not None and r3["masked_combo_accuracy"] is not None:
            delta_masked = r3["masked_combo_accuracy"] - r1["masked_combo_accuracy"]

        results.append({
            "seed": seed, "masked_combos": masked_desc,
            "k1_overall": r1["overall_accuracy"], "k1_masked": r1["masked_combo_accuracy"],
            "k3_overall": r3["overall_accuracy"], "k3_masked": r3["masked_combo_accuracy"],
            "delta_overall": r3["overall_accuracy"] - r1["overall_accuracy"],
            "delta_masked": delta_masked,
            "k1_bank_verified": r1["consolidation_stats"]["n_verified"],
            "k3_bank_verified": r3["consolidation_stats"]["n_verified"],
        })
        save_partial(results)

    print(f"\n\n{'='*90}\nMULTI-SEED SUMMARY (real-image CLEVR pipeline)\n{'='*90}")
    header = f"{'seed':>6} {'K1 overall':>11} {'K3 overall':>11} {'K1 masked':>10} {'K3 masked':>10} {'delta masked':>13}"
    print(header)
    for r in results:
        k1m = f"{r['k1_masked']*100:.0f}%" if r["k1_masked"] is not None else "N/A"
        k3m = f"{r['k3_masked']*100:.0f}%" if r["k3_masked"] is not None else "N/A"
        dm = f"{r['delta_masked']*100:+.0f}pp" if r["delta_masked"] is not None else "N/A"
        print(f"{r['seed']:>6} {r['k1_overall']*100:>10.1f}% {r['k3_overall']*100:>10.1f}% "
              f"{k1m:>10} {k3m:>10} {dm:>13}")

    overall_deltas = [r["delta_overall"] for r in results]
    masked_deltas = [r["delta_masked"] for r in results if r["delta_masked"] is not None]
    import statistics
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
        n_negative = sum(1 for d in masked_deltas if d < 0)
        print(f"Seeds where K3 masked < K1 masked: {n_negative}/{len(masked_deltas)}")

    print(f"\nFull results written to {OUT_PATH}")


if __name__ == "__main__":
    main()
