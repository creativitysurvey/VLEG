"""
Multi-seed extension of the real-image Experiment 2 (grounding modality
ablation) -- HYBRID ONLY. Abstract-Only and Raw-Only are deliberately
NOT re-tested at multi-seed scale: their pilot-run failure modes are
already mechanistically understood (Abstract-Only's query text carries
zero target-identifying information without an image, a task-design
degeneracy, not a capability question; Raw-Only exercises the same
multi-image attention collapse independently confirmed 3+ times
elsewhere in this project) -- more seeds would only re-confirm already-
understood degenerate behavior, not add information. Hybrid's pilot
result (87.5% overall, but 50% on masked combinations, n=12) is the one
genuinely open question worth more data.

Uses the same 3-seed list (42, 7, 123) as other multi-seed real-image
checks, at the same pilot scale (n_test_per_combo=3, 48 items/seed) as
the original Experiment 2 pilot, for direct comparability.

Run:
    python -m experiments_visual.experiment2_multi_seed
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_visual.llm_client import check_connection, check_vision, MODEL_NAME
from core_visual.task_env import make_level3_split_large
from core_visual.inference import predict_hybrid
from experiments_visual.experiment2_grounding_ablation import consolidate_memory, evaluate_modality

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs_visual"
OUTPUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUTPUT_DIR / "experiment2_visual_multiseed_results.json"

SEEDS = [42, 7, 123]
N_TEST_PER_COMBO = 3


def main():
    print(f"Checking Ollama connection (model={MODEL_NAME!r})...")
    check_connection()

    per_seed_results = []
    for seed in SEEDS:
        print(f"\n\n{'#'*70}\n# SEED = {seed}\n{'#'*70}")
        train_combos, test_items = make_level3_split_large(
            mask_count=4, seed=seed, n_test_per_combo=N_TEST_PER_COMBO)
        if seed == SEEDS[0]:
            check_vision(train_combos[0]["image_path"])

        print(f"\n=== Building VLEG (K=3) memory bank, seed={seed} ===")
        bank, stats = consolidate_memory(train_combos, K=3, log_prefix=f"seed{seed}_VLEG")
        print(f"Memory bank: {len(bank)}/{len(train_combos)} verified.")

        result = evaluate_modality(bank, test_items, predict_hybrid, f"seed{seed}_hybrid")
        seed_result = {"seed": seed, "consolidation_stats": stats, "hybrid": result}
        per_seed_results.append(seed_result)
        OUT_PATH.write_text(json.dumps({"per_seed": per_seed_results}, indent=2))
        print(f"  [checkpoint] seed {seed} complete, written to {OUT_PATH}")

    # Pool across seeds
    all_items = []
    for seed_result in per_seed_results:
        all_items.extend(seed_result["hybrid"]["per_item"])
    n = len(all_items)
    n_correct = sum(1 for i in all_items if i["correct"])
    masked_items = [i for i in all_items if i["masked_at_train"]]
    seen_items = [i for i in all_items if not i["masked_at_train"]]
    masked_acc = sum(i["correct"] for i in masked_items) / len(masked_items) if masked_items else None
    seen_acc = sum(i["correct"] for i in seen_items) / len(seen_items) if seen_items else None

    pooled = {
        "n_total": n, "n_correct": n_correct, "accuracy": n_correct / n,
        "n_masked": len(masked_items), "masked_combo_accuracy": masked_acc,
        "n_seen": len(seen_items), "seen_combo_accuracy": seen_acc,
    }
    output = {"per_seed": per_seed_results, "pooled_hybrid": pooled}
    OUT_PATH.write_text(json.dumps(output, indent=2))

    print(f"\n\n{'='*90}\nPOOLED HYBRID SUMMARY (across {len(SEEDS)} seeds)\n{'='*90}")
    print(f"Overall: {pooled['accuracy']*100:.1f}% ({n_correct}/{n})")
    print(f"Masked (n={pooled['n_masked']}): {pooled['masked_combo_accuracy']*100:.1f}%")
    print(f"Seen (n={pooled['n_seen']}): {pooled['seen_combo_accuracy']*100:.1f}%")
    print(f"\nPer-seed breakdown:")
    for r in per_seed_results:
        h = r["hybrid"]
        print(f"  seed={r['seed']}: overall={h['accuracy']*100:.1f}%  "
              f"masked={h['masked_combo_accuracy']*100:.1f}%  seen={h['seen_combo_accuracy']*100:.1f}%")

    print(f"\nFull results written to {OUT_PATH}")


if __name__ == "__main__":
    main()
