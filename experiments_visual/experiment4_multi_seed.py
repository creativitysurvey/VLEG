"""
Multi-seed extension of the real-image Experiment 4 (K in {1,2,3,5}
sweep). Unlike Experiment 2's multi-seed extension, ALL FOUR K values
are re-tested here (not a reduced subset) -- the pilot's "K=1 best,
K=2=K=3 tied and worse, K=5 partial recovery" shape is exactly the
thing being checked for seed-robustness, so skipping any K value would
defeat the purpose.

Uses the same 3-seed list (42, 7, 123) and pilot scale
(n_test_per_combo=3, 48 items/seed) as other multi-seed real-image
checks, for direct comparability with the original single-seed pilot.

This is the most expensive multi-seed real-image check run so far in
this project (12 total K-conditions: 4 K-values x 3 seeds, each with
its own consolidation + 48-item evaluation) -- expect a long run.

Run:
    python -m experiments_visual.experiment4_multi_seed
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_visual.llm_client import check_connection, check_vision, MODEL_NAME
from core_visual.task_env import make_level3_split_large
from experiments_visual.experiment4_parameter_sensitivity import run_condition, K_VALUES

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs_visual"
OUTPUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUTPUT_DIR / "experiment4_visual_multiseed_results.json"

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

        seed_conditions = []
        for K in K_VALUES:
            r = run_condition(K, train_combos, test_items)
            seed_conditions.append(r)
            # checkpoint after EVERY K-condition, not just every seed
            per_seed_snapshot = per_seed_results + [{"seed": seed, "conditions": seed_conditions}]
            OUT_PATH.write_text(json.dumps({"per_seed": per_seed_snapshot}, indent=2))
            print(f"  [checkpoint] seed={seed} K={K} complete, written to {OUT_PATH}")

        per_seed_results.append({"seed": seed, "conditions": seed_conditions})

    # Pool across seeds, per K value
    pooled_by_k = {}
    for K in K_VALUES:
        all_correct, all_n = 0, 0
        all_masked_correct, all_masked_n = 0, 0
        all_seen_correct, all_seen_n = 0, 0
        all_tokens_per_ep = []
        for seed_result in per_seed_results:
            cond = next(c for c in seed_result["conditions"] if c["K"] == K)
            all_correct += cond["n_correct"]
            all_n += cond["n_total"]
            if cond["masked_combo_accuracy"] is not None:
                all_masked_correct += round(cond["masked_combo_accuracy"] * cond["n_masked"])
                all_masked_n += cond["n_masked"]
            if cond["seen_combo_accuracy"] is not None:
                all_seen_correct += round(cond["seen_combo_accuracy"] * cond["n_seen"])
                all_seen_n += cond["n_seen"]
            all_tokens_per_ep.append(cond["tokens_per_episode"])
        pooled_by_k[K] = {
            "accuracy": all_correct / all_n, "n_correct": all_correct, "n_total": all_n,
            "masked_accuracy": all_masked_correct / all_masked_n if all_masked_n else None,
            "n_masked": all_masked_n,
            "seen_accuracy": all_seen_correct / all_seen_n if all_seen_n else None,
            "n_seen": all_seen_n,
            "mean_tokens_per_episode": sum(all_tokens_per_ep) / len(all_tokens_per_ep),
        }

    output = {"per_seed": per_seed_results, "pooled_by_k": pooled_by_k}
    OUT_PATH.write_text(json.dumps(output, indent=2))

    print(f"\n\n{'='*100}\nPOOLED SUMMARY (across {len(SEEDS)} seeds)\n{'='*100}")
    print(f"{'K':<4} {'Overall':<10} {'Masked':<10} {'Seen':<10} {'Mean tok/ep':<14}")
    for K in K_VALUES:
        p = pooled_by_k[K]
        m = f"{p['masked_accuracy']*100:.1f}%" if p['masked_accuracy'] is not None else "N/A"
        s = f"{p['seen_accuracy']*100:.1f}%" if p['seen_accuracy'] is not None else "N/A"
        print(f"{K:<4} {p['accuracy']*100:>8.1f}%   {m:<10} {s:<10} {p['mean_tokens_per_episode']:<14.0f}")

    print(f"\nPer-seed breakdown:")
    for seed_result in per_seed_results:
        seed = seed_result["seed"]
        for cond in seed_result["conditions"]:
            m = f"{cond['masked_combo_accuracy']*100:.1f}%" if cond['masked_combo_accuracy'] is not None else "N/A"
            print(f"  seed={seed} K={cond['K']}: overall={cond['accuracy']*100:.1f}%  masked={m}")

    print(f"\nFull results written to {OUT_PATH}")


if __name__ == "__main__":
    main()
