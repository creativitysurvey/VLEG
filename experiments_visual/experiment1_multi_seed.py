"""
Multi-seed extension of the real-image Experiment 1 (SOTA comparison:
VLEG vs ICL, Reflexion, HDR). ALL FOUR methods are re-tested across 3
seeds (42, 7, 123) at the same pilot scale as the original
(n_test_per_combo=2, 32 items/seed) -- this is a deliberate choice to
get complete, uniform multi-seed coverage rather than selectively
skipping the baselines whose pilot failure modes are already
mechanistically understood (see DEVIATIONS_LOG.md #41); the tradeoff is
that this is, by a wide margin, the most expensive multi-seed check run
in this project so far, dominated by HDR's cost (each HDR consolidation
iteration attaches up to 12 training images to BOTH a hypothesis-
generation call and a verification call, times up to K=3 iterations).

Writes an incremental checkpoint after every method completes within a
seed, and after every seed completes.

Run:
    python -m experiments_visual.experiment1_multi_seed
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_visual.llm_client import check_connection, check_vision, MODEL_NAME
from core_visual.task_env import make_level3_split_large
from core_visual.inference import predict_hybrid
from core_visual.baselines import (
    build_icl_bank, icl_predict,
    build_reflexion_bank, reflexion_predict,
    hdr_generate_global_hypothesis, hdr_predict,
)
from experiments_visual.experiment1_sota_comparison import build_vleg_bank, evaluate

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs_visual"
OUTPUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUTPUT_DIR / "experiment1_visual_multiseed_results.json"

SEEDS = [42, 7, 123]
N_TEST_PER_COMBO = 2


def run_seed(seed, train_combos, test_items, results_accum):
    seed_results = {}

    def checkpoint(method_name, per_item_so_far):
        seed_results[method_name] = {"per_item_partial": per_item_so_far}
        snapshot = dict(results_accum)
        snapshot[seed] = seed_results
        OUT_PATH.write_text(json.dumps(snapshot, indent=2))

    print(f"\n=== VLEG (K=3), seed={seed} ===")
    t0 = time.time()
    vleg_bank = build_vleg_bank(train_combos, K=3, log_prefix=f"seed{seed}_VLEG")
    r = evaluate(lambda c: predict_hybrid(vleg_bank, c, k=None), test_items, f"seed{seed}_VLEG",
                 checkpoint_cb=lambda p: checkpoint("VLEG", p))
    r["time_sec"] = round(time.time() - t0, 1)
    seed_results["VLEG"] = r
    print(f"VLEG accuracy: {r['accuracy']*100:.1f}%  [{r['time_sec']:.1f}s]")

    print(f"\n=== ICL, seed={seed} ===")
    t0 = time.time()
    icl_bank = build_icl_bank(train_combos)
    r = evaluate(lambda c: icl_predict(icl_bank, c), test_items, f"seed{seed}_ICL",
                 checkpoint_cb=lambda p: checkpoint("ICL", p))
    r["time_sec"] = round(time.time() - t0, 1)
    seed_results["ICL"] = r
    print(f"ICL accuracy: {r['accuracy']*100:.1f}%  [{r['time_sec']:.1f}s]")

    print(f"\n=== Reflexion (K=3), seed={seed} ===")
    t0 = time.time()
    reflexion_bank = build_reflexion_bank(train_combos, K=3, log_prefix=f"seed{seed}_reflexion")
    r = evaluate(lambda c: reflexion_predict(reflexion_bank, c), test_items, f"seed{seed}_Reflexion",
                 checkpoint_cb=lambda p: checkpoint("Reflexion", p))
    r["time_sec"] = round(time.time() - t0, 1)
    seed_results["Reflexion"] = r
    print(f"Reflexion accuracy: {r['accuracy']*100:.1f}%  [{r['time_sec']:.1f}s]")

    print(f"\n=== HDR (K=3), seed={seed} -- most expensive, up to 12 images/call ===")
    t0 = time.time()
    hdr_result = hdr_generate_global_hypothesis(train_combos, K=3, log_prefix=f"seed{seed}_HDR")
    print(f"HDR hypothesis (verified={hdr_result.verified}): \"{hdr_result.hypothesis}\"")
    r = evaluate(lambda c: hdr_predict(hdr_result.hypothesis, c), test_items, f"seed{seed}_HDR",
                 checkpoint_cb=lambda p: checkpoint("HDR", p))
    r["time_sec"] = round(time.time() - t0, 1)
    r["hypothesis"] = hdr_result.hypothesis
    r["hypothesis_verified"] = hdr_result.verified
    seed_results["HDR"] = r
    print(f"HDR accuracy: {r['accuracy']*100:.1f}%  (verified={hdr_result.verified})  [{r['time_sec']:.1f}s]")

    return seed_results


def main():
    print(f"Checking Ollama connection (model={MODEL_NAME!r})...")
    check_connection()

    results_accum = {}
    for seed in SEEDS:
        print(f"\n\n{'#'*70}\n# SEED = {seed}\n{'#'*70}")
        train_combos, test_items = make_level3_split_large(
            mask_count=4, seed=seed, n_test_per_combo=N_TEST_PER_COMBO)
        if seed == SEEDS[0]:
            check_vision(train_combos[0]["image_path"])
        print(f"Split: {len(train_combos)} train combos, {len(test_items)} test items.")

        seed_results = run_seed(seed, train_combos, test_items, results_accum)
        results_accum[seed] = seed_results
        OUT_PATH.write_text(json.dumps(results_accum, indent=2))
        print(f"\n  [checkpoint] seed {seed} fully complete, written to {OUT_PATH}")

    # Pool across seeds, per method
    pooled = {}
    for method in ["VLEG", "ICL", "Reflexion", "HDR"]:
        all_correct, all_n = 0, 0
        for seed in SEEDS:
            r = results_accum[seed][method]
            all_correct += r["n_correct"]
            all_n += r["n_total"]
        pooled[method] = {"accuracy": all_correct / all_n, "n_correct": all_correct, "n_total": all_n}

    output = {"per_seed": results_accum, "pooled": pooled}
    OUT_PATH.write_text(json.dumps(output, indent=2))

    print(f"\n\n{'='*90}\nPOOLED SUMMARY (across {len(SEEDS)} seeds)\n{'='*90}")
    for method, p in pooled.items():
        print(f"{method:<12} {p['accuracy']*100:>7.1f}%  ({p['n_correct']}/{p['n_total']})")

    print(f"\nPer-seed breakdown:")
    for seed in SEEDS:
        print(f"  seed={seed}:")
        for method in ["VLEG", "ICL", "Reflexion", "HDR"]:
            r = results_accum[seed][method]
            print(f"    {method:<12} {r['accuracy']*100:>6.1f}%")

    print(f"\nFull results written to {OUT_PATH}")


if __name__ == "__main__":
    main()
