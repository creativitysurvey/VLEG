"""
Multi-seed robustness check for Experiment 2's grounding-modality reversal
on Level 3 (Hybrid worst, Raw-Only best — the opposite of the paper's
hypothesis, under seed=42). Re-runs Level 3 (build VLEG bank + evaluate
3 modalities) across several different masking seeds. Level 2 has no
masking so it's identical every time — run once, not per seed.

Uses the SAME seed list as experiment3_multi_seed.py for consistency.

Run:
    python -m experiments.experiment2_multi_seed
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.llm_client import check_connection, MODEL_NAME
from core.task_env import make_level3_split, all_combinations
from experiments.experiment2_grounding_ablation import run_level, OUTPUT_DIR, MODALITIES

SEEDS = [42, 7, 123, 2024, 99]  # seed=42 reproduces the earlier single-seed run


def main():
    print(f"Checking Ollama connection (model={MODEL_NAME!r})...")
    check_connection()

    level2_combos = all_combinations()
    print("\n\n" + "#"*70 + "\n# LEVEL 2 (no masking — run once, identical across seeds)\n" + "#"*70)
    level2_result = run_level("Level2", level2_combos, level2_combos)

    level3_results = []
    for seed in SEEDS:
        print(f"\n\n{'#'*70}\n# LEVEL 3 — SEED = {seed}\n{'#'*70}")
        train, test = make_level3_split(mask_count=4, seed=seed)
        r = run_level(f"Level3_seed{seed}", train, test)
        r["seed"] = seed
        level3_results.append(r)

    print(f"\n\n{'='*90}\nLEVEL 3 MULTI-SEED SUMMARY\n{'='*90}")
    header = f"{'seed':>6} {'Hybrid':>10} {'Abstract':>10} {'Raw-Only':>10}   {'Hyb-masked':>11} {'Abs-masked':>11} {'Raw-masked':>11}"
    print(header)
    for r in level3_results:
        h = r["modality_results"]["hybrid"]["accuracy"] * 100
        a = r["modality_results"]["abstract_only"]["accuracy"] * 100
        raw = r["modality_results"]["raw_only"]["accuracy"] * 100
        hm = r["masked_vs_seen_breakdown"]["hybrid"]["masked_accuracy"]
        am = r["masked_vs_seen_breakdown"]["abstract_only"]["masked_accuracy"]
        rm = r["masked_vs_seen_breakdown"]["raw_only"]["masked_accuracy"]
        hm_s = f"{hm*100:.0f}%" if hm is not None else "N/A"
        am_s = f"{am*100:.0f}%" if am is not None else "N/A"
        rm_s = f"{rm*100:.0f}%" if rm is not None else "N/A"
        print(f"{r['seed']:>6} {h:>9.1f}% {a:>9.1f}% {raw:>9.1f}%   {hm_s:>11} {am_s:>11} {rm_s:>11}")

    # Pooled accuracy across all 5 seeds (n=80 test trials each modality)
    def pooled(modality, key="accuracy"):
        total_correct = sum(r["modality_results"][modality]["n_correct"] for r in level3_results)
        total_n = sum(r["modality_results"][modality]["n_total"] for r in level3_results)
        return total_correct, total_n

    print(f"\nPooled Level3 overall accuracy across {len(SEEDS)} seeds (n={len(SEEDS)*16} each):")
    for modality in MODALITIES:
        c, n = pooled(modality)
        print(f"  {modality:<15} {c}/{n} = {c/n*100:.1f}%")

    # Pooled masked-only accuracy (n=4 per seed -> n=20 pooled)
    def pooled_masked(modality):
        total_correct, total_n = 0, 0
        for r in level3_results:
            items = r["raw_results"][modality]["per_item"]
            masked_items = [i for i in items if i["masked_at_train"]]
            total_correct += sum(i["correct"] for i in masked_items)
            total_n += len(masked_items)
        return total_correct, total_n

    print(f"\nPooled Level3 MASKED-combo accuracy across {len(SEEDS)} seeds (n={len(SEEDS)*4} each):")
    for modality in MODALITIES:
        c, n = pooled_masked(modality)
        print(f"  {modality:<15} {c}/{n} = {c/n*100:.1f}%")

    out_path = OUTPUT_DIR / "experiment2_multiseed_results.json"
    out_path.write_text(json.dumps({"level2": level2_result, "level3_by_seed": level3_results}, indent=2))
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
