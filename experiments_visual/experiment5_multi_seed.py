"""
Multi-seed extension of the real-image Experiment 5 (memory quality
diagnostic). Exp5 has no held-out test set to enlarge -- it audits the
CONSOLIDATED reflections themselves -- so "scaling up" here means
pooling across multiple masking seeds (more independent train_combos
draws, hence more independent audited reflections) rather than more
test items per combo. Uses the same 3-seed list (42, 7, 123) used
throughout this project's other multi-seed checks.

Writes an incremental checkpoint after every seed.

Run:
    python -m experiments_visual.experiment5_multi_seed
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_visual.llm_client import check_connection, MODEL_NAME
from core_visual.task_env import make_level3_split_large
from experiments_visual.experiment5_memory_quality import (
    build_vleg_bank, build_single_step_bank_local, summarize,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs_visual"
OUTPUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUTPUT_DIR / "experiment5_visual_multiseed_results.json"

SEEDS = [42, 7, 123]


def pool_and_summarize(per_seed_results, strategy_name):
    """Pools per_item audits across all seeds for one strategy and
    recomputes rates over the full pooled set."""
    all_items = []
    for seed_result in per_seed_results:
        all_items.extend(seed_result[strategy_name]["per_item"])
    n = len(all_items)
    n_label_correct = sum(1 for a in all_items if a["label_correct"])
    correct_items = [a for a in all_items if a["label_correct"]]
    n_correct = len(correct_items)
    def rate(key, pool):
        return sum(a[key] for a in pool) / len(pool) if pool else None
    return {
        "strategy": strategy_name, "n_committed": n,
        "label_correct_rate": n_label_correct / n if n else None,
        "shape_hallucination_rate_given_correct": rate("shape_hallucination", correct_items),
        "color_hallucination_rate_given_correct": rate("color_hallucination", correct_items),
        "material_hallucination_rate_given_correct": rate("material_hallucination", correct_items),
        "spurious_size_rate_given_correct": rate("spurious_size", correct_items),
    }


def main():
    print(f"Checking Ollama connection (model={MODEL_NAME!r})...")
    check_connection()

    per_seed_results = []
    for seed in SEEDS:
        print(f"\n\n{'#'*70}\n# SEED = {seed}\n{'#'*70}")
        train_combos, _ = make_level3_split_large(mask_count=4, seed=seed, n_test_per_combo=1)

        print(f"\n=== single_step (unverified, always commits) ===")
        ss_bank = build_single_step_bank_local(train_combos, f"seed{seed}_single_step")
        ss_summary = summarize(ss_bank, "single_step")

        print(f"\n=== VLEG (K=3, verify-or-discard) ===")
        vleg_bank = build_vleg_bank(train_combos, K=3, log_prefix=f"seed{seed}_VLEG")
        vleg_summary = summarize(vleg_bank, "VLEG")

        seed_result = {"seed": seed, "single_step": ss_summary, "VLEG": vleg_summary}
        per_seed_results.append(seed_result)
        OUT_PATH.write_text(json.dumps({"per_seed": per_seed_results}, indent=2))
        print(f"  [checkpoint] seed {seed} complete, written to {OUT_PATH}")

    pooled_ss = pool_and_summarize(per_seed_results, "single_step")
    pooled_vleg = pool_and_summarize(per_seed_results, "VLEG")
    output = {"per_seed": per_seed_results, "pooled": {"single_step": pooled_ss, "VLEG": pooled_vleg}}
    OUT_PATH.write_text(json.dumps(output, indent=2))

    print(f"\n\n{'='*100}\nPOOLED SUMMARY (across {len(SEEDS)} seeds)\n{'='*100}")
    print(f"{'Strategy':<14} {'Committed':<10} {'Label-correct':<15} {'Shape-halluc':<14} "
          f"{'Color-halluc':<14} {'Material-halluc':<16} {'Spurious-size':<14}")
    for name, r in [("single_step", pooled_ss), ("VLEG", pooled_vleg)]:
        def fmt(v):
            return f"{v*100:.1f}%" if v is not None else "N/A"
        print(f"{r['strategy']:<14} {r['n_committed']:<10} {fmt(r.get('label_correct_rate')):<15} "
              f"{fmt(r.get('shape_hallucination_rate_given_correct')):<14} "
              f"{fmt(r.get('color_hallucination_rate_given_correct')):<14} "
              f"{fmt(r.get('material_hallucination_rate_given_correct')):<16} "
              f"{fmt(r.get('spurious_size_rate_given_correct')):<14}")

    print(f"\nFull results written to {OUT_PATH}")


if __name__ == "__main__":
    main()
