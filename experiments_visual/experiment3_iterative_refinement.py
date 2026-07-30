"""
Experiment 3 (real-image / CLEVR, LARGE-SCALE) -- Iterative Refinement
Ablation. Supersedes the earlier pilot version: consolidation still uses
ONE representative image per (unmasked) combo, but evaluation now uses
10 held-out instances per combo (160 test queries total: 120 seen + 40
masked, versus the pilot's 16) -- see core_visual/task_env.py's
docstring for why.

Writes an incremental checkpoint after EVERY evaluated test item (not
just per K-condition) -- 160 real vision-model calls per condition is a
much larger commitment than the pilot's 16, and past sessions in this
project have dropped mid-run.

Run:
    python -m experiments_visual.experiment3_iterative_refinement
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_visual.llm_client import check_connection, check_vision, MODEL_NAME
from core_visual.task_env import make_level3_split_large, verifier_view_text, generator_view_text
from core_visual.refinement import refine_episode
from core_visual.memory import MemoryBank
from core_visual.inference import predict_hybrid

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs_visual"
OUTPUT_DIR.mkdir(exist_ok=True)

N_TEST_PER_COMBO = 10


def consolidate_memory(train_combos, K: int, log_prefix: str):
    bank = MemoryBank()
    stats = {"n_episodes": len(train_combos), "n_verified": 0, "n_discarded": 0}
    for idx, combo in enumerate(train_combos):
        vtext = verifier_view_text(combo)
        gtext = generator_view_text(combo)
        result = refine_episode(combo["image_path"], gtext, vtext, K=K)
        if result.verified:
            bank.add(combo, combo["image_path"], result.final_reflection)
            stats["n_verified"] += 1
            print(f"  [{log_prefix}] episode {idx+1}/{len(train_combos)} "
                  f"({combo['shape']}/{combo['color']}/{combo['material']}/{combo['size']}, "
                  f"true={combo['label']}): VERIFIED after {result.iterations_used} iter(s) -> "
                  f"\"{result.final_reflection}\"")
        else:
            stats["n_discarded"] += 1
            print(f"  [{log_prefix}] episode {idx+1}/{len(train_combos)} "
                  f"({combo['shape']}/{combo['color']}/{combo['material']}/{combo['size']}, "
                  f"true={combo['label']}): DISCARDED after {K} iter(s)")
    return bank, stats


def evaluate(bank: MemoryBank, test_items, log_prefix: str, checkpoint_cb=None):
    correct = 0
    per_item = []
    for idx, combo in enumerate(test_items):
        pred = predict_hybrid(bank, combo, k=None)
        is_correct = (pred == combo["label"])
        correct += int(is_correct)
        record = {
            "combo": {k: v for k, v in combo.items() if k != "masked_at_train"},
            "masked_at_train": combo["masked_at_train"],
            "predicted": pred, "correct": is_correct,
        }
        per_item.append(record)
        if (idx + 1) % 10 == 0 or (idx + 1) == len(test_items):
            print(f"  [{log_prefix}] evaluated {idx+1}/{len(test_items)} "
                  f"(running accuracy: {correct/(idx+1)*100:.1f}%)")
        if checkpoint_cb:
            checkpoint_cb(per_item)
    accuracy = correct / len(test_items) if test_items else 0.0
    return {"accuracy": accuracy, "n_correct": correct, "n_total": len(test_items), "per_item": per_item}


def run_condition(K: int, train_combos, test_items, label: str, out_path: Path) -> dict:
    print(f"\n=== Condition: {label} (K={K}) ===")
    bank, stats = consolidate_memory(train_combos, K=K, log_prefix=label)
    print(f"Memory bank: {len(bank)}/{len(train_combos)} verified.")

    def checkpoint(per_item_so_far):
        out_path.write_text(json.dumps({"label": label, "K": K,
                                          "consolidation_stats": stats,
                                          "per_item_partial": per_item_so_far}, indent=2))

    eval_results = evaluate(bank, test_items, log_prefix=label, checkpoint_cb=checkpoint)
    print(f"Task Accuracy ({label}, K={K}): {eval_results['accuracy']*100:.1f}% "
          f"({eval_results['n_correct']}/{eval_results['n_total']})")

    masked_items = [i for i in eval_results["per_item"] if i["masked_at_train"]]
    seen_items = [i for i in eval_results["per_item"] if not i["masked_at_train"]]
    masked_acc = sum(i["correct"] for i in masked_items) / len(masked_items) if masked_items else None
    seen_acc = sum(i["correct"] for i in seen_items) / len(seen_items) if seen_items else None
    print(f"  masked-combo accuracy (n={len(masked_items)}): "
          f"{masked_acc*100:.1f}%" if masked_acc is not None else "  masked-combo accuracy: N/A")
    print(f"  seen-combo accuracy (n={len(seen_items)}): "
          f"{seen_acc*100:.1f}%" if seen_acc is not None else "  seen-combo accuracy: N/A")

    return {
        "label": label, "K": K, "consolidation_stats": stats,
        "overall_accuracy": eval_results["accuracy"],
        "masked_combo_accuracy": masked_acc, "seen_combo_accuracy": seen_acc,
        "n_masked": len(masked_items), "n_seen": len(seen_items),
        "per_item": eval_results["per_item"],
    }


def main():
    print(f"Checking Ollama connection (model={MODEL_NAME!r})...")
    check_connection()

    train_combos, test_items = make_level3_split_large(mask_count=4, seed=42, n_test_per_combo=N_TEST_PER_COMBO)
    print(f"\nSanity-checking vision on the first training image...")
    check_vision(train_combos[0]["image_path"])

    print(f"\nLarge-scale Level 3 split: {len(train_combos)} training combos, "
          f"{len(test_items)} test items ({N_TEST_PER_COMBO} instances x 16 combos, "
          f"4 combos masked at train time).")

    k1_out_path = OUTPUT_DIR / "experiment3_visual_large_k1.json"
    k3_out_path = OUTPUT_DIR / "experiment3_visual_large_k3.json"
    summary_path = OUTPUT_DIR / "experiment3_visual_large_results.json"

    result_k1 = run_condition(K=1, train_combos=train_combos, test_items=test_items,
                               label="ablated_single_step", out_path=k1_out_path)
    summary_path.write_text(json.dumps({"k1_result": result_k1}, indent=2))
    print(f"  [checkpoint] K=1 full result written to {summary_path}")

    result_k3 = run_condition(K=3, train_combos=train_combos, test_items=test_items,
                               label="full_VLEG", out_path=k3_out_path)
    out = {"k1_result": result_k1, "k3_result": result_k3,
           "accuracy_delta_pp": (result_k3["overall_accuracy"] - result_k1["overall_accuracy"]) * 100}
    summary_path.write_text(json.dumps(out, indent=2))

    print("\n=== SUMMARY ===")
    print(f"K=1: {result_k1['overall_accuracy']*100:.1f}% overall (n={result_k1['n_masked']+result_k1['n_seen']}) "
          f"| masked(n={result_k1['n_masked']}): {result_k1['masked_combo_accuracy']*100:.1f}%")
    print(f"K=3: {result_k3['overall_accuracy']*100:.1f}% overall (n={result_k3['n_masked']+result_k3['n_seen']}) "
          f"| masked(n={result_k3['n_masked']}): {result_k3['masked_combo_accuracy']*100:.1f}%")
    print(f"Delta: {out['accuracy_delta_pp']:+.1f}pp")
    print(f"\nFull results written to {summary_path}")


if __name__ == "__main__":
    main()
