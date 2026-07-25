"""
Experiment 3 — Iterative Refinement Ablation (paper Section IV-D).

Compares K=1 (single-step reflection, no verification loop — mimics
Reflexion-style baseline) against K=3 (full VLEG cyclic refinement) on the
Level 3 Experience-Transfer task (masked combinations).

Run:
    cd /home/claude/vleg_exp3
    python -m experiments.experiment3_iterative_refinement

Requires `ollama serve` running locally with the model set in
core/llm_client.py (MODEL_NAME).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.llm_client import check_connection, MODEL_NAME
from core.task_env import make_level3_split, episode_text, generator_view_text
from core.refinement import refine_episode
from core.memory import MemoryBank
from core.inference import predict

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def consolidate_memory(train_combos, K: int, log_prefix: str) -> tuple[MemoryBank, dict]:
    bank = MemoryBank()
    stats = {"n_episodes": len(train_combos), "n_verified": 0, "n_discarded": 0,
              "total_iterations": 0}
    for idx, combo in enumerate(train_combos):
        etext = episode_text(combo)
        gtext = generator_view_text(combo)
        result = refine_episode(gtext, etext, K=K)
        stats["total_iterations"] += result.iterations_used
        if result.verified:
            bank.add(combo, etext, result.final_reflection)
            stats["n_verified"] += 1
            print(f"  [{log_prefix}] episode {idx+1}/{len(train_combos)}: "
                  f"VERIFIED after {result.iterations_used} iter(s) -> "
                  f"\"{result.final_reflection}\"")
        else:
            stats["n_discarded"] += 1
            print(f"  [{log_prefix}] episode {idx+1}/{len(train_combos)}: "
                  f"DISCARDED after {K} iter(s) (no valid reflection found)")
    return bank, stats


def evaluate(bank: MemoryBank, test_combos, k_retrieval: int = 3) -> dict:
    correct = 0
    per_item = []
    for combo in test_combos:
        pred = predict(bank, combo, k=k_retrieval)
        is_correct = (pred == combo["label"])
        correct += int(is_correct)
        per_item.append({
            "combo": {k: v for k, v in combo.items() if k != "masked_at_train"},
            "masked_at_train": combo["masked_at_train"],
            "predicted": pred,
            "correct": is_correct,
        })
    accuracy = correct / len(test_combos) if test_combos else 0.0
    return {"accuracy": accuracy, "n_correct": correct, "n_total": len(test_combos),
            "per_item": per_item}


def run_condition(K: int, train_combos, test_combos, label: str) -> dict:
    print(f"\n=== Condition: {label} (K={K}) ===")
    print(f"Consolidating memory from {len(train_combos)} training episodes...")
    t0 = time.time()
    bank, consolidation_stats = consolidate_memory(train_combos, K=K, log_prefix=label)
    t1 = time.time()
    print(f"Memory bank size: {len(bank)} / {len(train_combos)} episodes verified.")
    print(f"Evaluating on {len(test_combos)} test queries (Level 3, incl. masked combos)...")
    eval_results = evaluate(bank, test_combos)
    t2 = time.time()
    print(f"Task Accuracy ({label}, K={K}): {eval_results['accuracy']*100:.1f}% "
          f"({eval_results['n_correct']}/{eval_results['n_total']})")

    # Break out accuracy on masked-at-train vs seen-at-train combos —
    # this is the part of Level 3 that specifically tests generalization.
    masked_items = [i for i in eval_results["per_item"] if i["masked_at_train"]]
    seen_items = [i for i in eval_results["per_item"] if not i["masked_at_train"]]
    masked_acc = sum(i["correct"] for i in masked_items) / len(masked_items) if masked_items else None
    seen_acc = sum(i["correct"] for i in seen_items) / len(seen_items) if seen_items else None

    return {
        "label": label, "K": K,
        "consolidation_stats": consolidation_stats,
        "consolidation_time_sec": round(t1 - t0, 1),
        "eval_time_sec": round(t2 - t1, 1),
        "overall_accuracy": eval_results["accuracy"],
        "masked_combo_accuracy": masked_acc,
        "seen_combo_accuracy": seen_acc,
        "per_item": eval_results["per_item"],
    }


def main():
    print(f"Checking Ollama connection (model={MODEL_NAME!r})...")
    check_connection()

    train_combos, test_combos = make_level3_split(mask_count=4, seed=42)
    print(f"\nLevel 3 split: {len(train_combos)} training combos, "
          f"{len(test_combos)} test combos (4 of them masked at train time).")

    result_k1 = run_condition(K=1, train_combos=train_combos,
                               test_combos=test_combos, label="ablated_single_step")
    result_k3 = run_condition(K=3, train_combos=train_combos,
                               test_combos=test_combos, label="full_VLEG")

    delta = (result_k3["overall_accuracy"] - result_k1["overall_accuracy"]) * 100
    print("\n=== SUMMARY ===")
    print(f"K=1 (single-step, ablated): {result_k1['overall_accuracy']*100:.1f}% overall "
          f"(masked-combo acc: {result_k1['masked_combo_accuracy']*100:.1f}%)")
    print(f"K=3 (full VLEG):            {result_k3['overall_accuracy']*100:.1f}% overall "
          f"(masked-combo acc: {result_k3['masked_combo_accuracy']*100:.1f}%)")
    print(f"Absolute accuracy delta (K=3 minus K=1): {delta:+.1f} percentage points")

    out = {
        "experiment": "Experiment 3 - Iterative Refinement Ablation",
        "model": MODEL_NAME,
        "k1_result": result_k1,
        "k3_result": result_k3,
        "accuracy_delta_pp": delta,
    }
    out_path = OUTPUT_DIR / "experiment3_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
