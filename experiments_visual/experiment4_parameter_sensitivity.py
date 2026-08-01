"""
Experiment 4 (real-image / CLEVR) -- Parameter Sensitivity and Overhead
Analysis. Sweeps K in {1, 2, 3, 5}, tracking task accuracy AND
consolidation token overhead, mirroring the symbolic pipeline's
experiments/experiment4_parameter_sensitivity.py but with real images.

PILOT SCALE: uses n_test_per_combo=3 (48 test items: 36 seen + 12
masked) rather than the 160-item scale used for the large Experiment 3
re-run, to keep this tractable as a first pass -- Exp4's primary
purpose is token-overhead scaling, which the consolidation side alone
already measures regardless of test-set size; accuracy at this smaller
scale is a secondary, lower-precision read that can be scaled up later
if the pilot's results warrant it.

Writes an incremental checkpoint after every evaluated test item.

Run:
    python -m experiments_visual.experiment4_parameter_sensitivity
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_visual.llm_client import check_connection, check_vision, MODEL_NAME, reset_token_usage, get_token_usage
from core_visual.task_env import make_level3_split_large, verifier_view_text, generator_view_text
from core_visual.refinement import refine_episode
from core_visual.memory import MemoryBank
from core_visual.inference import predict_hybrid

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs_visual"
OUTPUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUTPUT_DIR / "experiment4_visual_results.json"

K_VALUES = [1, 2, 3, 5]
N_TEST_PER_COMBO = 3


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
                  f"true={combo['label']}): VERIFIED after {result.iterations_used} iter(s)")
        else:
            stats["n_discarded"] += 1
            print(f"  [{log_prefix}] episode {idx+1}/{len(train_combos)} "
                  f"({combo['shape']}/{combo['color']}/{combo['material']}/{combo['size']}, "
                  f"true={combo['label']}): DISCARDED after {K} iter(s)")
    return bank, stats


def evaluate(bank, test_items, log_prefix, checkpoint_cb=None):
    correct = 0
    per_item = []
    for idx, combo in enumerate(test_items):
        pred = predict_hybrid(bank, combo, k=None)
        is_correct = (pred == combo["label"])
        correct += int(is_correct)
        record = {"combo": {k: v for k, v in combo.items() if k != "masked_at_train"},
                   "masked_at_train": combo["masked_at_train"], "predicted": pred, "correct": is_correct}
        per_item.append(record)
        if checkpoint_cb:
            checkpoint_cb(per_item)
    print(f"  [{log_prefix}] evaluated {len(test_items)}/{len(test_items)} "
          f"(accuracy: {correct/len(test_items)*100:.1f}%)")
    return {"accuracy": correct / len(test_items), "n_correct": correct, "n_total": len(test_items),
            "per_item": per_item}


def run_condition(K: int, train_combos, test_items) -> dict:
    label = f"K{K}"
    print(f"\n=== Condition: K={K} ===")
    reset_token_usage()
    t0 = time.time()
    bank, stats = consolidate_memory(train_combos, K=K, log_prefix=label)
    t1 = time.time()
    consolidation_usage = get_token_usage()  # tokens used ONLY during consolidation
    print(f"Memory bank: {len(bank)}/{len(train_combos)} verified. "
          f"Consolidation: {consolidation_usage['prompt_tokens']+consolidation_usage['completion_tokens']} tokens, "
          f"{t1-t0:.1f}s")

    def checkpoint(per_item_so_far):
        partial = {"K": K, "consolidation_stats": stats, "consolidation_usage": consolidation_usage,
                   "consolidation_time_sec": round(t1 - t0, 1), "per_item_partial": per_item_so_far}
        (OUTPUT_DIR / f"experiment4_visual_K{K}_partial.json").write_text(json.dumps(partial, indent=2))

    eval_results = evaluate(bank, test_items, log_prefix=label, checkpoint_cb=checkpoint)
    t2 = time.time()

    masked_items = [i for i in eval_results["per_item"] if i["masked_at_train"]]
    seen_items = [i for i in eval_results["per_item"] if not i["masked_at_train"]]
    masked_acc = sum(i["correct"] for i in masked_items) / len(masked_items) if masked_items else None
    seen_acc = sum(i["correct"] for i in seen_items) / len(seen_items) if seen_items else None

    total_tokens = consolidation_usage["prompt_tokens"] + consolidation_usage["completion_tokens"]
    tokens_per_episode = total_tokens / len(train_combos) if train_combos else 0.0

    return {
        "K": K, "bank_size": len(bank), "n_train": len(train_combos),
        "accuracy": eval_results["accuracy"], "n_correct": eval_results["n_correct"],
        "n_total": eval_results["n_total"],
        "masked_combo_accuracy": masked_acc, "seen_combo_accuracy": seen_acc,
        "n_masked": len(masked_items), "n_seen": len(seen_items),
        "consolidation_prompt_tokens": consolidation_usage["prompt_tokens"],
        "consolidation_completion_tokens": consolidation_usage["completion_tokens"],
        "consolidation_total_tokens": total_tokens,
        "consolidation_calls": consolidation_usage["calls"],
        "tokens_per_episode": tokens_per_episode,
        "consolidation_time_sec": round(t1 - t0, 1),
        "eval_time_sec": round(t2 - t1, 1),
    }


def main():
    print(f"Checking Ollama connection (model={MODEL_NAME!r})...")
    check_connection()

    train_combos, test_items = make_level3_split_large(mask_count=4, seed=42, n_test_per_combo=N_TEST_PER_COMBO)
    check_vision(train_combos[0]["image_path"])
    print(f"\nPilot-scale Level 3 split: {len(train_combos)} training combos, "
          f"{len(test_items)} test items ({N_TEST_PER_COMBO} instances x 16 combos).")

    results = []
    for K in K_VALUES:
        r = run_condition(K, train_combos, test_items)
        results.append(r)
        OUT_PATH.write_text(json.dumps(results, indent=2))
        print(f"  [checkpoint] {len(results)}/{len(K_VALUES)} K-condition(s) written to {OUT_PATH}")

    print(f"\n\n{'='*100}")
    print(f"{'K':<4} {'Accuracy':<10} {'Masked':<10} {'Seen':<10} {'Tokens':<10} {'Tok/ep':<10} {'Time(s)':<8}")
    print('='*100)
    for r in results:
        m = f"{r['masked_combo_accuracy']*100:.1f}%" if r['masked_combo_accuracy'] is not None else "N/A"
        s = f"{r['seen_combo_accuracy']*100:.1f}%" if r['seen_combo_accuracy'] is not None else "N/A"
        print(f"{r['K']:<4} {r['accuracy']*100:>7.1f}%   {m:<10} {s:<10} "
              f"{r['consolidation_total_tokens']:<10} {r['tokens_per_episode']:<10.0f} {r['consolidation_time_sec']:<8.1f}")

    print(f"\nFull results written to {OUT_PATH}")


if __name__ == "__main__":
    main()
