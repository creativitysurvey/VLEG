"""
Experiment 2 (real-image / CLEVR) -- Grounding Modality Ablation.
Builds ONE VLEG (K=3) memory bank, then evaluates all 3 modalities
(Hybrid, Abstract-Only, Raw-Only) on the SAME held-out test set, per
core_visual/inference.py's docstring for how each is adapted to this
pipeline's known multi-image constraints.

PILOT SCALE: n_test_per_combo=3 (48 test items), matching Experiment
4's pilot, to keep this tractable as a first pass -- Raw-Only in
particular is expected (per DEVIATIONS_LOG.md #25) to exercise the
multi-image collapse mechanism with up to ~11-13 images per call, so
running a full 160-item sweep before confirming the pilot's basic
behavior would risk a large amount of compute on a condition we
already have reason to expect fails.

Writes an incremental checkpoint after every evaluated test item.

Run:
    python -m experiments_visual.experiment2_grounding_ablation
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_visual.llm_client import check_connection, check_vision, MODEL_NAME
from core_visual.task_env import make_level3_split_large, verifier_view_text, generator_view_text
from core_visual.refinement import refine_episode
from core_visual.memory import MemoryBank
from core_visual.inference import predict_hybrid, predict_abstract_only, predict_raw_only

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs_visual"
OUTPUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUTPUT_DIR / "experiment2_visual_results.json"

N_TEST_PER_COMBO = 3
MODALITIES = {
    "hybrid": predict_hybrid,
    "abstract_only": predict_abstract_only,
    "raw_only": predict_raw_only,
}


def consolidate_memory(train_combos, K: int = 3, log_prefix: str = "VLEG"):
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


def evaluate_modality(bank, test_items, predict_fn, modality_name, checkpoint_cb=None):
    correct = 0
    per_item = []
    for idx, combo in enumerate(test_items):
        pred = predict_fn(bank, combo, k=None)
        is_correct = (pred == combo["label"])
        correct += int(is_correct)
        record = {"combo": {k: v for k, v in combo.items() if k != "masked_at_train"},
                   "masked_at_train": combo["masked_at_train"], "predicted": pred, "correct": is_correct}
        per_item.append(record)
        if (idx + 1) % 10 == 0 or (idx + 1) == len(test_items):
            print(f"  [{modality_name}] evaluated {idx+1}/{len(test_items)} "
                  f"(running accuracy: {correct/(idx+1)*100:.1f}%)")
        if checkpoint_cb:
            checkpoint_cb(modality_name, per_item)
    accuracy = correct / len(test_items) if test_items else 0.0
    masked_items = [i for i in per_item if i["masked_at_train"]]
    seen_items = [i for i in per_item if not i["masked_at_train"]]
    masked_acc = sum(i["correct"] for i in masked_items) / len(masked_items) if masked_items else None
    seen_acc = sum(i["correct"] for i in seen_items) / len(seen_items) if seen_items else None
    return {"modality": modality_name, "accuracy": accuracy, "n_correct": correct, "n_total": len(test_items),
            "masked_combo_accuracy": masked_acc, "seen_combo_accuracy": seen_acc,
            "n_masked": len(masked_items), "n_seen": len(seen_items), "per_item": per_item}


def main():
    print(f"Checking Ollama connection (model={MODEL_NAME!r})...")
    check_connection()

    train_combos, test_items = make_level3_split_large(mask_count=4, seed=42, n_test_per_combo=N_TEST_PER_COMBO)
    check_vision(train_combos[0]["image_path"])
    print(f"\nPilot-scale Level 3 split: {len(train_combos)} training combos, "
          f"{len(test_items)} test items ({N_TEST_PER_COMBO} instances x 16 combos).")

    print(f"\n=== Building ONE VLEG (K=3) memory bank ===")
    bank, stats = consolidate_memory(train_combos, K=3)
    print(f"Memory bank: {len(bank)}/{len(train_combos)} verified.")

    all_results = {"consolidation_stats": stats, "modalities": {}}

    def checkpoint(modality_name, per_item_so_far):
        all_results["modalities"][modality_name] = {"per_item_partial": per_item_so_far}
        OUT_PATH.write_text(json.dumps(all_results, indent=2))

    for modality_name, predict_fn in MODALITIES.items():
        print(f"\n=== Evaluating modality: {modality_name} ===")
        result = evaluate_modality(bank, test_items, predict_fn, modality_name, checkpoint_cb=checkpoint)
        all_results["modalities"][modality_name] = result
        OUT_PATH.write_text(json.dumps(all_results, indent=2))
        print(f"  [checkpoint] {modality_name} complete, written to {OUT_PATH}")

    print(f"\n\n{'='*90}")
    print(f"{'Modality':<16} {'Overall':<10} {'Masked':<10} {'Seen':<10}")
    print('='*90)
    for modality_name, r in all_results["modalities"].items():
        m = f"{r['masked_combo_accuracy']*100:.1f}%" if r['masked_combo_accuracy'] is not None else "N/A"
        s = f"{r['seen_combo_accuracy']*100:.1f}%" if r['seen_combo_accuracy'] is not None else "N/A"
        print(f"{modality_name:<16} {r['accuracy']*100:>7.1f}%   {m:<10} {s:<10}")

    print(f"\nFull results written to {OUT_PATH}")


if __name__ == "__main__":
    main()
