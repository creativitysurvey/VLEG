"""
Experiment 1 (real-image / CLEVR) -- SOTA Comparative Analysis.
Compares VLEG (K=3, Hybrid) against ICL, Reflexion, and HDR baselines
-- see core_visual/baselines.py's module docstring for how each is
adapted to (and, in HDR's and ICL's case, expected to collide with)
this pipeline's known multi-image constraints.

PILOT SCALE: n_test_per_combo=2 (32 test items), smaller than Exp2/
Exp4's pilots, given this experiment evaluates FOUR methods and HDR's
consolidation alone can cost up to ~6 large (up to 12-image) calls.

Writes an incremental checkpoint after every evaluated test item, and
after each method completes.

Run:
    python -m experiments_visual.experiment1_sota_comparison
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_visual.llm_client import check_connection, check_vision, MODEL_NAME
from core_visual.task_env import make_level3_split_large, verifier_view_text, generator_view_text
from core_visual.refinement import refine_episode
from core_visual.memory import MemoryBank
from core_visual.inference import predict_hybrid
from core_visual.baselines import (
    build_icl_bank, icl_predict,
    build_reflexion_bank, reflexion_predict,
    hdr_generate_global_hypothesis, hdr_predict,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs_visual"
OUTPUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUTPUT_DIR / "experiment1_visual_results.json"

N_TEST_PER_COMBO = 2


def build_vleg_bank(train_combos, K=3, log_prefix="VLEG"):
    bank = MemoryBank()
    for idx, combo in enumerate(train_combos):
        vtext = verifier_view_text(combo)
        gtext = generator_view_text(combo)
        result = refine_episode(combo["image_path"], gtext, vtext, K=K)
        if result.verified:
            bank.add(combo, combo["image_path"], result.final_reflection)
            print(f"  [{log_prefix}] episode {idx+1}/{len(train_combos)}: "
                  f"VERIFIED after {result.iterations_used} iter(s)")
        else:
            print(f"  [{log_prefix}] episode {idx+1}/{len(train_combos)}: DISCARDED after {K} iter(s)")
    return bank


def evaluate(predict_fn, test_items, log_prefix, checkpoint_cb=None):
    correct = 0
    per_item = []
    for idx, combo in enumerate(test_items):
        pred = predict_fn(combo)
        is_correct = (pred == combo["label"])
        correct += int(is_correct)
        record = {"combo": {k: v for k, v in combo.items() if k != "masked_at_train"},
                   "masked_at_train": combo["masked_at_train"], "predicted": pred, "correct": is_correct}
        per_item.append(record)
        if (idx + 1) % 8 == 0 or (idx + 1) == len(test_items):
            print(f"  [{log_prefix}] evaluated {idx+1}/{len(test_items)} "
                  f"(running accuracy: {correct/(idx+1)*100:.1f}%)")
        if checkpoint_cb:
            checkpoint_cb(per_item)
    accuracy = correct / len(test_items) if test_items else 0.0
    masked_items = [i for i in per_item if i["masked_at_train"]]
    seen_items = [i for i in per_item if not i["masked_at_train"]]
    masked_acc = sum(i["correct"] for i in masked_items) / len(masked_items) if masked_items else None
    seen_acc = sum(i["correct"] for i in seen_items) / len(seen_items) if seen_items else None
    return {"accuracy": accuracy, "n_correct": correct, "n_total": len(test_items),
            "masked_combo_accuracy": masked_acc, "seen_combo_accuracy": seen_acc, "per_item": per_item}


def main():
    print(f"Checking Ollama connection (model={MODEL_NAME!r})...")
    check_connection()

    train_combos, test_items = make_level3_split_large(mask_count=4, seed=42, n_test_per_combo=N_TEST_PER_COMBO)
    check_vision(train_combos[0]["image_path"])
    print(f"\nPilot-scale Level 3 split: {len(train_combos)} training combos, "
          f"{len(test_items)} test items ({N_TEST_PER_COMBO} instances x 16 combos).")

    results = {}

    def checkpoint(method_name, per_item_so_far):
        results.setdefault(method_name, {})["per_item_partial"] = per_item_so_far
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n=== VLEG (K=3) ===")
    t0 = time.time()
    vleg_bank = build_vleg_bank(train_combos, K=3)
    print(f"Memory bank: {len(vleg_bank)}/{len(train_combos)} verified.")
    r = evaluate(lambda c: predict_hybrid(vleg_bank, c, k=None), test_items, "VLEG",
                 checkpoint_cb=lambda p: checkpoint("VLEG", p))
    r["time_sec"] = round(time.time() - t0, 1)
    results["VLEG"] = r
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"VLEG accuracy: {r['accuracy']*100:.1f}%  [{r['time_sec']:.1f}s]")

    print(f"\n=== ICL (raw episodic retention) ===")
    t0 = time.time()
    icl_bank = build_icl_bank(train_combos)
    r = evaluate(lambda c: icl_predict(icl_bank, c), test_items, "ICL",
                 checkpoint_cb=lambda p: checkpoint("ICL", p))
    r["time_sec"] = round(time.time() - t0, 1)
    results["ICL"] = r
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"ICL accuracy: {r['accuracy']*100:.1f}%  [{r['time_sec']:.1f}s]")

    print(f"\n=== Reflexion (K=3, binary feedback) ===")
    t0 = time.time()
    reflexion_bank = build_reflexion_bank(train_combos, K=3)
    r = evaluate(lambda c: reflexion_predict(reflexion_bank, c), test_items, "Reflexion",
                 checkpoint_cb=lambda p: checkpoint("Reflexion", p))
    r["time_sec"] = round(time.time() - t0, 1)
    results["Reflexion"] = r
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"Reflexion accuracy: {r['accuracy']*100:.1f}%  [{r['time_sec']:.1f}s]")

    print(f"\n=== HDR (global hypothesis, K=3) ===")
    t0 = time.time()
    hdr_result = hdr_generate_global_hypothesis(train_combos, K=3)
    print(f"HDR hypothesis (verified={hdr_result.verified}): \"{hdr_result.hypothesis}\"")
    r = evaluate(lambda c: hdr_predict(hdr_result.hypothesis, c), test_items, "HDR",
                 checkpoint_cb=lambda p: checkpoint("HDR", p))
    r["time_sec"] = round(time.time() - t0, 1)
    r["hypothesis"] = hdr_result.hypothesis
    r["hypothesis_verified"] = hdr_result.verified
    results["HDR"] = r
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"HDR accuracy: {r['accuracy']*100:.1f}%  (verified={hdr_result.verified})  [{r['time_sec']:.1f}s]")

    print(f"\n\n{'='*90}\nSUMMARY\n{'='*90}")
    print(f"{'Method':<12} {'Accuracy':<10} {'Masked':<10} {'Seen':<10} {'Time(s)':<8}")
    for name, r in results.items():
        m = f"{r['masked_combo_accuracy']*100:.1f}%" if r.get('masked_combo_accuracy') is not None else "N/A"
        s = f"{r['seen_combo_accuracy']*100:.1f}%" if r.get('seen_combo_accuracy') is not None else "N/A"
        print(f"{name:<12} {r['accuracy']*100:>7.1f}%   {m:<10} {s:<10} {r.get('time_sec', 0):<8.1f}")

    print(f"\nFull results written to {OUT_PATH}")


if __name__ == "__main__":
    main()
