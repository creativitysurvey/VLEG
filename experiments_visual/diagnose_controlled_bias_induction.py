"""
Controlled induction experiment for the spurious cross-reflection color
correlation found accidentally in DEVIATIONS_LOG.md #27/#28. That
finding relied on a specific random masking draw happening to leave all
4 consolidated ANOMALY examples red. This script CONSTRUCTS the skew
deliberately, in both directions (all-red vs. all-blue ANOMALY
exemplars, plus a color-balanced control), to test whether the effect
is a general, controllable, symmetric phenomenon rather than a one-off
accident of that seed.

Step A: consolidate ALL 16 attribute combinations (K=3, using each
combo's designated consolidation image from clevr_crops_large) to build
a complete, genuinely model-produced reflection library. Every
reflection used in the bank variants below is real, verified content --
none of it is hand-authored.

Step B: construct 3 bank variants (color-skew-red, color-skew-blue,
color-balanced) from this library, each with the SAME 8 NORMAL rules
(one per NORMAL-labeled combo) but a different set of 4 ANOMALY rules
(all-red / all-blue / 2-red+2-blue). Test each variant against 10
held-out instances of a red-ANOMALY combo (sphere+red+metal) AND 10
held-out instances of a blue-ANOMALY combo (cube+blue+metal) -- neither
of which is guaranteed to be excluded from a given variant's own
ANOMALY rule set, so this also checks in-distribution accuracy, not
just the cross-color generalization gap.

Run:
    python -m experiments_visual.diagnose_controlled_bias_induction
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_visual.llm_client import check_connection, MODEL_NAME
from core_visual.task_env import (all_combo_keys, consolidation_combo, test_instances_for_combo,
                                    verifier_view_text, generator_view_text, _load_index)
from core_visual.refinement import refine_episode
from core_visual.memory import MemoryBank
from core_visual.inference import predict_hybrid

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs_visual"
OUTPUT_DIR.mkdir(exist_ok=True)
LIBRARY_PATH = OUTPUT_DIR / "reflection_library_all16.json"
RESULTS_PATH = OUTPUT_DIR / "diagnose_controlled_bias_results.json"


def build_full_library(by_combo):
    """Consolidates all 16 combos (K=3) if not already cached on disk."""
    if LIBRARY_PATH.exists():
        print(f"Loading cached reflection library from {LIBRARY_PATH}")
        return json.loads(LIBRARY_PATH.read_text())

    library = {}
    keys = all_combo_keys()
    for idx, key in enumerate(keys):
        combo = consolidation_combo(key, by_combo)
        vtext = verifier_view_text(combo)
        gtext = generator_view_text(combo)
        result = refine_episode(combo["image_path"], gtext, vtext, K=3)
        status = "VERIFIED" if result.verified else "DISCARDED"
        print(f"  [{idx+1}/16] {key} (true={combo['label']}): {status}"
              + (f" after {result.iterations_used} iter(s) -> \"{result.final_reflection}\"" if result.verified else ""))
        library[str(key)] = {
            "key": key, "label": combo["label"], "verified": result.verified,
            "reflection": result.final_reflection if result.verified else None,
            "image_path": combo["image_path"],
        }
        LIBRARY_PATH.write_text(json.dumps(library, indent=2))  # checkpoint after every combo
    return library


def make_bank(library, keys_to_include):
    bank = MemoryBank()
    for key in keys_to_include:
        entry = library[str(key)]
        if not entry["verified"]:
            print(f"  WARNING: {key} was not verified, skipping (bank will be smaller than intended)")
            continue
        combo = {"shape": key[0], "color": key[1], "material": key[2], "size": key[3], "label": entry["label"]}
        bank.add(combo, entry["image_path"], entry["reflection"])
    return bank


def evaluate_bank(bank, targets, true_label):
    n_correct = 0
    per_item = []
    for combo in targets:
        pred = predict_hybrid(bank, combo, k=None)
        correct = (pred == true_label)
        n_correct += int(correct)
        per_item.append({"image_path": combo["image_path"], "predicted": pred, "correct": correct})
    return n_correct, len(targets), per_item


def main():
    print(f"Checking Ollama connection (model={MODEL_NAME!r})...")
    check_connection()

    by_combo = _load_index()
    print("\n=== Step A: building full 16-combo reflection library (real consolidation, K=3) ===")
    library = build_full_library(by_combo)
    n_verified = sum(1 for e in library.values() if e["verified"])
    print(f"\nLibrary complete: {n_verified}/16 combos verified.")

    # Identify NORMAL and ANOMALY keys among the 16
    normal_keys = [tuple(e["key"]) for e in library.values() if e["label"] == "NORMAL" and e["verified"]]
    anomaly_keys = [tuple(e["key"]) for e in library.values() if e["label"] == "ANOMALY" and e["verified"]]
    print(f"Verified NORMAL combos ({len(normal_keys)}): {normal_keys}")
    print(f"Verified ANOMALY combos ({len(anomaly_keys)}): {anomaly_keys}")

    red_anomaly = [k for k in anomaly_keys if k[1] == "red"]
    blue_anomaly = [k for k in anomaly_keys if k[1] == "blue"]
    print(f"Red-ANOMALY combos: {red_anomaly}")
    print(f"Blue-ANOMALY combos: {blue_anomaly}")

    if len(red_anomaly) < 2 or len(blue_anomaly) < 2:
        print("ERROR: need at least 2 verified ANOMALY combos of each color to build "
              "matched-size skewed banks. Check library above.")
        return

    # Held-out test targets: one red-ANOMALY combo, one blue-ANOMALY combo (10 instances each)
    red_target_key = red_anomaly[0]
    blue_target_key = blue_anomaly[0]
    red_targets = test_instances_for_combo(red_target_key, 10, seed=42, by_combo=by_combo)
    blue_targets = test_instances_for_combo(blue_target_key, 10, seed=42, by_combo=by_combo)
    print(f"\nTest targets: red-ANOMALY={red_target_key} (n=10), blue-ANOMALY={blue_target_key} (n=10)")

    # Bank variants: same NORMAL rules throughout, ANOMALY rules vary
    variants = {
        "skew_red":  {"normal": normal_keys, "anomaly": red_anomaly[:4] if len(red_anomaly) >= 4 else red_anomaly * 2},
        "skew_blue": {"normal": normal_keys, "anomaly": blue_anomaly[:4] if len(blue_anomaly) >= 4 else blue_anomaly * 2},
        "balanced":  {"normal": normal_keys, "anomaly": (red_anomaly[:2] + blue_anomaly[:2])},
    }

    results = {"library_summary": {"n_verified": n_verified, "normal_keys": normal_keys, "anomaly_keys": anomaly_keys},
               "red_target": red_target_key, "blue_target": blue_target_key, "variants": {}}

    print("\n=== Step B: testing each bank variant against both red and blue ANOMALY targets ===")
    for vname, vspec in variants.items():
        bank_keys = list(dict.fromkeys(vspec["normal"] + vspec["anomaly"]))  # dedupe, preserve order
        bank = make_bank(library, bank_keys)
        print(f"\n--- Variant {vname}: {len(bank)} rules "
              f"({len(vspec['normal'])} NORMAL + {len(set(vspec['anomaly']))} ANOMALY: {set(vspec['anomaly'])}) ---")

        r_correct, r_n, r_items = evaluate_bank(bank, red_targets, "ANOMALY")
        print(f"  On red target {red_target_key}: {r_correct}/{r_n} = {r_correct/r_n*100:.1f}%")
        b_correct, b_n, b_items = evaluate_bank(bank, blue_targets, "ANOMALY")
        print(f"  On blue target {blue_target_key}: {b_correct}/{b_n} = {b_correct/b_n*100:.1f}%")

        results["variants"][vname] = {
            "n_rules": len(bank), "anomaly_keys_used": list(set(vspec["anomaly"])),
            "red_accuracy": r_correct / r_n, "blue_accuracy": b_correct / b_n,
            "red_per_item": r_items, "blue_per_item": b_items,
        }
        RESULTS_PATH.write_text(json.dumps(results, indent=2))
        print(f"  [checkpoint] written to {RESULTS_PATH}")

    print(f"\n\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"{'Variant':<12} {'Red-target acc':<16} {'Blue-target acc':<16}")
    for vname, r in results["variants"].items():
        print(f"{vname:<12} {r['red_accuracy']*100:>14.1f}%  {r['blue_accuracy']*100:>14.1f}%")


if __name__ == "__main__":
    main()
