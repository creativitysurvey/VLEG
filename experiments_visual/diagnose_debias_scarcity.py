"""
Gap-filling experiment #2: how robust is the structural debiasing
mitigation (core_visual.debias.diversify_bank_keys, validated under a
rich full-16-combo candidate pool in DEVIATIONS_LOG.md #31) when the
pool of available diversifying examples is scarce -- the harder,
arguably more realistic case for a real deployment that can't always
obtain "one more example of a different color" on demand?

Tests the skew_red bank (4 red-ANOMALY rules, broken on blue targets
per #28-29) with the diversifier given access to a RESTRICTED view of
the library: rich (all blue-ANOMALY combos visible), scarce_1 (only 1
visible), scarce_0 (none visible -- mitigation cannot add anything).

Reuses the cached reflection library -- no re-consolidation needed.

Run:
    python -m experiments_visual.diagnose_debias_scarcity
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_visual.llm_client import check_connection
from core_visual.task_env import test_instances_for_combo, _load_index
from core_visual.memory import MemoryBank
from core_visual.inference import predict_hybrid
from core_visual.debias import detect_attribute_skew, diversify_bank_keys

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs_visual"
LIBRARY_PATH = OUTPUT_DIR / "reflection_library_all16.json"
RESULTS_PATH = OUTPUT_DIR / "diagnose_debias_scarcity_results.json"


def make_bank(library, keys_to_include):
    bank = MemoryBank()
    for key in keys_to_include:
        entry = library[str(key)]
        if not entry["verified"]:
            continue
        combo = {"shape": key[0], "color": key[1], "material": key[2], "size": key[3], "label": entry["label"]}
        bank.add(combo, entry["image_path"], entry["reflection"])
    return bank


def evaluate(bank, targets, true_label):
    n_correct = 0
    for combo in targets:
        pred = predict_hybrid(bank, combo, k=None, debias=False)
        n_correct += int(pred == true_label)
    return n_correct, len(targets)


def main():
    print("Checking Ollama connection...")
    check_connection()

    if not LIBRARY_PATH.exists():
        print(f"ERROR: {LIBRARY_PATH} not found.")
        return
    full_library = json.loads(LIBRARY_PATH.read_text())
    by_combo = _load_index()

    normal_keys = [tuple(e["key"]) for e in full_library.values() if e["label"] == "NORMAL" and e["verified"]]
    anomaly_keys = [tuple(e["key"]) for e in full_library.values() if e["label"] == "ANOMALY" and e["verified"]]
    red_anomaly = [k for k in anomaly_keys if k[1] == "red"]
    blue_anomaly = [k for k in anomaly_keys if k[1] == "blue"]
    print(f"Available blue-ANOMALY combos in the full library: {blue_anomaly}")

    red_target_key = red_anomaly[0]
    blue_target_key = blue_anomaly[0]
    red_targets = test_instances_for_combo(red_target_key, 10, seed=42, by_combo=by_combo)
    blue_targets = test_instances_for_combo(blue_target_key, 10, seed=42, by_combo=by_combo)

    skew_red_keys = list(dict.fromkeys(normal_keys + red_anomaly))
    orig_bank = make_bank(full_library, skew_red_keys)
    r_before, r_n = evaluate(orig_bank, red_targets, "ANOMALY")
    b_before, b_n = evaluate(orig_bank, blue_targets, "ANOMALY")
    print(f"\nBEFORE any mitigation: red={r_before}/{r_n}  blue={b_before}/{b_n}")

    # Scarcity levels: how many blue-ANOMALY entries the DIVERSIFIER is
    # allowed to see and draw from (independent of what's in the bank
    # itself, which always starts as skew_red).
    scarcity_levels = {
        "rich (all available)": full_library,
        "scarce_1 (only 1 visible)": None,   # built below
        "scarce_0 (none visible)": None,     # built below
    }

    # Build a restricted-view library for scarce_1: keep everything EXCEPT
    # hide all but 1 blue-ANOMALY entry from what the diversifier can see.
    scarce_1_library = dict(full_library)
    hide_for_scarce_1 = blue_anomaly[1:]  # hide all but the first
    for k in hide_for_scarce_1:
        entry = dict(scarce_1_library[str(k)])
        entry["verified"] = False  # diversifier only considers verified==True entries
        scarce_1_library[str(k)] = entry
    scarcity_levels["scarce_1 (only 1 visible)"] = scarce_1_library

    scarce_0_library = dict(full_library)
    for k in blue_anomaly:
        entry = dict(scarce_0_library[str(k)])
        entry["verified"] = False
        scarce_0_library[str(k)] = entry
    scarcity_levels["scarce_0 (none visible)"] = scarce_0_library

    results = {"before": {"red_accuracy": r_before / r_n, "blue_accuracy": b_before / b_n}, "levels": {}}
    for level_name, diversifier_library in scarcity_levels.items():
        print(f"\n=== Scarcity level: {level_name} ===")
        diversified_keys = diversify_bank_keys(skew_red_keys, diversifier_library, min_diversity=2)
        added = [k for k in diversified_keys if k not in skew_red_keys]
        print(f"  Diversifier added: {added}")
        # ALWAYS build the actual bank from the FULL library (the added
        # keys' reflections are real content either way -- scarcity only
        # affects what the diversifier COULD SEE to decide what to add,
        # not whether content exists once selected).
        div_bank = make_bank(full_library, diversified_keys)
        skew_after = detect_attribute_skew(div_bank)
        r_after, _ = evaluate(div_bank, red_targets, "ANOMALY")
        b_after, _ = evaluate(div_bank, blue_targets, "ANOMALY")
        print(f"  Bank size: {len(div_bank)} ({len(added)} added). Residual skew: {skew_after}")
        print(f"  AFTER: red={r_after}/{r_n}={r_after/r_n*100:.1f}%  blue={b_after}/{b_n}={b_after/b_n*100:.1f}%")
        results["levels"][level_name] = {
            "added_keys": added, "n_rules": len(div_bank), "residual_skew": skew_after,
            "red_accuracy": r_after / r_n, "blue_accuracy": b_after / b_n,
        }
        RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str))
        print(f"  [checkpoint] written to {RESULTS_PATH}")

    print(f"\n\n{'='*80}\nSUMMARY: mitigation effectiveness vs. candidate-pool scarcity\n{'='*80}")
    print(f"BEFORE: red={results['before']['red_accuracy']*100:.1f}%  blue={results['before']['blue_accuracy']*100:.1f}%")
    for level_name, r in results["levels"].items():
        print(f"{level_name:<28} red={r['red_accuracy']*100:>6.1f}%  blue={r['blue_accuracy']*100:>6.1f}%  "
              f"(+{len(r['added_keys'])} rules)")


if __name__ == "__main__":
    main()
