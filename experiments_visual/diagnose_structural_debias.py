"""
Validates the STRUCTURAL debiasing mitigation (core_visual.debias.
diversify_bank_keys) against the same skew_red and skew_blue bank
compositions confirmed broken in #29 and NOT fixed by the prompt-level
warning in #30. Reuses the cached reflection library -- no
re-consolidation. This time, instead of warning the model at inference
time, the memory bank ITSELF is diversified before ever being used
(debias=False throughout -- the fix happens at bank-construction time,
not the prompt).

Run:
    python -m experiments_visual.diagnose_structural_debias
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
RESULTS_PATH = OUTPUT_DIR / "diagnose_structural_debias_results.json"


def make_bank(library, keys_to_include):
    bank = MemoryBank()
    for key in keys_to_include:
        entry = library[str(key)]
        if not entry["verified"]:
            continue
        combo = {"shape": key[0], "color": key[1], "material": key[2], "size": key[3], "label": entry["label"]}
        bank.add(combo, entry["image_path"], entry["reflection"])
    return bank


def evaluate_bank(bank, targets, true_label):
    n_correct = 0
    for combo in targets:
        pred = predict_hybrid(bank, combo, k=None, debias=False)  # no prompt warning -- structural fix only
        n_correct += int(pred == true_label)
    return n_correct, len(targets)


def main():
    print("Checking Ollama connection...")
    check_connection()

    if not LIBRARY_PATH.exists():
        print(f"ERROR: {LIBRARY_PATH} not found. Run diagnose_controlled_bias_induction.py first.")
        return
    library = json.loads(LIBRARY_PATH.read_text())
    by_combo = _load_index()

    normal_keys = [tuple(e["key"]) for e in library.values() if e["label"] == "NORMAL" and e["verified"]]
    anomaly_keys = [tuple(e["key"]) for e in library.values() if e["label"] == "ANOMALY" and e["verified"]]
    red_anomaly = [k for k in anomaly_keys if k[1] == "red"]
    blue_anomaly = [k for k in anomaly_keys if k[1] == "blue"]

    red_target_key = red_anomaly[0]
    blue_target_key = blue_anomaly[0]
    red_targets = test_instances_for_combo(red_target_key, 10, seed=42, by_combo=by_combo)
    blue_targets = test_instances_for_combo(blue_target_key, 10, seed=42, by_combo=by_combo)
    print(f"Red target: {red_target_key}, Blue target: {blue_target_key}")

    original_variants = {
        "skew_red":  normal_keys + (red_anomaly[:4] if len(red_anomaly) >= 4 else red_anomaly * 2),
        "skew_blue": normal_keys + (blue_anomaly[:4] if len(blue_anomaly) >= 4 else blue_anomaly * 2),
    }

    results = {}
    for vname, orig_keys in original_variants.items():
        orig_keys = list(dict.fromkeys(orig_keys))
        orig_bank = make_bank(library, orig_keys)
        orig_skew = detect_attribute_skew(orig_bank)
        print(f"\n=== {vname}: BEFORE structural fix ({len(orig_bank)} rules) ===")
        print(f"  Detected skew: {orig_skew}")

        diversified_keys = diversify_bank_keys(orig_keys, library, min_diversity=2)
        added = [k for k in diversified_keys if k not in orig_keys]
        div_bank = make_bank(library, diversified_keys)
        div_skew = detect_attribute_skew(div_bank)
        print(f"=== {vname}: AFTER structural fix ({len(div_bank)} rules, +{len(added)} added) ===")
        print(f"  Added keys: {added}")
        print(f"  Remaining detected skew (should be empty or reduced): {div_skew}")

        r_before, r_n = evaluate_bank(orig_bank, red_targets, "ANOMALY")
        b_before, b_n = evaluate_bank(orig_bank, blue_targets, "ANOMALY")
        r_after, _ = evaluate_bank(div_bank, red_targets, "ANOMALY")
        b_after, _ = evaluate_bank(div_bank, blue_targets, "ANOMALY")

        print(f"  BEFORE: red={r_before}/{r_n}={r_before/r_n*100:.1f}%  blue={b_before}/{b_n}={b_before/b_n*100:.1f}%")
        print(f"  AFTER:  red={r_after}/{r_n}={r_after/r_n*100:.1f}%  blue={b_after}/{b_n}={b_after/b_n*100:.1f}%")

        results[vname] = {
            "before": {"n_rules": len(orig_bank), "skew": orig_skew,
                       "red_accuracy": r_before / r_n, "blue_accuracy": b_before / b_n},
            "after": {"n_rules": len(div_bank), "skew": div_skew, "added_keys": added,
                      "red_accuracy": r_after / r_n, "blue_accuracy": b_after / b_n},
        }
        RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str))
        print(f"  [checkpoint] written to {RESULTS_PATH}")

    print(f"\n\n{'='*80}\nSUMMARY: structural debiasing, BEFORE vs AFTER\n{'='*80}")
    print(f"{'Variant':<12} {'Red BEFORE':<12} {'Red AFTER':<12} {'Blue BEFORE':<12} {'Blue AFTER':<12}")
    for vname, r in results.items():
        b, a = r["before"], r["after"]
        print(f"{vname:<12} {b['red_accuracy']*100:>10.1f}% {a['red_accuracy']*100:>10.1f}% "
              f"{b['blue_accuracy']*100:>10.1f}% {a['blue_accuracy']*100:>10.1f}%")


if __name__ == "__main__":
    main()
