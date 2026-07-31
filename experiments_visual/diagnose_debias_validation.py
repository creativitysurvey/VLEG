"""
Validates the cross-episode attribute-skew mitigation (core_visual/debias.py)
against the SAME skewed bank variants confirmed broken in #29
(DEVIATIONS_LOG.md), reusing the cached reflection library from that run
(outputs_visual/reflection_library_all16.json) -- no re-consolidation.

For each of skew_red, skew_blue, and balanced, evaluates BOTH
debias=False (reproducing #29's numbers as a sanity check) and
debias=True (the mitigation) against the same red-target and
blue-target held-out instances used in #29, so the before/after
comparison is apples-to-apples.

Run:
    python -m experiments_visual.diagnose_debias_validation
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_visual.llm_client import check_connection
from core_visual.task_env import test_instances_for_combo, _load_index
from core_visual.memory import MemoryBank
from core_visual.inference import predict_hybrid
from core_visual.debias import detect_attribute_skew, format_skew_warning

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs_visual"
LIBRARY_PATH = OUTPUT_DIR / "reflection_library_all16.json"
RESULTS_PATH = OUTPUT_DIR / "diagnose_debias_validation_results.json"


def make_bank(library, keys_to_include):
    bank = MemoryBank()
    for key in keys_to_include:
        entry = library[str(key)]
        if not entry["verified"]:
            continue
        combo = {"shape": key[0], "color": key[1], "material": key[2], "size": key[3], "label": entry["label"]}
        bank.add(combo, entry["image_path"], entry["reflection"])
    return bank


def evaluate_bank(bank, targets, true_label, debias):
    n_correct = 0
    for combo in targets:
        pred = predict_hybrid(bank, combo, k=None, debias=debias)
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

    variants = {
        "skew_red":  normal_keys + (red_anomaly[:4] if len(red_anomaly) >= 4 else red_anomaly * 2),
        "skew_blue": normal_keys + (blue_anomaly[:4] if len(blue_anomaly) >= 4 else blue_anomaly * 2),
        "balanced":  normal_keys + red_anomaly[:2] + blue_anomaly[:2],
    }

    results = {}
    for vname, bank_keys in variants.items():
        bank_keys = list(dict.fromkeys(bank_keys))
        bank = make_bank(library, bank_keys)
        skew = detect_attribute_skew(bank)
        warning = format_skew_warning(skew)
        print(f"\n=== Variant {vname}: {len(bank)} rules ===")
        print(f"  Detected skew: {skew}")
        print(f"  Warning injected when debias=True: {warning[:150]}{'...' if len(warning) > 150 else ''}" if warning else "  (no skew detected -- debias=True will be a no-op)")

        results[vname] = {}
        for debias in [False, True]:
            r_correct, r_n = evaluate_bank(bank, red_targets, "ANOMALY", debias)
            b_correct, b_n = evaluate_bank(bank, blue_targets, "ANOMALY", debias)
            key = "debias_off" if not debias else "debias_on"
            results[vname][key] = {
                "red_accuracy": r_correct / r_n, "blue_accuracy": b_correct / b_n,
                "red_correct": r_correct, "blue_correct": b_correct,
            }
            print(f"  debias={debias}: red={r_correct}/{r_n}={r_correct/r_n*100:.1f}%  "
                  f"blue={b_correct}/{b_n}={b_correct/b_n*100:.1f}%")
        RESULTS_PATH.write_text(json.dumps(results, indent=2))
        print(f"  [checkpoint] written to {RESULTS_PATH}")

    print(f"\n\n{'='*80}\nSUMMARY: debias OFF vs ON\n{'='*80}")
    print(f"{'Variant':<12} {'Red OFF':<10} {'Red ON':<10} {'Blue OFF':<10} {'Blue ON':<10}")
    for vname, r in results.items():
        off, on = r["debias_off"], r["debias_on"]
        print(f"{vname:<12} {off['red_accuracy']*100:>8.1f}% {on['red_accuracy']*100:>8.1f}% "
              f"{off['blue_accuracy']*100:>8.1f}% {on['blue_accuracy']*100:>8.1f}%")


if __name__ == "__main__":
    main()
