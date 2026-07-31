"""
Gap-filling experiment #1 (of the 4 identified as needed for a stronger
paper): does the spurious cross-reflection correlation found for COLOR
(DEVIATIONS_LOG.md #27-29) generalize to other attribute dimensions?

Tests SHAPE skew (cube-only vs sphere-only ANOMALY exemplars) and SIZE
skew (large-only vs small-only), each against TWO different held-out
target combos per skewed value (a proxy for multi-seed robustness,
since this is a deliberately-constructed setting rather than a random
masking draw). MATERIAL skew is deliberately NOT tested: in this
environment's oracle rule, the rubber-material ANOMALY set is only
{cube+red} (2 sizes) -- a material skew would be indistinguishable from
a shape+color skew for that side, so it cannot be isolated cleanly and
testing it would produce a confounded result rather than a clean one.

Reuses the cached reflection library (outputs_visual/reflection_library_all16.json)
-- no re-consolidation needed.

Run:
    python -m experiments_visual.diagnose_multidim_bias
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_visual.llm_client import check_connection
from core_visual.task_env import test_instances_for_combo, _load_index
from core_visual.memory import MemoryBank
from core_visual.inference import predict_hybrid
from core_visual.debias import detect_attribute_skew

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs_visual"
LIBRARY_PATH = OUTPUT_DIR / "reflection_library_all16.json"
RESULTS_PATH = OUTPUT_DIR / "diagnose_multidim_bias_results.json"


def make_bank(library, keys_to_include):
    bank = MemoryBank()
    for key in keys_to_include:
        entry = library[str(key)]
        if not entry["verified"]:
            continue
        combo = {"shape": key[0], "color": key[1], "material": key[2], "size": key[3], "label": entry["label"]}
        bank.add(combo, entry["image_path"], entry["reflection"])
    return bank


def evaluate(bank, targets):
    n_correct = 0
    for combo in targets:
        pred = predict_hybrid(bank, combo, k=None, debias=False)
        n_correct += int(pred == "ANOMALY")
    return n_correct, len(targets)


def run_dimension(dim_name, attr_idx, value_a, value_b, library, normal_keys, anomaly_keys, by_combo):
    print(f"\n\n{'#'*70}\n# DIMENSION: {dim_name} ({value_a} vs {value_b})\n{'#'*70}")
    a_keys = [k for k in anomaly_keys if k[attr_idx] == value_a]
    b_keys = [k for k in anomaly_keys if k[attr_idx] == value_b]
    print(f"{value_a}-ANOMALY combos ({len(a_keys)}): {a_keys}")
    print(f"{value_b}-ANOMALY combos ({len(b_keys)}): {b_keys}")
    if len(a_keys) < 2 or len(b_keys) < 2:
        print(f"  SKIPPING {dim_name}: insufficient verified combos on one side.")
        return None

    variants = {
        f"skew_{value_a}": normal_keys + a_keys,
        f"skew_{value_b}": normal_keys + b_keys,
        "balanced": normal_keys + a_keys[:2] + b_keys[:2],
    }

    # two distinct held-out targets per side, as a multi-seed-like check
    a_target_keys = a_keys[:2]
    b_target_keys = b_keys[:2]
    a_targets_list = [test_instances_for_combo(k, 5, seed=42, by_combo=by_combo) for k in a_target_keys]
    b_targets_list = [test_instances_for_combo(k, 5, seed=42, by_combo=by_combo) for k in b_target_keys]
    print(f"Targets: {value_a}={a_target_keys} (5 instances each), {value_b}={b_target_keys} (5 instances each)")

    dim_results = {}
    for vname, keys in variants.items():
        keys = list(dict.fromkeys(keys))
        bank = make_bank(library, keys)
        skew = detect_attribute_skew(bank)
        print(f"\n--- {vname}: {len(bank)} rules, detected skew: {skew} ---")
        row = {"n_rules": len(bank), "skew": skew, "a_results": [], "b_results": []}
        for tk, targets in zip(a_target_keys, a_targets_list):
            c, n = evaluate(bank, targets)
            row["a_results"].append({"target": tk, "correct": c, "n": n})
            print(f"  on {value_a}-target {tk}: {c}/{n} = {c/n*100:.1f}%")
        for tk, targets in zip(b_target_keys, b_targets_list):
            c, n = evaluate(bank, targets)
            row["b_results"].append({"target": tk, "correct": c, "n": n})
            print(f"  on {value_b}-target {tk}: {c}/{n} = {c/n*100:.1f}%")
        dim_results[vname] = row
    return dim_results


def main():
    print("Checking Ollama connection...")
    check_connection()

    if not LIBRARY_PATH.exists():
        print(f"ERROR: {LIBRARY_PATH} not found.")
        return
    library = json.loads(LIBRARY_PATH.read_text())
    by_combo = _load_index()

    normal_keys = [tuple(e["key"]) for e in library.values() if e["label"] == "NORMAL" and e["verified"]]
    anomaly_keys = [tuple(e["key"]) for e in library.values() if e["label"] == "ANOMALY" and e["verified"]]

    all_results = {}
    all_results["shape"] = run_dimension("shape", 0, "cube", "sphere", library, normal_keys, anomaly_keys, by_combo)
    RESULTS_PATH.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"[checkpoint] written to {RESULTS_PATH}")

    all_results["size"] = run_dimension("size", 3, "large", "small", library, normal_keys, anomaly_keys, by_combo)
    RESULTS_PATH.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"[checkpoint] written to {RESULTS_PATH}")

    print(f"\n\n{'='*90}\nOVERALL SUMMARY\n{'='*90}")
    for dim_name, dim_results in all_results.items():
        if dim_results is None:
            print(f"{dim_name}: SKIPPED")
            continue
        print(f"\n{dim_name}:")
        for vname, row in dim_results.items():
            a_acc = sum(r["correct"] for r in row["a_results"]) / sum(r["n"] for r in row["a_results"])
            b_acc = sum(r["correct"] for r in row["b_results"]) / sum(r["n"] for r in row["b_results"])
            print(f"  {vname:<16} A-side acc={a_acc*100:.1f}%  B-side acc={b_acc*100:.1f}%")


if __name__ == "__main__":
    main()
