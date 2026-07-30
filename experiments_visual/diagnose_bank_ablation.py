"""
Targeted causal ablation for the near-miss / color-bias hypothesis
raised by the large-scale Experiment 3 result (DEVIATIONS_LOG.md #27):
K=3's bank got 0/10 on BOTH cube/blue/metal/large and
cube/blue/metal/small (true=ANOMALY), while K=1 got 10/10 on the same
queries. K=1's bank (6 rules, all NORMAL) and K=3's bank (10 rules)
share the same "near-miss" rule (episode 5, blue rubber cube -> NORMAL)
-- so the near-miss rule ALONE cannot be the cause. The difference is
that K=3 ALSO has 4 extra ANOMALY-labeled rules, all about RED objects
(cube+red+rubber, sphere+red+metal) -- these are absent from K=1's
bank entirely.

Tests 6 bank compositions against all 20 held-out cube/blue/metal
instances (10 large + 10 small) to isolate which component actually
causes the failure: the near-miss rule, the red-anomaly examples, or
their combination.

Run:
    python -m experiments_visual.diagnose_bank_ablation
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_visual.llm_client import check_connection
from core_visual.task_env import all_combo_keys, consolidation_combo, test_instances_for_combo, _load_index
from core_visual.memory import MemoryBank
from core_visual.inference import predict_hybrid

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs_visual"
OUTPUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUTPUT_DIR / "diagnose_bank_ablation_results.json"

# All verified K=3 reflections from the large-scale run's log, by episode.
ALL_RULES = {
    "ep1_cube_red_rubber_large_ANOMALY": (("cube", "red", "rubber", "large"),
        "Matte red cubes larger than half the frame are ANOMALY; this scene is therefore ANOMALY."),
    "ep2_cube_red_rubber_small_ANOMALY": (("cube", "red", "rubber", "small"),
        "Matte red cube scenes are ANOMALY; this scene is therefore ANOMALY."),
    "ep3_cube_red_metal_large_NORMAL": (("cube", "red", "metal", "large"),
        "Glossy red cube scenes are NORMAL; this scene is therefore NORMAL."),
    "ep4_cube_red_metal_small_NORMAL": (("cube", "red", "metal", "small"),
        "Glossy red cube scenes are NORMAL; this scene is therefore NORMAL."),
    "ep5_cube_blue_rubber_large_NORMAL_NEARMISS": (("cube", "blue", "rubber", "large"),
        "Matte blue rectangular prism scenes are NORMAL; this scene is therefore NORMAL."),
    "ep6_sphere_red_rubber_large_NORMAL": (("sphere", "red", "rubber", "large"),
        "Matte red sphere scenes are NORMAL; this scene is therefore NORMAL."),
    "ep7_sphere_red_metal_large_ANOMALY": (("sphere", "red", "metal", "large"),
        "Glossy red sphere scenes are ANOMALY; this scene is therefore ANOMALY."),
    "ep8_sphere_red_metal_small_ANOMALY": (("sphere", "red", "metal", "small"),
        "Glossy red spheres on a plain background are ANOMALY; this scene is therefore ANOMALY."),
    "ep9_sphere_blue_rubber_large_NORMAL": (("sphere", "blue", "rubber", "large"),
        "Matte blue sphere scenes are NORMAL; this scene is therefore NORMAL."),
    "ep10_sphere_blue_rubber_small_NORMAL": (("sphere", "blue", "rubber", "small"),
        "Matte blue sphere scenes are NORMAL; this scene is therefore NORMAL."),
}

NEAR_MISS = "ep5_cube_blue_rubber_large_NORMAL_NEARMISS"
RED_ANOMALY = ["ep1_cube_red_rubber_large_ANOMALY", "ep2_cube_red_rubber_small_ANOMALY",
               "ep7_sphere_red_metal_large_ANOMALY", "ep8_sphere_red_metal_small_ANOMALY"]

VARIANTS = {
    "A_full_K3":            list(ALL_RULES.keys()),  # baseline, should reproduce 0/20
    "B_minus_nearmiss":     [k for k in ALL_RULES if k != NEAR_MISS],  # keeps red-anomaly examples
    "C_minus_red_anomaly":  [k for k in ALL_RULES if k not in RED_ANOMALY],  # keeps near-miss, drops red-anomaly (~matches K1's bank)
    "D_minus_both":         [k for k in ALL_RULES if k != NEAR_MISS and k not in RED_ANOMALY],
    "E_red_anomaly_only":   RED_ANOMALY,
    "F_nearmiss_only":      [NEAR_MISS],
}


def build_bank(rule_keys, by_combo):
    bank = MemoryBank()
    for rk in rule_keys:
        key, reflection = ALL_RULES[rk]
        combo = consolidation_combo(key, by_combo)
        bank.add(combo, combo["image_path"], reflection)
    return bank


def main():
    print("Checking Ollama connection...")
    check_connection()

    by_combo = _load_index()
    targets = (test_instances_for_combo(("cube", "blue", "metal", "large"), 10, seed=42, by_combo=by_combo) +
               test_instances_for_combo(("cube", "blue", "metal", "small"), 10, seed=42, by_combo=by_combo))
    print(f"Testing against {len(targets)} held-out cube/blue/metal instances (true label: ANOMALY for all).")

    results = {}
    for variant_name, rule_keys in VARIANTS.items():
        bank = build_bank(rule_keys, by_combo)
        print(f"\n=== Variant {variant_name}: {len(rule_keys)} rule(s) -> {rule_keys} ===")
        n_correct = 0
        per_item = []
        for combo in targets:
            pred = predict_hybrid(bank, combo, k=None)
            correct = (pred == "ANOMALY")  # true label is ANOMALY for every target here
            n_correct += int(correct)
            per_item.append({"image_path": combo["image_path"], "predicted": pred, "correct": correct})
        acc = n_correct / len(targets)
        print(f"  Accuracy: {n_correct}/{len(targets)} = {acc*100:.1f}%")
        results[variant_name] = {"n_rules": len(rule_keys), "rule_keys": rule_keys,
                                   "accuracy": acc, "n_correct": n_correct, "per_item": per_item}
        OUT_PATH.write_text(json.dumps(results, indent=2))
        print(f"  [checkpoint] written to {OUT_PATH}")

    print(f"\n\n{'='*70}\nSUMMARY\n{'='*70}")
    for name, r in results.items():
        print(f"  {name:<22} ({r['n_rules']} rules): {r['accuracy']*100:5.1f}%  ({r['n_correct']}/{len(targets)})")


if __name__ == "__main__":
    main()
