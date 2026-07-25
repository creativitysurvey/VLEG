"""
Ground-truth-aware re-audit of Experiment 5's committed reflections —
pairs each reflection with the SPECIFIC combo (true attributes + true
label) it was generated from, and asks the oracle two independent
questions: was the label call correct, and (independently) was the
causal justification clean, over-specified, or causally wrong. This
disentangles the confound found in the previous (text-only) detailed
audit, where identical-looking judgments were being made blind to which
instance a reflection actually came from.

Run:
    python -m experiments.reaudit_experiment5_ground_truth
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.task_env import make_level3_split, all_combinations
from core.oracle_verifier import audit_with_ground_truth

# Reflections copied verbatim from the original Experiment 5 run log,
# in the same order as the combos they were generated from.
LEVEL2_SINGLE_STEP = [
    "On a white background, circle+red scenes are ANOMALY; this scene is therefore ANOMALY.",
    "On a black background, small circle+red scenes are ANOMALY; this scene is therefore ANOMALY.",
    "On a white background, striped large shapes are ANOMALY; this scene is therefore ANOMALY.",
    "On a black background, striped scenes are ANOMALY; this scene is therefore ANOMALY.",
    "On a white background, blue solid small circle scenes are ANOMALY; this scene is therefore ANOMALY.",
    "On a black background, large solid blue circle scenes are ANOMALY; this scene is therefore ANOMALY.",
    "On a white background, striped scenes are ANOMALY; this scene is therefore ANOMALY.",
    "On a black background, striped large circle scenes are ANOMALY; this scene is therefore ANOMALY.",
    "On a white background, large solid-colored square scenes are ANOMALY; this scene is therefore ANOMALY.",
    "On a black background, large solid-colored square scenes are ANOMALY; this scene is therefore ANOMALY.",
    "On a white background, striped red small shapes are ANOMALY; this scene is therefore ANOMALY.",
    "On a black background, striped scenes are ANOMALY; this scene is therefore ANOMALY.",
    "On a white background, large solid blue square scenes are ANOMALY; this scene is therefore ANOMALY.",
    "On a black background, large solid-colored square scenes are ANOMALY; this scene is therefore ANOMALY.",
    "On a white background, large striped scenes are ANOMALY; this scene is therefore ANOMALY.",
    "On a black background, large striped scenes are ANOMALY; this scene is therefore ANOMALY.",
]

# NOTE: episode 7 (0-indexed 6) was DISCARDED under VLEG (never verified),
# so it's omitted here — 15 entries, aligned to combo indices
# [0,1,2,3,4,5,7,8,9,10,11,12,13,14,15] (skipping index 6).
LEVEL2_VLEG = [
    (0, "On a white background, circle+red scenes are ANOMALY; this scene is therefore ANOMALY."),
    (1, "On a black background, small circle+red scenes are NORMAL; this scene is therefore NORMAL."),
    (2, "On a white background, striped large shapes are ANOMALY; this scene is therefore ANOMALY."),
    (3, "On a black background, circle+red scenes are NORMAL; this scene is therefore NORMAL."),
    (4, "On a white background, blue solid small circle scenes are NORMAL; this scene is therefore NORMAL."),
    (5, "On a black background, large solid blue circle scenes are ANOMALY; this scene is therefore ANOMALY."),
    (7, "On a black background, striped large circle scenes are ANOMALY; this scene is therefore ANOMALY."),
    (8, "On a white background, large solid-colored square scenes are NORMAL; this scene is therefore NORMAL."),
    (9, "On a black background, large solid-colored square scenes are ANOMALY; this scene is therefore ANOMALY."),
    (10, "On a white background, striped red small shapes are NORMAL; this scene is therefore NORMAL."),
    (11, "On a black background, striped scenes are ANOMALY; this scene is therefore ANOMALY."),
    (12, "On a white background, large solid blue square scenes are NORMAL; this scene is therefore NORMAL."),
    (13, "On a black background, large solid-colored square scenes are ANOMALY; this scene is therefore ANOMALY."),
    (14, "On a white background, large striped scenes are NORMAL; this scene is therefore NORMAL."),
    (15, "On a black background, large striped scenes are ANOMALY; this scene is therefore ANOMALY."),
]

LEVEL3_SINGLE_STEP = [
    "On a white background, large striped scenes are ANOMALY; this scene is therefore ANOMALY.",
    "On a white background, striped red small shapes are ANOMALY; this scene is therefore ANOMALY.",
    "On a white background, large solid blue square scenes are ANOMALY; this scene is therefore ANOMALY.",
    "On a white background, large solid-colored square scenes are ANOMALY; this scene is therefore ANOMALY.",
    "On a black background, small circle+red scenes are ANOMALY; this scene is therefore ANOMALY.",
    "On a white background, striped large shapes are ANOMALY; this scene is therefore ANOMALY.",
    "On a black background, large solid-colored square scenes are ANOMALY; this scene is therefore ANOMALY.",
    "On a black background, large striped scenes are ANOMALY; this scene is therefore ANOMALY.",
    "On a white background, blue solid small circle scenes are ANOMALY; this scene is therefore ANOMALY.",
    "On a black background, striped scenes are ANOMALY; this scene is therefore ANOMALY.",
    "On a white background, circle+red scenes are ANOMALY; this scene is therefore ANOMALY.",
    "On a black background, striped scenes are ANOMALY; this scene is therefore ANOMALY.",
]

# All 12 Level3 training episodes were verified under VLEG (100% validity
# in the original run), so this list aligns 1:1 with train_combos.
LEVEL3_VLEG = [
    "On a white background, large striped scenes are NORMAL; this scene is therefore NORMAL.",
    "On a white background, striped red small shapes are NORMAL; this scene is therefore NORMAL.",
    "On a white background, large solid blue square scenes are NORMAL; this scene is therefore NORMAL.",
    "On a white background, large solid-colored square scenes are NORMAL; this scene is therefore NORMAL.",
    "On a black background, small circle+red scenes are NORMAL; this scene is therefore NORMAL.",
    "On a white background, striped large shapes are ANOMALY; this scene is therefore ANOMALY.",
    "On a black background, large solid-colored square scenes are ANOMALY; this scene is therefore ANOMALY.",
    "On a black background, large striped scenes are ANOMALY; this scene is therefore ANOMALY.",
    "On a white background, blue solid small circle scenes are NORMAL; this scene is therefore NORMAL.",
    "On a black background, striped scenes are ANOMALY; this scene is therefore ANOMALY.",
    "On a white background, circle+red scenes are ANOMALY; this scene is therefore ANOMALY.",
    "On a black background, circle+red scenes are NORMAL; this scene is therefore NORMAL.",
]


def audit_condition(level_name, strategy, combo_reflection_pairs):
    print(f"\n=== {level_name} / {strategy} ({len(combo_reflection_pairs)} reflections) ===")
    label_correct_count = 0
    quality_counts = Counter()          # over ALL reflections
    quality_counts_given_correct = Counter()  # only where label_correct
    details = []
    for combo, reflection in combo_reflection_pairs:
        result = audit_with_ground_truth(combo, reflection)
        label_correct_count += int(result["label_correct"])
        quality_counts[result["condition_quality"]] += 1
        if result["label_correct"]:
            quality_counts_given_correct[result["condition_quality"]] += 1
        details.append({"combo": {k: v for k, v in combo.items() if k != "masked_at_train"},
                         "reflection": reflection, **result})
        print(f"  label_correct={result['label_correct']!s:<5} "
              f"quality={result['condition_quality']:<15} \"{reflection[:70]}...\" — {result['reason']}")

    n = len(combo_reflection_pairs)
    label_correct_rate = label_correct_count / n if n else 0.0
    n_correct = label_correct_count
    print(f"\n-- {level_name}/{strategy}: label_correct_rate={label_correct_rate*100:.1f}% "
          f"({label_correct_count}/{n})")
    if n_correct:
        print(f"   Among label-correct reflections, condition quality breakdown:")
        for cat in ("clean_correct", "over_specified", "causally_wrong", "vague"):
            c = quality_counts_given_correct.get(cat, 0)
            print(f"     {cat:<16} {c}/{n_correct} = {c/n_correct*100:.1f}%")
    else:
        print("   (no label-correct reflections to break down)")

    return {
        "level": level_name, "strategy": strategy, "n": n,
        "label_correct_rate": label_correct_rate,
        "quality_given_correct": {k: v / n_correct if n_correct else None
                                   for k, v in quality_counts_given_correct.items()},
        "quality_overall": dict(quality_counts),
        "details": details,
    }


def main():
    level2_combos = all_combinations()
    level3_train_combos, _ = make_level3_split(mask_count=4, seed=42)

    results = []
    results.append(audit_condition(
        "Level2", "single_step",
        list(zip(level2_combos, LEVEL2_SINGLE_STEP))))
    results.append(audit_condition(
        "Level2", "VLEG",
        [(level2_combos[i], r) for i, r in LEVEL2_VLEG]))
    results.append(audit_condition(
        "Level3", "single_step",
        list(zip(level3_train_combos, LEVEL3_SINGLE_STEP))))
    results.append(audit_condition(
        "Level3", "VLEG",
        list(zip(level3_train_combos, LEVEL3_VLEG))))

    print(f"\n\n{'='*100}")
    print(f"{'Level':<8} {'Strategy':<14} {'Label-correct':<15} {'clean|overspec|wrong (of correct)'}")
    print('='*100)
    for r in results:
        q = r["quality_given_correct"]
        clean = q.get("clean_correct")
        over = q.get("over_specified")
        wrong = q.get("causally_wrong")
        fmt = lambda x: f"{x*100:.0f}%" if x is not None else "-"
        print(f"{r['level']:<8} {r['strategy']:<14} {r['label_correct_rate']*100:>13.1f}%   "
              f"{fmt(clean):>6} | {fmt(over):>7} | {fmt(wrong):>6}")

    out_path = Path(__file__).resolve().parent.parent / "outputs" / "experiment5_ground_truth_audit.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
