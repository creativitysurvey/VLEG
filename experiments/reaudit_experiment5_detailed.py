"""
Re-audit Experiment 5's already-committed reflections (copied verbatim
from the pasted run log) using the DETAILED oracle categories instead of
the binary hallucinated/not — splits "over-specified but core condition
correct" from "core condition genuinely wrong". No need to re-run
Generator/Verifier consolidation; only re-runs the oracle audit step.

Run:
    python -m experiments.reaudit_experiment5_detailed
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.oracle_verifier import audit_reflection_detailed

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

LEVEL2_VLEG = [
    "On a white background, circle+red scenes are ANOMALY; this scene is therefore ANOMALY.",
    "On a black background, small circle+red scenes are NORMAL; this scene is therefore NORMAL.",
    "On a white background, striped large shapes are ANOMALY; this scene is therefore ANOMALY.",
    "On a black background, circle+red scenes are NORMAL; this scene is therefore NORMAL.",
    "On a white background, blue solid small circle scenes are NORMAL; this scene is therefore NORMAL.",
    "On a black background, large solid blue circle scenes are ANOMALY; this scene is therefore ANOMALY.",
    "On a black background, striped large circle scenes are ANOMALY; this scene is therefore ANOMALY.",
    "On a white background, large solid-colored square scenes are NORMAL; this scene is therefore NORMAL.",
    "On a black background, large solid-colored square scenes are ANOMALY; this scene is therefore ANOMALY.",
    "On a white background, striped red small shapes are NORMAL; this scene is therefore NORMAL.",
    "On a black background, striped scenes are ANOMALY; this scene is therefore ANOMALY.",
    "On a white background, large solid blue square scenes are NORMAL; this scene is therefore NORMAL.",
    "On a black background, large solid-colored square scenes are ANOMALY; this scene is therefore ANOMALY.",
    "On a white background, large striped scenes are NORMAL; this scene is therefore NORMAL.",
    "On a black background, large striped scenes are ANOMALY; this scene is therefore ANOMALY.",
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

CONDITIONS = [
    ("Level2", "single_step", LEVEL2_SINGLE_STEP),
    ("Level2", "VLEG", LEVEL2_VLEG),
    ("Level3", "single_step", LEVEL3_SINGLE_STEP),
    ("Level3", "VLEG", LEVEL3_VLEG),
]


def main():
    all_results = []
    for level, strategy, reflections in CONDITIONS:
        print(f"\n=== {level} / {strategy} ({len(reflections)} reflections) ===")
        counts = Counter()
        details = []
        for r in reflections:
            category, reason = audit_reflection_detailed(r)
            counts[category] += 1
            details.append({"reflection": r, "category": category, "reason": reason})
            print(f"  [{category:>15}] \"{r}\" — {reason}")
        n = len(reflections)
        wrong_rate = counts["wrong_condition"] / n
        overspec_rate = counts["over_specified"] / n
        correct_rate = counts["correct"] / n
        vague_rate = counts["vague"] / n
        print(f"-- {level}/{strategy}: correct={correct_rate*100:.1f}%  "
              f"over_specified={overspec_rate*100:.1f}%  "
              f"wrong_condition={wrong_rate*100:.1f}%  vague={vague_rate*100:.1f}%")
        all_results.append({
            "level": level, "strategy": strategy, "n": n,
            "correct_rate": correct_rate, "over_specified_rate": overspec_rate,
            "wrong_condition_rate": wrong_rate, "vague_rate": vague_rate,
            "details": details,
        })

    print(f"\n\n{'='*100}")
    print(f"{'Level':<8} {'Strategy':<14} {'Correct':<10} {'Over-specified':<16} {'Wrong condition':<18} {'Vague':<8}")
    print('='*100)
    for r in all_results:
        print(f"{r['level']:<8} {r['strategy']:<14} {r['correct_rate']*100:>7.1f}%  "
              f"{r['over_specified_rate']*100:>13.1f}%  {r['wrong_condition_rate']*100:>15.1f}%  "
              f"{r['vague_rate']*100:>5.1f}%")

    out_path = Path(__file__).resolve().parent.parent / "outputs" / "experiment5_detailed_audit.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
