"""
Experiment 5 (real-image / CLEVR) -- Diagnostic Assessment of Memory
Quality. Compares single-step (unverified, always-commit) vs VLEG
(K=3, verify-or-discard) memory banks, auditing every committed
reflection with DETERMINISTIC checks against known ground truth --
no LLM oracle is used (see DEVIATIONS_LOG.md #12 for why the symbolic
pipeline's LLM-oracle audit was found unreliable; this project has
since standardized on deterministic ground-truth + keyword checks).

Checks performed on every committed reflection, using the REAL
attribute ground truth for that specific image (something the symbolic
pipeline could not check at all, since it had no independent "ground
truth image" to compare against -- this is a genuinely new capability
of the real-image pipeline):
  - label_correct: does the reflection's stated ANOMALY/NORMAL
    conclusion match the true label?
  - shape_hallucination / color_hallucination / material_hallucination:
    does the reflection's text correctly name the TRUE shape/color/
    material of the image it was generated from, or does it claim a
    different one (a genuine visual-attribute hallucination, distinct
    from over-specification)?
  - spurious_size: does the reflection mention the known non-causal
    distractor attribute (size) as if it were determinative?

Run:
    python -m experiments_visual.experiment5_memory_quality
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_visual.llm_client import check_connection, MODEL_NAME
from core_visual.task_env import make_level3_split_large, verifier_view_text, generator_view_text
from core_visual.refinement import refine_episode
from core_visual.single_step import single_step_commit
from core_visual.memory import MemoryBank

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs_visual"
OUTPUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUTPUT_DIR / "experiment5_visual_results.json"

CUBE_WORDS = ["cube", "box", "rectangular", "square", "block"]
SPHERE_WORDS = ["sphere", "ball", "round", "circular"]
RED_WORDS = ["red"]
BLUE_WORDS = ["blue"]
MATTE_WORDS = ["matte", "rubber", "dull"]
GLOSSY_WORDS = ["glossy", "metal", "metallic", "shiny", "reflective"]
SIZE_WORDS = ["large", "small", "larger", "smaller", "bigger", "size", "medium", "half the frame"]


def _contains_any(text: str, words: list) -> bool:
    low = text.lower()
    return any(w in low for w in words)


def audit_reflection(reflection: str, combo: dict) -> dict:
    low = reflection.lower()
    stated_label = None
    m = re.search(r"therefore\s+(ANOMALY|NORMAL)", reflection, re.IGNORECASE)
    if m:
        stated_label = m.group(1).upper()
    label_correct = (stated_label == combo["label"])

    true_shape, true_color, true_material = combo["shape"], combo["color"], combo["material"]
    wrong_shape_words = SPHERE_WORDS if true_shape == "cube" else CUBE_WORDS
    right_shape_words = CUBE_WORDS if true_shape == "cube" else SPHERE_WORDS
    shape_hallucination = _contains_any(low, wrong_shape_words) and not _contains_any(low, right_shape_words)

    wrong_color_words = BLUE_WORDS if true_color == "red" else RED_WORDS
    right_color_words = RED_WORDS if true_color == "red" else BLUE_WORDS
    color_hallucination = _contains_any(low, wrong_color_words) and not _contains_any(low, right_color_words)

    wrong_material_words = GLOSSY_WORDS if true_material == "rubber" else MATTE_WORDS
    right_material_words = MATTE_WORDS if true_material == "rubber" else GLOSSY_WORDS
    material_hallucination = _contains_any(low, wrong_material_words) and not _contains_any(low, right_material_words)

    spurious_size = _contains_any(low, SIZE_WORDS)

    return {
        "stated_label": stated_label, "label_correct": label_correct,
        "shape_hallucination": shape_hallucination, "color_hallucination": color_hallucination,
        "material_hallucination": material_hallucination, "spurious_size": spurious_size,
    }


def build_vleg_bank(train_combos, K, log_prefix):
    bank = MemoryBank()
    for idx, combo in enumerate(train_combos):
        vtext = verifier_view_text(combo)
        gtext = generator_view_text(combo)
        result = refine_episode(combo["image_path"], gtext, vtext, K=K)
        if result.verified:
            bank.add(combo, combo["image_path"], result.final_reflection)
            print(f"  [{log_prefix}] episode {idx+1}/{len(train_combos)} "
                  f"({combo['shape']}/{combo['color']}/{combo['material']}/{combo['size']}, "
                  f"true={combo['label']}): VERIFIED after {result.iterations_used} iter(s) -> "
                  f"\"{result.final_reflection}\"")
        else:
            print(f"  [{log_prefix}] episode {idx+1}/{len(train_combos)}: DISCARDED after {K} iter(s)")
    return bank


def build_single_step_bank_local(train_combos, log_prefix):
    bank = MemoryBank()
    for idx, combo in enumerate(train_combos):
        gtext = generator_view_text(combo)
        reflection = single_step_commit(combo["image_path"], gtext)
        bank.add(combo, combo["image_path"], reflection)
        print(f"  [{log_prefix}] episode {idx+1}/{len(train_combos)} "
              f"({combo['shape']}/{combo['color']}/{combo['material']}/{combo['size']}, "
              f"true={combo['label']}): COMMITTED (no gate) -> \"{reflection}\"")
    return bank


def summarize(bank, label):
    n = len(bank.tuples)
    if n == 0:
        return {"strategy": label, "n_committed": 0}
    audits = [audit_reflection(t.verified_reflection, t.combo) for t in bank.tuples]
    n_label_correct = sum(a["label_correct"] for a in audits)
    correct_audits = [a for i, a in enumerate(audits) if a["label_correct"]]
    n_correct = len(correct_audits)
    def rate(key, pool):
        return sum(a[key] for a in pool) / len(pool) if pool else None
    return {
        "strategy": label, "n_committed": n,
        "label_correct_rate": n_label_correct / n,
        "shape_hallucination_rate_given_correct": rate("shape_hallucination", correct_audits),
        "color_hallucination_rate_given_correct": rate("color_hallucination", correct_audits),
        "material_hallucination_rate_given_correct": rate("material_hallucination", correct_audits),
        "spurious_size_rate_given_correct": rate("spurious_size", correct_audits),
        "per_item": [{"combo": {k: v for k, v in t.combo.items()}, "reflection": t.verified_reflection,
                      **a} for t, a in zip(bank.tuples, audits)],
    }


def main():
    print(f"Checking Ollama connection (model={MODEL_NAME!r})...")
    check_connection()

    train_combos, _ = make_level3_split_large(mask_count=4, seed=42, n_test_per_combo=1)
    print(f"\n{len(train_combos)} training combos (Level 3, seed=42).")

    results = {}

    print(f"\n=== single_step (unverified, always commits) ===")
    ss_bank = build_single_step_bank_local(train_combos, "single_step")
    results["single_step"] = summarize(ss_bank, "single_step")
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"  [checkpoint] written to {OUT_PATH}")

    print(f"\n=== VLEG (K=3, verify-or-discard) ===")
    vleg_bank = build_vleg_bank(train_combos, K=3, log_prefix="VLEG")
    results["VLEG"] = summarize(vleg_bank, "VLEG")
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"  [checkpoint] written to {OUT_PATH}")

    print(f"\n\n{'='*100}")
    print(f"{'Strategy':<14} {'Committed':<10} {'Label-correct':<15} {'Shape-halluc':<14} "
          f"{'Color-halluc':<14} {'Material-halluc':<16} {'Spurious-size':<14}")
    print('='*100)
    for name, r in results.items():
        def fmt(v):
            return f"{v*100:.1f}%" if v is not None else "N/A"
        print(f"{r['strategy']:<14} {r['n_committed']:<10} {fmt(r.get('label_correct_rate')):<15} "
              f"{fmt(r.get('shape_hallucination_rate_given_correct')):<14} "
              f"{fmt(r.get('color_hallucination_rate_given_correct')):<14} "
              f"{fmt(r.get('material_hallucination_rate_given_correct')):<16} "
              f"{fmt(r.get('spurious_size_rate_given_correct')):<14}")

    print(f"\nFull results written to {OUT_PATH}")


if __name__ == "__main__":
    main()
