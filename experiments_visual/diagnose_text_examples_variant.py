"""
Follow-up diagnostic: does collapse to constant NORMAL go away if past
examples are shown as TEXT-ONLY (their verified rule, no image), with
ONLY the target combo's image actually attached? Tests the hypothesis
that the bottleneck in diagnose_hybrid_collapse.py is specifically the
NUMBER OF IMAGES in one message, not the amount of textual context.

Writes incremental checkpoints to
outputs_visual/diagnose_text_examples_variant_results.json after EVERY
single prediction -- this is a short diagnostic but the person's SSH
session has been dropping, so nothing should depend on the terminal
still being attached when it finishes.

Run:
    python -m experiments_visual.diagnose_text_examples_variant
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_visual.llm_client import chat, check_connection
from core_visual.task_env import all_combinations, query_text

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs_visual"
OUTPUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUTPUT_DIR / "diagnose_text_examples_variant_results.json"

combos_by_key = {(c["shape"], c["color"], c["material"], c["size"]): c for c in all_combinations()}

K3_VERIFIED = [
    (("sphere", "blue", "metal", "large"), "Glossy blue sphere scenes are ANOMALY; this scene is therefore ANOMALY."),
    (("sphere", "red", "metal", "large"), "Glossy red sphere scenes are ANOMALY; this scene is therefore ANOMALY."),
    (("sphere", "blue", "rubber", "large"), "Matte blue sphere scenes are NORMAL; this scene is therefore NORMAL."),
    (("sphere", "red", "rubber", "large"), "Matte red sphere scenes are NORMAL; this scene is therefore NORMAL."),
    (("cube", "red", "rubber", "small"), "Matte red cube scenes are ANOMALY; this scene is therefore ANOMALY."),
    (("cube", "red", "metal", "large"), "Glossy red cube scenes are NORMAL; this scene is therefore NORMAL."),
    (("sphere", "blue", "rubber", "small"), "Matte blue sphere scenes are NORMAL; this scene is therefore NORMAL."),
    (("cube", "blue", "rubber", "large"), "Matte blue rectangular prism scenes are NORMAL; this scene is therefore NORMAL."),
    (("sphere", "red", "metal", "small"), "Glossy red spheres on a plain background are ANOMALY; this scene is therefore ANOMALY."),
    (("cube", "red", "rubber", "large"), "Matte red cubes larger than half the frame are ANOMALY; this scene is therefore ANOMALY."),
    (("cube", "red", "metal", "small"), "Glossy red cube scenes are NORMAL; this scene is therefore NORMAL."),
]

SYSTEM_PROMPT_TEXT_EXAMPLES = (
    "You classify new scenes as ANOMALY or NORMAL. You are given several "
    "verified rules distilled from past observations (no images of "
    "those past scenes are shown to you, only the rules), followed by "
    "ONE target image to classify. Use the rules and your own visual "
    "inspection of the target image to decide. "
    "Reply with ONLY one word: ANOMALY or NORMAL."
)


def predict_text_examples(n_examples: int, target_combo: dict):
    tuples = K3_VERIFIED[:n_examples]
    parts = [f"Verified rule {i}: {reflection}" for i, (key, reflection) in enumerate(tuples, start=1)]
    parts.append(f"Target scene (image attached). {query_text(target_combo)}")
    prompt = "\n\n".join(parts)
    reply = chat([{"role": "system", "content": SYSTEM_PROMPT_TEXT_EXAMPLES},
                  {"role": "user", "content": prompt}], temperature=0.0,
                 images=[target_combo["image_path"]])  # ONLY the target image, always
    match = re.search(r"\bANOMALY\b|\bNORMAL\b", reply.upper())
    return (match.group(0) if match else "NORMAL"), reply.strip()


def save_partial(results):
    OUT_PATH.write_text(json.dumps(results, indent=2))


def main():
    print("Checking Ollama connection...")
    check_connection()

    target_anomaly = combos_by_key[("cube", "blue", "metal", "large")]
    target_normal = combos_by_key[("cube", "red", "metal", "large")]
    target_anomaly2 = combos_by_key[("sphere", "blue", "metal", "small")]
    target_normal2 = combos_by_key[("sphere", "red", "rubber", "small")]

    targets = [
        ("target_anomaly", target_anomaly),
        ("target_normal", target_normal),
        ("target_anomaly2", target_anomaly2),
        ("target_normal2", target_normal2),
    ]

    results = []
    print("\n--- Text-only examples + single target image, scaling n_examples (0, 1, 3, 6, 11) ---")
    for n in [0, 1, 3, 6, 11]:
        print(f"\n  n_examples={n}:")
        for name, combo in targets:
            label, raw = predict_text_examples(n, combo)
            correct = (label == combo["label"])
            print(f"    {name} (true={combo['label']}): predicted={label} "
                  f"[{'OK' if correct else 'WRONG'}]  (raw: {raw!r})")
            results.append({
                "n_examples": n, "target_name": name,
                "shape": combo["shape"], "color": combo["color"],
                "material": combo["material"], "size": combo["size"],
                "true_label": combo["label"], "predicted": label,
                "correct": correct, "raw_reply": raw,
            })
            save_partial(results)  # checkpoint after EVERY single prediction

    n_correct = sum(1 for r in results if r["correct"])
    print(f"\nOverall: {n_correct}/{len(results)} correct across all (n_examples, target) combinations.")
    print(f"Full results written to {OUT_PATH}")


if __name__ == "__main__":
    main()
