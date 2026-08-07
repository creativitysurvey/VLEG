"""
Diagnostic: predict_hybrid collapsed to predicting NORMAL for all 16/16
test queries under BOTH K=1 and K=3 (see chat log). Isolates whether
this is (a) a many-images-in-one-message scaling issue, (b) something
about the ANOMALY/NORMAL classification framing specifically (as
opposed to simple image description, which worked fine in
diagnose_multi_image.py), or (c) both.

Reuses the ALREADY-VERIFIED K=3 reflections from the earlier pilot run
(pasted from its terminal log) -- no need to re-run the expensive
consolidation phase.

Run:
    python -m experiments_visual.diagnose_hybrid_collapse
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_visual.llm_client import chat, check_connection
from core_visual.task_env import all_combinations, query_text

combos_by_key = {(c["shape"], c["color"], c["material"], c["size"]): c for c in all_combinations()}

# Verified K=3 reflections, copied verbatim from the pilot run's terminal log.
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

SYSTEM_PROMPT_HYBRID = (
    "You classify new scenes as ANOMALY or NORMAL. You will see several "
    "past example images IN ORDER, each followed by a verified rule "
    "that was validated against it, and FINALLY a new target image to "
    "classify (always the LAST image shown). Use the examples, their "
    "rules, and your own visual inspection of the target image to "
    "decide. Reply with ONLY one word: ANOMALY or NORMAL."
)


def predict_with_n_examples(n_examples: int, target_combo: dict) -> str:
    tuples = K3_VERIFIED[:n_examples]
    image_paths = []
    parts = []
    for i, (key, reflection) in enumerate(tuples, start=1):
        combo = combos_by_key[key]
        image_paths.append(combo["image_path"])
        parts.append(f"[Image {i}] Past example. Verified rule: {reflection}")
    target_idx = len(tuples) + 1
    image_paths.append(target_combo["image_path"])
    parts.append(f"[Image {target_idx}] TARGET scene to classify (this is the last image). "
                 f"{query_text(target_combo)}")
    prompt = "\n\n".join(parts)
    reply = chat([{"role": "system", "content": SYSTEM_PROMPT_HYBRID},
                  {"role": "user", "content": prompt}], temperature=0.0, images=image_paths)
    match = re.search(r"\bANOMALY\b|\bNORMAL\b", reply.upper())
    return (match.group(0) if match else "NORMAL"), reply.strip()


def predict_zero_shot(target_combo: dict) -> str:
    """Just the target image, no examples at all, direct classification ask."""
    prompt = (
        "Classify this scene as ANOMALY or NORMAL based on whatever visual pattern "
        "you can infer. Reply with ONLY one word: ANOMALY or NORMAL."
    )
    reply = chat([{"role": "user", "content": prompt}], temperature=0.0,
                 images=[target_combo["image_path"]])
    match = re.search(r"\bANOMALY\b|\bNORMAL\b", reply.upper())
    return (match.group(0) if match else "NORMAL"), reply.strip()


print("Checking Ollama connection...")
check_connection()

# Two target combos with different true labels, to see if predictions ever vary at all.
target_anomaly = combos_by_key[("cube", "blue", "metal", "large")]   # true=ANOMALY, masked at train
target_normal = combos_by_key[("cube", "red", "metal", "large")]     # true=NORMAL

print("\n--- Test A: zero-shot (no examples), direct ANOMALY/NORMAL ask ---")
for name, combo in [("target_anomaly (true=ANOMALY)", target_anomaly), ("target_normal (true=NORMAL)", target_normal)]:
    label, raw = predict_zero_shot(combo)
    print(f"  {name}: predicted={label}  (raw reply: {raw!r})")

print("\n--- Test B: scaling number of hybrid examples (0, 1, 3, 6, 11) ---")
for n in [0, 1, 3, 6, 11]:
    print(f"\n  n_examples={n}:")
    for name, combo in [("target_anomaly", target_anomaly), ("target_normal", target_normal)]:
        label, raw = predict_with_n_examples(n, combo)
        print(f"    {name} (true={combo['label']}): predicted={label}  (raw: {raw!r})")
