"""
Diagnostic: does qwen2.5vl correctly associate positional text labels
("[Image 1]", "[Image 2]", ...) with the actual images passed in the
same order? Tests this directly and cheaply before trusting any
multi-image hybrid-grounding result (see chat log: K=1 and K=3 tied at
exactly 50% in the Experiment 3 pilot, despite K=3 having much better
memory bank coverage -- suspicious).

Run:
    python -m experiments_visual.diagnose_multi_image
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_visual.llm_client import chat, check_connection
from core_visual.task_env import all_combinations

combos = all_combinations()
# pick 4 combos with visually distinct shape+color so a correct answer is unambiguous
by_key = {(c["shape"], c["color"]): c for c in combos}
picks = [
    by_key[("cube", "red")],
    by_key[("sphere", "blue")],
    by_key[("cube", "blue")],
    by_key[("sphere", "red")],
]

print("Checking Ollama connection...")
check_connection()

print("\n--- Test 1: single image, direct question ---")
for combo in picks:
    reply = chat(
        [{"role": "user", "content": "What shape and color is the object? Answer in 2 words."}],
        images=[combo["image_path"]],
    )
    print(f"  {combo['shape']}/{combo['color']} ({combo['image_path']}) -> model said: {reply.strip()!r}")

print("\n--- Test 2: 4 images in ONE message, asked to describe EACH by position ---")
image_paths = [c["image_path"] for c in picks]
prompt = (
    "You are given 4 images, in order: [Image 1], [Image 2], [Image 3], [Image 4]. "
    "For EACH image, state its shape and color. Answer as a numbered list:\n"
    "1. <shape> <color>\n2. <shape> <color>\n3. <shape> <color>\n4. <shape> <color>"
)
reply = chat([{"role": "user", "content": prompt}], images=image_paths)
print(f"  True order: {[(c['shape'], c['color']) for c in picks]}")
print(f"  Model said:\n{reply}")

print("\n--- Test 3: 4 images, explicitly asking to classify ONLY the LAST one ---")
prompt2 = (
    "You are given 4 images in order: [Image 1], [Image 2], [Image 3], [Image 4]. "
    "Ignore images 1-3. Describe ONLY [Image 4]'s shape and color, in 2 words."
)
reply2 = chat([{"role": "user", "content": prompt2}], images=image_paths)
print(f"  True [Image 4]: {picks[3]['shape']}/{picks[3]['color']}")
print(f"  Model said: {reply2.strip()!r}")
