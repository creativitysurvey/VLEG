"""
Diagnostic: does the Verifier actually reject bad reflections, or does it
rubber-stamp everything? Run this BEFORE trusting Experiment 3's K=1 vs K=3
comparison — if the verifier never rejects anything here either, that
confirms it's not doing real work in the main experiment.

Run:
    python -m experiments.sanity_check_verifier
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.verifier import verify_reflection
from core.task_env import episode_text

# A real training episode
combo = {"shape": "circle", "color": "red", "texture": "solid", "background": "white", "size": "large", "label": "ANOMALY"}
etext = episode_text(combo)
print(f"Episode: {etext}\n")

test_cases = [
    ("Correct reflection", "On a white background, circle+red scenes are ANOMALY."),
    ("WRONG label claim", "On a white background, circle+red scenes are NORMAL."),
    ("WRONG condition (references wrong color)", "On a white background, circle+blue scenes are ANOMALY."),
    ("Hallucinated attribute (size not in episode)", "Large circle+red scenes on white backgrounds are ANOMALY."),
    ("Overly vague / unfalsifiable", "Unusual-looking scenes tend to be ANOMALY."),
    ("Over-specified (adds irrelevant texture claim)",
     "On a white background, circle+red+solid scenes are ANOMALY, but circle+red+striped scenes are NORMAL."),
]

for name, reflection in test_cases:
    valid, critique = verify_reflection(etext, reflection)
    status = "ACCEPTED" if valid else "REJECTED"
    print(f"[{status}] {name}")
    print(f"    reflection: \"{reflection}\"")
    print(f"    critique:   \"{critique}\"\n")
