"""
Targeted fix for the sphere+blue+metal (both sizes) consolidation gap
that has affected several diagnostics since #27 (see DEVIATIONS_LOG.md
#29, #34): this specific combo was DISCARDED at K=3 in both prior
consolidation attempts, leaving the reflection library incomplete and
confounding the shape-dimension check in #34.

Since temperature=0 consolidation is otherwise deterministic in this
project, simply re-running the same K=3 attempt is unlikely to change
the outcome. This script instead gives ONLY these two combos a larger
budget (K=5) to see if they converge with more attempts, and updates
the cached reflection_library_all16.json in place if they do.

Run:
    python -m experiments_visual.fill_sphere_blue_metal_gap
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_visual.llm_client import check_connection
from core_visual.task_env import consolidation_combo, verifier_view_text, generator_view_text, _load_index
from core_visual.refinement import refine_episode

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs_visual"
LIBRARY_PATH = OUTPUT_DIR / "reflection_library_all16.json"

TARGET_KEYS = [("sphere", "blue", "metal", "large"), ("sphere", "blue", "metal", "small")]
K_RETRY = 5


def main():
    print("Checking Ollama connection...")
    check_connection()

    if not LIBRARY_PATH.exists():
        print(f"ERROR: {LIBRARY_PATH} not found.")
        return
    library = json.loads(LIBRARY_PATH.read_text())
    by_combo = _load_index()

    for key in TARGET_KEYS:
        print(f"\n=== Retrying {key} with K={K_RETRY} (was DISCARDED at K=3 twice previously) ===")
        combo = consolidation_combo(key, by_combo)
        vtext = verifier_view_text(combo)
        gtext = generator_view_text(combo)
        result = refine_episode(combo["image_path"], gtext, vtext, K=K_RETRY)

        if result.verified:
            print(f"  SUCCESS after {result.iterations_used} iter(s) -> \"{result.final_reflection}\"")
            library[str(key)] = {
                "key": list(key), "label": combo["label"], "verified": True,
                "reflection": result.final_reflection, "image_path": combo["image_path"],
            }
        else:
            print(f"  STILL DISCARDED after {K_RETRY} iter(s), even with the larger budget.")
            print(f"  Full attempt history:")
            for j, (reflection, valid, critique) in enumerate(result.history, start=1):
                print(f"    attempt {j}: \"{reflection}\" -> valid={valid}, critique=\"{critique}\"")
            library[str(key)] = {
                "key": list(key), "label": combo["label"], "verified": False,
                "reflection": None, "image_path": combo["image_path"],
            }
        LIBRARY_PATH.write_text(json.dumps(library, indent=2))
        print(f"  [checkpoint] library updated at {LIBRARY_PATH}")

    n_verified = sum(1 for e in library.values() if e["verified"])
    print(f"\nLibrary now has {n_verified}/16 combos verified.")


if __name__ == "__main__":
    main()
