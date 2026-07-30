"""
Large-scale, multi-instance real-image (CLEVR) Experience-Transfer
environment. Supersedes the earlier single-instance-per-combo pilot
version (which used clevr_crops/index.json, 1 image per combo).

Loads clevr_crops_large/index.json (built by prepare_clevr.py with
--n-per-combo 32): 16 attribute combinations, ~32 distinct CLEVR crop
instances each.

Key design change from the pilot: CONSOLIDATION still uses exactly ONE
representative instance per (unmasked) combo -- there is no benefit to
running the generate-verify-refine loop 32 times on the same attribute
combination, since the resulting rule should already generalize across
instances of it. EVALUATION now samples MANY held-out instances per
combo (default 10), guaranteed disjoint from the consolidation image,
so the test set actually exercises genuine instance-level
generalization (never-seen-exact-image) in addition to the existing
combo-level masking (never-seen-attribute-combination). This also
directly fixes a recurring problem throughout this project: previous
masked-combo accuracy was always estimated from just 4 test items
(n=4), producing very noisy per-seed estimates; with 10 instances per
masked combo, masked-combo accuracy is now estimated from n=40.

Attribute mapping unchanged from the pilot (see DEVIATIONS_LOG.md):
  shape    in {cube, sphere}   -- causal
  color    in {red, blue}      -- causal
  material in {rubber, metal}  -- background-analog, flips the rule
  size     in {large, small}   -- distractor, non-causal
"""
import itertools
import json
import random
from pathlib import Path

INDEX_PATH = Path(__file__).resolve().parent.parent / "clevr_crops_large" / "index.json"

SHAPES = ["cube", "sphere"]
COLORS = ["red", "blue"]
MATERIALS = ["rubber", "metal"]
SIZES = ["large", "small"]


def oracle_label(shape: str, color: str, material: str) -> str:
    is_cube_red = (shape == "cube" and color == "red")
    if material == "rubber":
        anomaly = is_cube_red
    else:  # metal -- rule flips
        anomaly = not is_cube_red
    return "ANOMALY" if anomaly else "NORMAL"


def _load_index() -> dict:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"clevr_crops_large/index.json not found at {INDEX_PATH}. "
            f"Run `python prepare_clevr.py --clevr-root <path> --split train "
            f"--n-per-combo 32` first."
        )
    entries = json.loads(INDEX_PATH.read_text())
    by_combo = {(e["shape"], e["color"], e["material"], e["size"]): e for e in entries}
    missing = []
    for shape, color, material, size in itertools.product(SHAPES, COLORS, MATERIALS, SIZES):
        if (shape, color, material, size) not in by_combo:
            missing.append((shape, color, material, size))
    if missing:
        raise ValueError(f"clevr_crops_large/index.json is missing {len(missing)} combo(s): {missing}")
    return by_combo


def all_combo_keys():
    """The 16 (shape, color, material, size) attribute tuples, independent of instance count."""
    return list(itertools.product(SHAPES, COLORS, MATERIALS, SIZES))


def _combo_base(shape, color, material, size) -> dict:
    return {"shape": shape, "color": color, "material": material, "size": size,
            "label": oracle_label(shape, color, material)}


def consolidation_combo(key, by_combo=None) -> dict:
    """The ONE designated training instance for this combo (always instance #1,
    deterministic -- so re-runs consolidate the same image)."""
    by_combo = by_combo or _load_index()
    shape, color, material, size = key
    entry = by_combo[key]
    base = _combo_base(*key)
    base["image_path"] = entry["instances"][0]["image_path"]
    return base


def test_instances_for_combo(key, n: int, seed: int, by_combo=None) -> list:
    """Samples `n` held-out test instances for this combo, ALWAYS EXCLUDING
    instance #1 (reserved for consolidation), deterministic given `seed`."""
    by_combo = by_combo or _load_index()
    entry = by_combo[key]
    pool = entry["instances"][1:]  # exclude the consolidation instance
    if len(pool) < n:
        raise ValueError(f"combo {key} only has {len(pool)} non-consolidation instances, "
                          f"requested {n}. Re-run prepare_clevr.py with a larger --n-per-combo.")
    rng = random.Random(f"{seed}-{key}")  # per-combo deterministic sub-seed
    chosen = rng.sample(pool, n)
    base = _combo_base(*key)
    items = []
    for inst in chosen:
        item = dict(base)
        item["image_path"] = inst["image_path"]
        items.append(item)
    return items


def all_combinations():
    """Backward-compatible: one representative (consolidation) instance per
    combo, matching the pilot version's all_combinations() signature."""
    by_combo = _load_index()
    return [consolidation_combo(key, by_combo) for key in all_combo_keys()]


def level1_combinations():
    """8 combos on the uniform 'rubber' background-analog, consolidation instances."""
    return [c for c in all_combinations() if c["material"] == "rubber"]


def make_level3_split_large(mask_count: int = 4, seed: int = 42, n_test_per_combo: int = 10):
    """
    Returns (train_combos, test_items):
      train_combos: 12 consolidation-instance combos (the unmasked attribute
        combinations), each a single dict as before -- feed these to
        build_vleg_bank / refine_episode exactly as in the pilot version.
      test_items: n_test_per_combo * 16 dicts (one per sampled test
        instance across ALL 16 combos, masked and unmasked alike), each
        tagged with masked_at_train. Use ALL of these for evaluation.
    """
    by_combo = _load_index()
    keys = all_combo_keys()
    rng = random.Random(seed)
    shuffled = keys[:]
    rng.shuffle(shuffled)
    masked_keys = set(shuffled[:mask_count])
    train_keys = [k for k in keys if k not in masked_keys]

    train_combos = [consolidation_combo(k, by_combo) for k in train_keys]

    test_items = []
    for key in keys:
        instances = test_instances_for_combo(key, n_test_per_combo, seed, by_combo)
        for item in instances:
            item["masked_at_train"] = key in masked_keys
            test_items.append(item)

    return train_combos, test_items


def make_level3_split(mask_count: int = 4, seed: int = 42):
    """Backward-compatible alias matching the pilot's single-instance
    signature -- delegates to the large-scale version with n_test_per_combo=1
    (equivalent to the pilot's behavior). Prefer make_level3_split_large
    directly for new code."""
    train, test_items = make_level3_split_large(mask_count=mask_count, seed=seed, n_test_per_combo=1)
    return train, test_items


def verifier_view_text(combo: dict) -> str:
    return f"Observed scene (image attached). Ground-truth outcome label: {combo['label']}."


def generator_view_text(combo: dict) -> str:
    return ("Observed scene (image attached). Outcome label is unknown -- "
            "infer whether this scene is ANOMALY or NORMAL based on what you see.")


def query_text(combo: dict) -> str:
    return "New scene (image attached). Is this ANOMALY or NORMAL?"
