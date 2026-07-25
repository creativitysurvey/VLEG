"""
Synthetic Level 3 Experience-Transfer task environment.

Symbolic stand-in for the paper's multi-modal object-combination task
(see DEVIATIONS_LOG.md #1, #4, #5).

Attribute space (causal):
  shape   in {circle, square}
  color   in {red, blue}
  texture in {solid, striped}
  -> 8 "core combos" (matches paper's Level-1 count)

  background in {white, black}
  -> 16 (core_combo, background) pairs total (matches paper's Level-2 count)

Distractor attribute (non-causal, see DEVIATIONS_LOG.md #6):
  size in {small, large} — deterministically assigned per combo via a hash,
  but never used by oracle_label(). Exists purely to give the Generator a
  spurious feature it could (wrongly) latch onto.

Ground-truth oracle rule (synthetic, since the paper doesn't specify one —
see DEVIATIONS_LOG.md #4):
  white background: ANOMALY iff shape == circle AND color == red
  black background: ANOMALY iff NOT (shape == circle AND color == red)
  (a clean rule "flip" across backgrounds, matching the paper's qualitative
  description of context-dependent rules)

Level 3 masking: 4 of the 16 (combo, background) pairs are withheld from
the training/episodic stream (never shown to the memory consolidator) but
are included in the test set, forcing generalization from verified local
reflections about neighboring, non-masked combos.

IMPORTANT (see DEVIATIONS_LOG.md #6): the Generator's view of an episode
(`generator_view_text`) does NOT include the ground-truth label — it must
infer ANOMALY/NORMAL itself from the attributes and justify it. The
Verifier's view (`episode_text`) DOES include the ground-truth label, since
its role is to strictly check the Generator's claim against the real
outcome. This is what gives K=1 (one blind guess, no correction) genuine
room to differ from K=3 (guess, get told the true label via critique,
correct) — with the previous design the Generator was directly handed the
label to restate, so it essentially never failed.
"""
import itertools
import random

SHAPES = ["circle", "square"]
COLORS = ["red", "blue"]
TEXTURES = ["solid", "striped"]
BACKGROUNDS = ["white", "black"]


def oracle_label(shape: str, color: str, texture: str, background: str) -> str:
    is_circle_red = (shape == "circle" and color == "red")
    if background == "white":
        anomaly = is_circle_red
    else:  # black — rule flips
        anomaly = not is_circle_red
    return "ANOMALY" if anomaly else "NORMAL"


def _distractor_size(shape: str, color: str, texture: str, background: str) -> str:
    """
    Deterministic but causally irrelevant to the label — a spurious feature.
    Uses random.Random(seed_string), NOT Python's builtin hash(), because
    hash() on strings is salted per-process (PYTHONHASHSEED) and gives a
    DIFFERENT result every time the interpreter restarts — this was a real
    bug: it silently changed which `size` value each combo got between the
    original experiment run and a later diagnostic run in a fresh process,
    making their results non-comparable. random.Random with a string seed
    is documented as deterministic regardless of PYTHONHASHSEED.
    """
    seed_key = f"{shape}|{color}|{texture}|{background}|size_distractor_salt"
    return "large" if random.Random(seed_key).random() < 0.5 else "small"


def all_combinations():
    """All 16 (shape, color, texture, background) tuples with ground-truth labels."""
    combos = []
    for shape, color, texture, background in itertools.product(
        SHAPES, COLORS, TEXTURES, BACKGROUNDS
    ):
        label = oracle_label(shape, color, texture, background)
        size = _distractor_size(shape, color, texture, background)
        combos.append({
            "shape": shape, "color": color, "texture": texture,
            "background": background, "size": size, "label": label,
        })
    return combos


def level1_combinations():
    """
    All 8 (shape, color, texture) tuples on a UNIFORM white background —
    matches the paper's Level 1 (Section IV-A: "uniform background ...
    with 8 unique object combinations"). Straightforward, non-flipping
    rule: ANOMALY iff shape=circle AND color=red (the white-background
    half of the Level 2/3 oracle rule; there is no background-dependent
    flip to worry about at Level 1).
    """
    combos = []
    for shape, color, texture in itertools.product(SHAPES, COLORS, TEXTURES):
        label = oracle_label(shape, color, texture, "white")
        size = _distractor_size(shape, color, texture, "white")
        combos.append({
            "shape": shape, "color": color, "texture": texture,
            "background": "white", "size": size, "label": label,
        })
    return combos


def make_level3_split(mask_count: int = 4, seed: int = 42):
    """
    Returns (train_combos, test_combos).
    train_combos: 12 combos usable to build episodes for memory consolidation.
    test_combos: all 16 combos (the 4 masked ones + 12 seen ones), used for
                 inference evaluation — matches the paper's Level 3 design
                 where masked combos must still be handled at test time.
    """
    combos = all_combinations()
    rng = random.Random(seed)
    shuffled = combos[:]
    rng.shuffle(shuffled)
    masked = shuffled[:mask_count]
    train = shuffled[mask_count:]
    masked_keys = {(c["shape"], c["color"], c["texture"], c["background"]) for c in masked}
    test = combos  # all 16, in canonical order
    for c in test:
        key = (c["shape"], c["color"], c["texture"], c["background"])
        c["masked_at_train"] = key in masked_keys
    return train, test


def episode_text(combo: dict) -> str:
    """
    Full raw episode text INCLUDING ground-truth label — given to the
    Verifier (and stored in the memory bank for downstream hybrid
    inference). NOT given to the Generator — see generator_view_text().
    """
    return (
        f"Observed scene: background={combo['background']}, "
        f"shape={combo['shape']}, color={combo['color']}, "
        f"texture={combo['texture']}, size={combo['size']}. "
        f"Ground-truth outcome label: {combo['label']}."
    )


def generator_view_text(combo: dict) -> str:
    """
    Redacted view given to the Generator at r^(0): attributes only, no
    label. The Generator must infer whether the scene is ANOMALY or
    NORMAL and justify it — genuine risk of being wrong, unlike the
    previous design where the label was handed to it directly.
    """
    return (
        f"Observed scene: background={combo['background']}, "
        f"shape={combo['shape']}, color={combo['color']}, "
        f"texture={combo['texture']}, size={combo['size']}. "
        f"Outcome label is unknown — infer whether this scene is "
        f"ANOMALY or NORMAL."
    )


def query_text(combo: dict) -> str:
    """Render a combo as a target task query at inference time (no label given)."""
    return (
        f"New scene: background={combo['background']}, "
        f"shape={combo['shape']}, color={combo['color']}, "
        f"texture={combo['texture']}, size={combo['size']}. "
        f"Is this ANOMALY or NORMAL?"
    )
