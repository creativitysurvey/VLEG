"""
Grounding-modality inference -- real-image version. Three modalities,
mirroring the symbolic pipeline's Experiment 2 (Hybrid, Abstract-Only,
Raw-Only), adapted to this pipeline's known constraints:

- Hybrid: ALL verified rules as text + ONLY the target image (see
  DEVIATIONS_LOG.md #25 for why past-example IMAGES are not attached --
  multi-image collapse below ~3-4 images).
- Abstract-Only: ALL verified rules as text, NO image at all -- tests
  whether visual grounding of the TARGET matters, or verbal rules alone
  suffice.
- Raw-Only: ALL past episodes shown as their RAW IMAGES (bare label,
  no distilled rule) + target image. This directly re-tests the
  multi-image collapse mechanism from #25 with a different payload
  (label-only, not rule-text) -- expected, per that prior finding, to
  also collapse once several past-episode images are attached
  alongside the target; reported either way, not assumed.
"""
import re
from core_visual.llm_client import chat
from core_visual.memory import MemoryBank
from core_visual.task_env import query_text

SYSTEM_PROMPT_HYBRID = (
    "You classify new scenes as ANOMALY or NORMAL. You are given several "
    "verified rules distilled from past observations (no images of "
    "those past scenes are shown to you, only the rules), followed by "
    "ONE target image to classify. Use the rules and your own visual "
    "inspection of the target image to decide. "
    "Reply with ONLY one word: ANOMALY or NORMAL."
)

SYSTEM_PROMPT_ABSTRACT_ONLY = (
    "You classify new scenes as ANOMALY or NORMAL. You are given "
    "several verified rules distilled from past observations. NO IMAGE "
    "of the new scene is available to you -- you must decide based "
    "ENTIRELY on how the rules generalize, with no visual inspection "
    "possible. Reply with ONLY one word: ANOMALY or NORMAL."
)

SYSTEM_PROMPT_RAW_ONLY = (
    "You classify new scenes as ANOMALY or NORMAL. You are given "
    "several past example images IN ORDER, each labeled with its true "
    "outcome (no distilled rule, just the raw label), and FINALLY a "
    "new target image to classify (always the LAST image shown). "
    "Compare the target visually against the labeled examples to "
    "decide. Reply with ONLY one word: ANOMALY or NORMAL."
)


def predict_hybrid(bank: MemoryBank, target_combo: dict, k: int = None, debias: bool = False) -> str:
    """`k` is accepted for call-signature compatibility with the symbolic
    version but ignored -- ALL verified rules in the bank are used as
    text context. Set debias=True to enable the cross-episode
    attribute-skew warning (see core_visual/debias.py) -- NOT
    recommended based on DEVIATIONS_LOG.md #30's finding that it can
    worsen accuracy; kept as an option for completeness/comparison only."""
    tuples = bank.tuples
    parts = [f"Verified rule {i}: {t.verified_reflection}" for i, t in enumerate(tuples, start=1)]

    system_prompt = SYSTEM_PROMPT_HYBRID
    if debias:
        from core_visual.debias import detect_attribute_skew, format_skew_warning
        skew = detect_attribute_skew(bank)
        warning = format_skew_warning(skew)
        if warning:
            system_prompt = SYSTEM_PROMPT_HYBRID + " " + warning

    parts.append(f"Target scene (image attached). {query_text(target_combo)}")
    prompt = "\n\n".join(parts)
    reply = chat(
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": prompt}],
        temperature=0.0, images=[target_combo["image_path"]],  # ONLY the target image
    )
    match = re.search(r"ANOMALY|NORMAL", reply.upper())
    return match.group(0) if match else "NORMAL"


def predict_abstract_only(bank: MemoryBank, target_combo: dict, k: int = None) -> str:
    """All verified rules as text, NO image attached at all."""
    tuples = bank.tuples
    parts = [f"Verified rule {i}: {t.verified_reflection}" for i, t in enumerate(tuples, start=1)]
    parts.append(
        "Target scene (no image available -- classify based on the rules alone). "
        + query_text(target_combo)
    )
    prompt = "\n\n".join(parts)
    reply = chat(
        [{"role": "system", "content": SYSTEM_PROMPT_ABSTRACT_ONLY},
         {"role": "user", "content": prompt}],
        temperature=0.0,  # no images kwarg at all
    )
    match = re.search(r"ANOMALY|NORMAL", reply.upper())
    return match.group(0) if match else "NORMAL"


def predict_raw_only(bank: MemoryBank, target_combo: dict, k: int = None) -> str:
    """All past episodes as raw IMAGES (bare label, no distilled rule)
    + target image. Expected to re-exercise the multi-image collapse
    mechanism from DEVIATIONS_LOG.md #25 -- reported as observed, not
    assumed."""
    tuples = bank.tuples
    image_paths = []
    parts = []
    for i, t in enumerate(tuples, start=1):
        image_paths.append(t.image_path)
        parts.append(f"[Image {i}] Past example, true label: {t.combo['label']}")
    target_idx = len(tuples) + 1
    image_paths.append(target_combo["image_path"])
    parts.append(f"[Image {target_idx}] TARGET scene to classify (this is the last image). "
                 f"{query_text(target_combo)}")
    prompt = "\n\n".join(parts)
    reply = chat(
        [{"role": "system", "content": SYSTEM_PROMPT_RAW_ONLY},
         {"role": "user", "content": prompt}],
        temperature=0.0, images=image_paths,  # potentially MANY images -- collapse risk
    )
    match = re.search(r"ANOMALY|NORMAL", reply.upper())
    return match.group(0) if match else "NORMAL"


def predict(bank: MemoryBank, target_combo: dict, k: int = None, debias: bool = False) -> str:
    return predict_hybrid(bank, target_combo, k=k, debias=debias)
