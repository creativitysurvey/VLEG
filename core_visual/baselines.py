"""
Baseline methods for the real-image Experiment 1 (SOTA Comparative
Analysis): ICL, Reflexion, HDR. VLEG itself is core_visual/refinement.py
+ core_visual/inference.py's predict_hybrid.

Each baseline hits a real, previously-diagnosed constraint of this
pipeline (multi-image collapse, #25; Abstract-Only's task-design
degeneracy, #39) that the symbolic pipeline never had to contend with.
Rather than silently redesigning each baseline until it happens to
avoid the problem, we implement each as close to its symbolic-pipeline
definition as this environment allows, and report honestly what
happens -- including if that means a baseline predictably fails for a
structural, non-arbitrary reason.

  - ICL = Pure Episodic Retention. No abstraction, no LLM calls to
    build its bank (raw images stored directly). At inference, ALL
    training images + target are shown together with bare labels, no
    distilled rule -- structurally identical to Experiment 2's
    Raw-Only mechanism (#39), since this pipeline's design (no usable
    image-embedding retrieval, attribute info living only in images)
    makes "raw episodic grounding with retrieval" and "raw episodic
    grounding without retrieval" collapse to the same thing once
    retrieval itself isn't feasible. Expected, per #25/#39, to exercise
    the multi-image collapse mechanism.
  - Reflexion = Localized Episodic Reflection, K-bounded trial-and-error
    with binary environmental feedback (mirrors the symbolic pipeline's
    fairness-fixed version, DEVIATIONS_LOG.md #32), always commits.
    DEVIATION from the symbolic pipeline's inference mapping: uses
    Hybrid-style grounding (rules as text + the target IMAGE) rather
    than pure Abstract-Only, because Abstract-Only was found (#39) to
    be a degenerate, uninformative-by-design condition in this
    pipeline (the query text carries zero identifying information
    without an image) -- using it for Reflexion would confound "is
    Reflexion a weak method" with "is this environment's Abstract-Only
    mode broken by construction," which would not be an informative
    test of either.
  - HDR = Global Semantic Consolidation. ONE hypothesis induced from
    the ENTIRE labeled training set AT ONCE, meaning all training
    images attached to a single call -- HDR's core design assumption
    (see the whole dataset simultaneously) is structurally incompatible
    with this pipeline's per-message image-count constraint in a way
    VLEG's per-episode design is not (VLEG never needs more than one
    training image per consolidation call). Implemented as close to
    its symbolic definition as possible; whether/how it fails is
    reported as a finding, not designed around.
"""
import json
import re

from core_visual.llm_client import chat
from core_visual.memory import MemoryBank
from core_visual.task_env import generator_view_text, verifier_view_text, query_text
from core_visual.inference import predict_hybrid, SYSTEM_PROMPT_RAW_ONLY

# ============================== ICL ==============================

def build_icl_bank(combos: list) -> MemoryBank:
    """No abstraction, no LLM calls -- wraps raw images for direct grounding."""
    bank = MemoryBank()
    for combo in combos:
        bank.add(combo, combo["image_path"], verified_reflection="")  # unused for ICL
    return bank


def icl_predict(bank: MemoryBank, target_combo: dict, k: int = None) -> str:
    """ALL bank images + target shown together with bare labels -- see
    module docstring for why this is structurally identical to
    Experiment 2's Raw-Only mechanism in this pipeline."""
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
    reply = chat([{"role": "system", "content": SYSTEM_PROMPT_RAW_ONLY},
                  {"role": "user", "content": prompt}], temperature=0.0, images=image_paths)
    match = re.search(r"\bANOMALY\b|\bNORMAL\b", reply.upper())
    return match.group(0) if match else "NORMAL"


# ============================== Reflexion ==============================

REFLEXION_SYSTEM_PROMPT = (
    "You are a careful visual pattern-analysis assistant attempting a "
    "task through trial and error. You are shown an image of a single "
    "3D object. The scene's true outcome label (ANOMALY or NORMAL) is "
    "NOT given to you -- you must infer it yourself, using whatever "
    "pattern seems most plausible, and propose a SHORT, CONCRETE, "
    "testable heuristic rule (one or two sentences) stating which "
    "visual attribute condition you believe determines ANOMALY vs "
    "NORMAL, and what this particular scene's label is under that "
    "rule. Describe only attributes you can actually see (shape, "
    "color, glossy/matte surface, apparent size). "
    "Reply with ONLY the rule sentence, nothing else."
)


def _reflexion_attempt_episode(image_path: str, gtext: str, true_label: str, K: int) -> tuple:
    prev_reflection = None
    for j in range(K):
        if prev_reflection is None:
            messages = [{"role": "system", "content": REFLEXION_SYSTEM_PROMPT},
                        {"role": "user", "content": gtext}]
        else:
            messages = [
                {"role": "system", "content": REFLEXION_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"{gtext}\n\nYour previous attempt proposed: \"{prev_reflection}\"\n"
                    f"That was WRONG -- the correct outcome for this scene was actually "
                    f"{true_label}. Try again: propose a different rule that would correctly "
                    f"explain this outcome. The image is attached again for reference."
                )},
            ]
        reply = chat(messages, temperature=0.0, images=[image_path]).strip().strip('"')
        match = re.search(r"\bANOMALY\b|\bNORMAL\b", reply.upper())
        predicted_label = match.group(0) if match else None
        if predicted_label == true_label:
            return reply, j + 1, True
        prev_reflection = reply
    return prev_reflection, K, False


def build_reflexion_bank(combos: list, K: int = 3, log_prefix: str = "reflexion") -> MemoryBank:
    bank = MemoryBank()
    for idx, combo in enumerate(combos):
        gtext = generator_view_text(combo)
        reflection, attempts_used, succeeded = _reflexion_attempt_episode(
            combo["image_path"], gtext, combo["label"], K)
        bank.add(combo, combo["image_path"], reflection)
        status = "SUCCEEDED" if succeeded else "COMMITTED (unresolved)"
        print(f"  [{log_prefix}] episode {idx+1}/{len(combos)} "
              f"({combo['shape']}/{combo['color']}/{combo['material']}/{combo['size']}, "
              f"true={combo['label']}): {status} after {attempts_used} attempt(s) -> \"{reflection}\"")
    return bank


def reflexion_predict(bank: MemoryBank, target_combo: dict, k: int = None) -> str:
    """Hybrid-style grounding (rules as text + target image) -- see
    module docstring for why NOT pure Abstract-Only."""
    return predict_hybrid(bank, target_combo, k=k)


# ============================== HDR ==============================

GLOBAL_GENERATOR_SYSTEM = (
    "You are shown several images IN ORDER, each a scene containing one "
    "3D object, each labeled with its true ANOMALY/NORMAL outcome. "
    "Propose ONE general hypothesis (2-4 sentences) that explains the "
    "pattern across ALL the images. If the rule differs by surface "
    "finish (glossy/metallic vs. matte/rubber), state both cases "
    "explicitly. Use only attributes you can see (shape, color, "
    "surface finish, apparent size) -- not all attributes need be "
    "relevant. Reply with ONLY the hypothesis, nothing else."
)

GLOBAL_VERIFIER_SYSTEM = (
    "You are a strict verifier. You are given a proposed global "
    "hypothesis and several labeled example images IN ORDER. For EACH "
    "image, mentally apply the hypothesis and check whether it predicts "
    "that scene's TRUE label correctly. The hypothesis is VALID only if "
    "it correctly predicts EVERY image with no exceptions. If any image "
    "is mispredicted, it is INVALID -- describe which image(s) (by "
    "position number) it gets wrong in your critique. "
    "Respond with ONLY a JSON object, no other text: "
    '{"valid": true or false, "critique": "describe which images it '
    'gets wrong, or empty string if valid"}'
)


def _parse_global_verifier(text: str) -> dict:
    from core.verification_response import parse_verification_response
    return parse_verification_response(text)


class HDRResult:
    def __init__(self, verified, hypothesis, iterations_used, history):
        self.verified = verified
        self.hypothesis = hypothesis
        self.iterations_used = iterations_used
        self.history = history


def hdr_generate_global_hypothesis(train_combos: list, K: int = 3, log_prefix: str = "HDR") -> "HDRResult":
    """Attaches ALL training images to a single call, matching HDR's
    symbolic-pipeline design (see whole dataset at once) as closely as
    this environment allows. Per the module docstring, this is expected
    to be at serious risk of the multi-image collapse mechanism (#25) --
    reported honestly, not designed around."""
    image_paths = [c["image_path"] for c in train_combos]
    labels_text = "\n".join(
        f"[Image {i+1}] true label: {c['label']}" for i, c in enumerate(train_combos)
    )

    prev_hypothesis, critique = None, None
    history = []
    for j in range(K):
        if prev_hypothesis is None:
            gen_prompt = labels_text
        else:
            gen_prompt = (
                f"{labels_text}\n\nYour previous hypothesis was: \"{prev_hypothesis}\"\n"
                f"A verifier rejected it: \"{critique}\"\nPropose a revised hypothesis."
            )
        hypothesis = chat(
            [{"role": "system", "content": GLOBAL_GENERATOR_SYSTEM},
             {"role": "user", "content": gen_prompt}],
            temperature=0.0, images=image_paths,
        ).strip().strip('"')

        verdict = _parse_global_verifier(chat(
            [{"role": "system", "content": GLOBAL_VERIFIER_SYSTEM},
             {"role": "user", "content": f"Hypothesis: \"{hypothesis}\"\n\n{labels_text}"}],
            temperature=0.0, images=image_paths,
        ))
        history.append({"hypothesis": hypothesis, "valid": verdict["valid"], "critique": verdict["critique"]})
        print(f"  [{log_prefix}] iter {j+1}/{K}: {'VALID' if verdict['valid'] else 'INVALID'} "
              f"-> \"{hypothesis[:100]}\"")
        if verdict["valid"]:
            return HDRResult(True, hypothesis, j + 1, history)
        prev_hypothesis, critique = hypothesis, verdict["critique"]

    return HDRResult(False, prev_hypothesis, K, history)


def hdr_predict(global_hypothesis: str, target_combo: dict) -> str:
    """Hypothesis text + target image (Hybrid-style, single image)."""
    prompt = f"Global rule: {global_hypothesis}\n\nTarget scene (image attached). {query_text(target_combo)}"
    reply = chat(
        [{"role": "system", "content": (
            "You classify new scenes as ANOMALY or NORMAL using a global rule "
            "and your own visual inspection of the target image. "
            "Reply with ONLY one word: ANOMALY or NORMAL.")},
         {"role": "user", "content": prompt}],
        temperature=0.0, images=[target_combo["image_path"]],
    )
    match = re.search(r"\bANOMALY\b|\bNORMAL\b", reply.upper())
    return match.group(0) if match else "NORMAL"
