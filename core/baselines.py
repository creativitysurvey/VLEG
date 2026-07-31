"""
Baseline methods for Experiment 1 (SOTA Comparative Analysis, paper
Section IV-B): ICL, Reflexion, and HDR. VLEG itself is already
implemented in core/refinement.py + core/inference.py ("hybrid" modality).

Methodological choices made here (see DEVIATIONS_LOG.md #16 for the
original writeup, #32 for the Reflexion fairness fix below):
  - ICL = Pure Episodic Retention, no abstraction at all. Zero
    Generator/Verifier calls during "consolidation" (there is none) —
    raw episodes are stored directly and retrieved via raw-episode
    embedding similarity, then shown to the reasoning LLM with no
    distilled rule (core.inference's raw_only prompt/system message).
  - Reflexion = Localized Episodic Reflection with BINARY environmental
    feedback across up to K attempts per episode, always committing
    SOME reflection at the end (never discards, unlike VLEG). This
    replaces an earlier single-shot, always-commit-on-first-try
    implementation that gave Reflexion no chance to self-correct at
    all — an unfair comparison against VLEG's K-bounded refinement loop
    (see DEVIATIONS_LOG.md #32). The feedback given between attempts is
    a plain right/wrong signal plus the true label (Reflexion's own
    "trial-and-error with environmental feedback" mechanism), NOT
    VLEG's richer verifier critique — the two methods' feedback
    richness is a real, reported difference, not something to paper
    over by reusing VLEG's Generator/Verifier machinery for Reflexion.
  - HDR = Global Semantic Consolidation + Iterative Cyclic Refinement +
    Hybrid Grounding. ONE hypothesis is induced from the ENTIRE labeled
    training set at once (not per-episode), verified against the whole
    set, refined up to K times. At inference, the single hypothesis is
    paired with top-k retrieved raw episodes (Hybrid Grounding).
"""
import json
import re
from dataclasses import dataclass, field

from core.llm_client import chat
from core.task_env import episode_text, generator_view_text, query_text
from core.memory import MemoryBank
from core.retrieval import retrieve_top_k, retrieve_top_k_by_episode
from core.inference import (
    build_raw_only_prompt, build_abstract_only_prompt,
    SYSTEM_PROMPT_RAW_ONLY, SYSTEM_PROMPT_ABSTRACT_ONLY, SYSTEM_PROMPT_HYBRID,
)

# ============================== ICL ==============================
# (unchanged)

def build_icl_bank(combos: list) -> MemoryBank:
    """No abstraction, no LLM calls — just wraps raw episodes for retrieval."""
    bank = MemoryBank()
    for combo in combos:
        bank.add(combo, episode_text(combo), verified_reflection="")  # unused for ICL
    return bank


def icl_predict(bank: MemoryBank, target_combo: dict, k: int = 3) -> str:
    top_tuples = retrieve_top_k_by_episode(bank, target_combo, k=k)
    if not top_tuples:
        return "NORMAL"
    prompt = build_raw_only_prompt(top_tuples, target_combo)
    reply = chat([{"role": "system", "content": SYSTEM_PROMPT_RAW_ONLY},
                  {"role": "user", "content": prompt}], temperature=0.0)
    match = re.search(r"ANOMALY|NORMAL", reply.upper())
    return match.group(0) if match else "NORMAL"


# ============================== Reflexion ==============================
# Deliberately NOT reusing core.generator.generate_reflection: that
# function's refinement-turn wording is hardcoded to VLEG's strict-
# verifier framing ("A verifier checked it... rejected it with this
# critique..."), which would be methodologically wrong to put in
# Reflexion's mouth — Reflexion has no verifier, only binary outcome
# feedback. core/generator.py is left completely untouched.

REFLEXION_SYSTEM_PROMPT = (
    "You are a careful pattern-analysis assistant attempting a task "
    "through trial and error. You are shown one observed scene's "
    "attributes (background, shape, color, texture, size). The scene's "
    "true outcome label (ANOMALY or NORMAL) is NOT given to you — you "
    "must infer it yourself, using whatever pattern seems most "
    "plausible, and propose a SHORT, CONCRETE, testable heuristic rule "
    "(one or two sentences) stating which attribute condition you "
    "believe determines ANOMALY vs NORMAL, and what this particular "
    "scene's label is under that rule. Use ONLY the attributes given. "
    "State it as a condition, e.g. 'On a white background, circle+red "
    "scenes are ANOMALY; this scene is therefore ANOMALY.' "
    "Reply with ONLY the rule sentence, nothing else."
)


def _reflexion_attempt_episode(gtext: str, true_label: str, K: int) -> tuple:
    """
    Up to K attempts, each given plain right/wrong environmental
    feedback (not a verifier critique) about the PREVIOUS attempt.
    Stops early and returns as soon as an attempt's predicted label
    matches the truth. If all K attempts are wrong, COMMITS the last
    attempt anyway — Reflexion never discards an episode, unlike VLEG's
    strict verifier gate.
    Returns (final_reflection, attempts_used, succeeded_within_budget).
    """
    prev_reflection = None
    for j in range(K):
        if prev_reflection is None:
            messages = [{"role": "system", "content": REFLEXION_SYSTEM_PROMPT},
                        {"role": "user", "content": gtext}]
        else:
            messages = [
                {"role": "system", "content": REFLEXION_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"{gtext}\n\n"
                    f"Your previous attempt proposed: \"{prev_reflection}\"\n"
                    f"That was WRONG — the correct outcome for this scene was "
                    f"actually {true_label}. Try again: propose a different rule "
                    f"that would correctly explain this outcome."
                )},
            ]
        reply = chat(messages, temperature=0.0).strip().strip('"')
        match = re.search(r"ANOMALY|NORMAL", reply.upper())
        predicted_label = match.group(0) if match else None
        if predicted_label == true_label:
            return reply, j + 1, True
        prev_reflection = reply
    return prev_reflection, K, False


def build_reflexion_bank(combos: list, K: int = 3, log_prefix: str = "reflexion") -> MemoryBank:
    """K-bounded trial-and-error with binary environmental feedback,
    always commits (matches Reflexion's own mechanism) — see
    DEVIATIONS_LOG.md #32 for why this replaced a single-shot,
    always-commit-on-first-try version that gave Reflexion no
    self-correction opportunity at all."""
    bank = MemoryBank()
    for idx, combo in enumerate(combos):
        gtext = generator_view_text(combo)
        reflection, attempts_used, succeeded = _reflexion_attempt_episode(gtext, combo["label"], K)
        bank.add(combo, episode_text(combo), reflection)
        status = "SUCCEEDED" if succeeded else "COMMITTED (unresolved after K attempts)"
        print(f"  [{log_prefix}] episode {idx+1}/{len(combos)}: {status} "
              f"after {attempts_used} attempt(s) -> \"{reflection}\"")
    return bank


def reflexion_predict(bank: MemoryBank, target_combo: dict, k: int = 3) -> str:
    top_tuples = retrieve_top_k(bank, target_combo, k=k)  # reflection-embedding retrieval
    if not top_tuples:
        return "NORMAL"
    prompt = build_abstract_only_prompt(top_tuples, target_combo)
    reply = chat([{"role": "system", "content": SYSTEM_PROMPT_ABSTRACT_ONLY},
                  {"role": "user", "content": prompt}], temperature=0.0)
    match = re.search(r"ANOMALY|NORMAL", reply.upper())
    return match.group(0) if match else "NORMAL"


# ============================== HDR ==============================
# (unchanged)

GLOBAL_GENERATOR_SYSTEM = (
    "You are shown a full labeled dataset of observed scenes, each with "
    "its ANOMALY/NORMAL outcome. Propose ONE general hypothesis (2-4 "
    "sentences) that explains the pattern across the ENTIRE dataset. "
    "If the rule differs by background color, state both cases "
    "explicitly. Use only the attributes given (background, shape, "
    "color, texture, size) — note that not all attributes need be "
    "relevant. Reply with ONLY the hypothesis, nothing else."
)

GLOBAL_VERIFIER_SYSTEM = (
    "You are a strict verifier. You are given a proposed global "
    "hypothesis and a full labeled dataset. For EACH scene, mentally "
    "apply the hypothesis and check whether it predicts that scene's "
    "TRUE label correctly. The hypothesis is VALID only if it correctly "
    "predicts EVERY scene in the dataset with no exceptions. "
    "If any scene is mispredicted, it is INVALID — list which scenes "
    "(by their attributes) it gets wrong in your critique. "
    "Respond with ONLY a JSON object, no other text: "
    '{"valid": true or false, "critique": "list the specific scenes it '
    'gets wrong, or empty string if valid"}'
)


def _parse_global_verifier(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    candidate = match.group(0) if match else text
    try:
        obj = json.loads(candidate)
        return {"valid": bool(obj["valid"]), "critique": str(obj.get("critique", ""))}
    except Exception:
        return {"valid": False, "critique": f"[unparseable verifier output: {text[:200]}]"}


def _all_episodes_text(combos: list) -> str:
    return "\n".join(f"{i+1}. {episode_text(c)}" for i, c in enumerate(combos))


@dataclass
class HDRResult:
    verified: bool
    hypothesis: str
    iterations_used: int
    history: list = field(default_factory=list)


def hdr_generate_global_hypothesis(train_combos: list, K: int = 3,
                                    log_prefix: str = "HDR") -> HDRResult:
    dataset_text = _all_episodes_text(train_combos)
    prev_hypothesis, critique = None, None
    history = []
    for j in range(K):
        if prev_hypothesis is None:
            messages = [{"role": "system", "content": GLOBAL_GENERATOR_SYSTEM},
                        {"role": "user", "content": dataset_text}]
        else:
            messages = [
                {"role": "system", "content": GLOBAL_GENERATOR_SYSTEM},
                {"role": "user", "content": (
                    f"{dataset_text}\n\nYour previous hypothesis was: "
                    f"\"{prev_hypothesis}\"\nA verifier rejected it: "
                    f"\"{critique}\"\nPropose a revised hypothesis."
                )},
            ]
        hypothesis = chat(messages, temperature=0.0).strip().strip('"')

        verify_messages = [
            {"role": "system", "content": GLOBAL_VERIFIER_SYSTEM},
            {"role": "user", "content": f"Hypothesis: \"{hypothesis}\"\n\nDataset:\n{dataset_text}"},
        ]
        verdict = _parse_global_verifier(chat(verify_messages, temperature=0.0))
        history.append({"hypothesis": hypothesis, "valid": verdict["valid"], "critique": verdict["critique"]})
        print(f"  [{log_prefix}] iter {j+1}/{K}: {'VALID' if verdict['valid'] else 'INVALID'} "
              f"-> \"{hypothesis[:100]}\"")
        if verdict["valid"]:
            return HDRResult(verified=True, hypothesis=hypothesis, iterations_used=j + 1, history=history)
        prev_hypothesis, critique = hypothesis, verdict["critique"]

    # Exhausted K iterations without a verified hypothesis — fall back to
    # the last proposed one anyway (HDR must produce SOME global rule to
    # be usable at inference; unlike VLEG's per-episode gate, there's no
    # sensible "discard the whole dataset" option here).
    return HDRResult(verified=False, hypothesis=prev_hypothesis, iterations_used=K, history=history)


def hdr_predict(global_hypothesis: str, raw_bank: MemoryBank, target_combo: dict, k: int = 3) -> str:
    top_tuples = retrieve_top_k_by_episode(raw_bank, target_combo, k=k)
    parts = [f"Example — {t.episode_text}" for t in top_tuples]
    parts.append(f"Global rule: {global_hypothesis}")
    parts.append(f"Target: {query_text(target_combo)}")
    prompt = "\n\n".join(parts)
    reply = chat([{"role": "system", "content": SYSTEM_PROMPT_HYBRID},
                  {"role": "user", "content": prompt}], temperature=0.0)
    match = re.search(r"ANOMALY|NORMAL", reply.upper())
    return match.group(0) if match else "NORMAL"
