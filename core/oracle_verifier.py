"""
Independent Oracle Verifier for Experiment 5 (paper Section IV-F / Table I).

Distinct from core/verifier.py (the Strict Verifier V_phi used inside the
VLEG consolidation loop, which only checks a reflection against the ONE
episode it was generated from). The paper's Experiment 5 explicitly uses
a SEPARATE "independent, high-capacity oracle verifier (prompted with
ground-truth environmental rules) to audit the final reflections stored
in the memory bank" — i.e. it has access to the TRUE global rule, not
just one instance, and can therefore catch over-specified / spurious
reflections that V_phi structurally cannot (see DEVIATIONS_LOG.md #6-#9
for why V_phi alone can't detect this class of error).
"""
import json
import re
from core.llm_client import chat

TRUE_RULE_DESCRIPTION = (
    "The TRUE rule governing ANOMALY/NORMAL labels in this environment is: "
    "on a WHITE background, a scene is ANOMALY if and only if shape=circle "
    "AND color=red (otherwise NORMAL). On a BLACK background, the rule is "
    "inverted: a scene is ANOMALY if and only if it is NOT the case that "
    "shape=circle AND color=red (otherwise NORMAL). "
    "The attributes 'texture' and 'size' have NO effect on the label "
    "whatsoever — they are pure noise."
)

SYSTEM_PROMPT = (
    "You are an independent oracle auditor with access to the TRUE "
    "ground-truth rule of an environment (given below). Your job is to "
    "audit a candidate heuristic rule that some other agent proposed, "
    "and determine whether it HALLUCINATES.\n\n"
    f"{TRUE_RULE_DESCRIPTION}\n\n"
    "A candidate rule HALLUCINATES if ANY of the following hold: "
    "(a) it claims 'texture' or 'size' is part of what determines the "
    "label (even if it happens to reach the correct label for one "
    "specific instance by coincidence); "
    "(b) it states the shape/color/background condition incorrectly "
    "(wrong attribute, wrong background behavior, or reversed logic); "
    "(c) it is too vague to be checked against the true rule at all. "
    "A candidate rule does NOT hallucinate if its stated condition is a "
    "correct, logically consistent instance or subset of the true rule "
    "above (mentioning background + shape + color correctly), even if "
    "it doesn't state the fully general rule. "
    "Respond with ONLY a JSON object, no other text: "
    '{"hallucinated": true or false, "reason": "one sentence"}'
)


def _parse_json_response(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    candidate = match.group(0) if match else text
    try:
        obj = json.loads(candidate)
        assert "hallucinated" in obj
        return {"hallucinated": bool(obj["hallucinated"]), "reason": str(obj.get("reason", ""))}
    except Exception:
        return {"hallucinated": True, "reason": f"[oracle output unparseable: {text[:200]}]"}


def audit_reflection(reflection: str) -> tuple[bool, str]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Candidate rule to audit: \"{reflection}\""},
    ]
    reply = chat(messages, temperature=0.0)
    parsed = _parse_json_response(reply)
    return parsed["hallucinated"], parsed["reason"]


# --- Detailed (categorized) audit, added after the first Experiment 5 run
# showed both strategies saturating near ~92-94% hallucination under the
# binary criterion. This splits "mentions an irrelevant attribute but the
# core causal condition is right" from "the core causal condition itself
# is wrong" — two very different severities that the binary version
# conflated. See DEVIATIONS_LOG.md #11.

SYSTEM_PROMPT_DETAILED = (
    "You are an independent oracle auditor with access to the TRUE "
    "ground-truth rule of an environment (given below). Classify a "
    "candidate heuristic rule into EXACTLY ONE category.\n\n"
    f"{TRUE_RULE_DESCRIPTION}\n\n"
    "Categories:\n"
    "- \"correct\": the rule's background+shape+color condition exactly "
    "matches the true rule, AND it does not treat texture or size as "
    "required/determinative.\n"
    "- \"over_specified\": the rule's background+shape+color condition "
    "is CORRECT (matches the true rule), but it ALSO includes texture "
    "and/or size as if they were required conditions — harmless for "
    "this one instance, but would wrongly fail to fire on other scenes "
    "that share the correct shape/color/background but differ in "
    "texture/size.\n"
    "- \"wrong_condition\": the background+shape+color logic itself is "
    "incorrect (wrong shape, wrong color, wrong background behavior, or "
    "reversed logic) — a genuine logical error, regardless of whether "
    "texture/size are also mentioned.\n"
    "- \"vague\": too vague or unfalsifiable to classify against the "
    "true rule at all.\n"
    "Respond with ONLY a JSON object, no other text: "
    '{"category": "correct" | "over_specified" | "wrong_condition" | "vague", "reason": "one sentence"}'
)


def _parse_category_response(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    candidate = match.group(0) if match else text
    valid_categories = {"correct", "over_specified", "wrong_condition", "vague"}
    try:
        obj = json.loads(candidate)
        cat = str(obj.get("category", "")).strip()
        if cat not in valid_categories:
            raise ValueError(f"unexpected category {cat!r}")
        return {"category": cat, "reason": str(obj.get("reason", ""))}
    except Exception:
        return {"category": "wrong_condition",
                "reason": f"[oracle output unparseable, defaulting to worst case: {text[:200]}]"}


def audit_reflection_detailed(reflection: str) -> tuple[str, str]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_DETAILED},
        {"role": "user", "content": f"Candidate rule to classify: \"{reflection}\""},
    ]
    reply = chat(messages, temperature=0.0)
    parsed = _parse_category_response(reply)
    return parsed["category"], parsed["reason"]


# --- Ground-truth-aware detailed audit, added to disentangle two
# previously-conflated axes: (1) did this reflection get the actual
# ANOMALY/NORMAL call right for the specific episode it came from, vs
# (2) independent of that, does its STATED justification correctly
# identify the causal attributes without over-specifying on texture/size.
# The earlier text-only audit couldn't separate these because it never
# saw which instance a given reflection was generated from. See
# DEVIATIONS_LOG.md #12.

SYSTEM_PROMPT_GROUND_TRUTH = (
    "You are an independent oracle auditor with access to the TRUE "
    "ground-truth rule of an environment, AND the specific scene a "
    "candidate rule was generated to explain.\n\n"
    f"{TRUE_RULE_DESCRIPTION}\n\n"
    "You will be given: the specific scene's true attributes and true "
    "label, and a candidate rule some agent proposed to explain it. "
    "Answer TWO independent questions:\n\n"
    "1. label_correct: Does the candidate rule's stated conclusion "
    "(ANOMALY or NORMAL) for THIS scene match the TRUE label given? "
    "(true/false)\n\n"
    "2. condition_quality: Independently of whether the label is "
    "correct, classify the CAUSAL JUSTIFICATION the rule gives:\n"
    "   - \"clean_correct\": correctly identifies background+shape+color "
    "as jointly determining the outcome (consistent with the true rule "
    "as a valid general instance), and does NOT treat texture or size "
    "as required.\n"
    "   - \"over_specified\": the background+shape+color portion is "
    "consistent with the true rule, but the rule ALSO wrongly treats "
    "texture and/or size as required/determinative.\n"
    "   - \"causally_wrong\": the rule attributes the outcome to the "
    "wrong factor — e.g. treats color or texture or size as the real "
    "determinant when shape/background is what actually matters for "
    "this scene, or gets the shape/background logic backwards. (Note: "
    "on a BLACK background, ANY square is ANOMALY regardless of color/"
    "texture/size, and ANY non-red circle is also ANOMALY — only a "
    "red circle is NORMAL on black. A rule that credits color or "
    "texture as the reason a square-on-black scene is ANOMALY is "
    "causally_wrong even though the label may be correct.)\n"
    "   - \"vague\": too vague to classify.\n\n"
    "Respond with ONLY a JSON object, no other text: "
    '{"label_correct": true or false, "condition_quality": '
    '"clean_correct" | "over_specified" | "causally_wrong" | "vague", '
    '"reason": "one sentence"}'
)


def _parse_ground_truth_response(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    candidate = match.group(0) if match else text
    valid_categories = {"clean_correct", "over_specified", "causally_wrong", "vague"}
    try:
        obj = json.loads(candidate)
        cat = str(obj.get("condition_quality", "")).strip()
        if cat not in valid_categories:
            raise ValueError(f"unexpected category {cat!r}")
        return {
            "label_correct": bool(obj["label_correct"]),
            "condition_quality": cat,
            "reason": str(obj.get("reason", "")),
        }
    except Exception:
        return {"label_correct": False, "condition_quality": "causally_wrong",
                "reason": f"[oracle output unparseable, defaulting to worst case: {text[:200]}]"}


def audit_with_ground_truth(combo: dict, reflection: str) -> dict:
    scene_desc = (
        f"background={combo['background']}, shape={combo['shape']}, "
        f"color={combo['color']}, texture={combo['texture']}, "
        f"size={combo['size']}. TRUE label: {combo['label']}."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_GROUND_TRUTH},
        {"role": "user", "content": (
            f"Scene: {scene_desc}\n"
            f"Candidate rule: \"{reflection}\"\n"
            f"Audit it."
        )},
    ]
    reply = chat(messages, temperature=0.0)
    return _parse_ground_truth_response(reply)
