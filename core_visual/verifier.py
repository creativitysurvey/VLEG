"""
Strict Verifier V_phi -- real-image version.

Sees the SAME image as the generator, PLUS the ground-truth label (via
core_visual.task_env.verifier_view_text). Checks two things a symbolic
verifier could never check: (a) does the reflection's claimed label
match the ground truth, same as before; and (b) does the reflection's
claimed VISUAL attributes (shape/color/material) actually match what is
visible in the image -- a new class of error (hallucinated visual
attributes) the symbolic version could not test at all.
"""
import json
import re
from core_visual.llm_client import chat

SYSTEM_PROMPT = (
    "You are a STRICT visual verifier. You are given an image of a "
    "single 3D object, its ground-truth outcome label, and a candidate "
    "heuristic rule that claims to explain that label. Reject the rule "
    "if ANY of the following hold: "
    "(a) the rule's stated condition, applied to what is ACTUALLY "
    "visible in the image, does not match the given ground-truth label; "
    "(b) the rule describes a visual attribute that does NOT match what "
    "is actually shown in the image (e.g. calls a cube a sphere, calls "
    "a matte/rubber-looking surface metallic, or vice versa -- look "
    "carefully at shape and surface finish); "
    "(c) the rule is too vague to be checked as true or false. "
    "Otherwise accept it. "
    "Respond with ONLY a JSON object, no other text: "
    '{"valid": true or false, "critique": "one sentence, empty string if valid"}'
)


def _parse_json_response(text: str) -> dict:
    from core.verification_response import parse_verification_response
    return parse_verification_response(text)


def verify_reflection(image_path: str, verifier_view_text: str, reflection: str) -> tuple:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"{verifier_view_text}\n"
            f"Candidate rule: \"{reflection}\"\n"
            f"Verify it against the image."
        )},
    ]
    reply = chat(messages, temperature=0.0, images=[image_path])
    parsed = _parse_json_response(reply)
    return parsed["valid"], parsed["critique"]
