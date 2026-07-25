"""
Strict Verifier V_phi — implements Eq. III.5 / III.8.

(v^(j), c^(j)) = V_phi( e_i, r_i^(j) )

Independent critic: checks the candidate reflection strictly against the
raw episode. Outputs a binary validity flag + a natural-language critique.
"""
import json
import re
from core.llm_client import chat

SYSTEM_PROMPT = (
    "You are a STRICT logical verifier. You are given (1) an observed "
    "scene with its ground-truth label, and (2) a candidate heuristic rule "
    "that claims to explain the label. "
    "Reject the rule if ANY of the following hold: "
    "(a) the condition in the rule, when applied to this scene's actual "
    "attribute values, does NOT match this scene's actual label; "
    "(b) the rule references an attribute or value not present in the "
    "scene description; "
    "(c) the rule is too vague to be checked as true or false (e.g. "
    "'unusual scenes are anomalies' with no concrete attribute condition). "
    "Otherwise accept it. "
    "Respond with ONLY a JSON object, no other text: "
    '{"valid": true or false, "critique": "one sentence, empty string if valid"}'
)


def _parse_json_response(text: str) -> dict:
    # models sometimes wrap JSON in ```json fences or add stray text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    candidate = match.group(0) if match else text
    try:
        obj = json.loads(candidate)
        assert "valid" in obj
        return {"valid": bool(obj["valid"]), "critique": str(obj.get("critique", ""))}
    except Exception:
        # Fail safe: treat unparseable verifier output as invalid, with the
        # raw text as the critique, so the loop can still make progress
        # instead of crashing.
        return {"valid": False, "critique": f"[verifier output unparseable: {text[:200]}]"}


def verify_reflection(episode_text: str, reflection: str) -> tuple[bool, str]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Scene: {episode_text}\n"
            f"Candidate rule: \"{reflection}\"\n"
            f"Verify it."
        )},
    ]
    reply = chat(messages, temperature=0.0)
    parsed = _parse_json_response(reply)
    return parsed["valid"], parsed["critique"]
