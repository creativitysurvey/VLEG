"""
Reflection Generator G_theta — implements Eq. III.3 / III.4 / III.7 / III.9.

r_i^(0) ~ G_theta( . | e_i )
r_i^(j) ~ G_theta( . | e_i, r_i^(j-1), c^(j-1) )    for j > 0

NOTE (see DEVIATIONS_LOG.md #6): at j=0 the Generator only sees the
REDACTED view of the episode (attributes, no ground-truth label) — it
must infer ANOMALY/NORMAL itself, not just restate a given label. This
is what gives it genuine room to be wrong, which K=1 vs K=3 depends on
to show any difference.
"""
from core.llm_client import chat

SYSTEM_PROMPT = (
    "You are a careful pattern-analysis assistant. You are shown one "
    "observed scene's attributes (background, shape, color, texture, "
    "size). The scene's true outcome label (ANOMALY or NORMAL) is NOT "
    "given to you — you must infer it yourself, using whatever pattern "
    "seems most plausible, and propose a SHORT, CONCRETE, testable "
    "heuristic rule (one or two sentences) stating which attribute "
    "condition you believe determines ANOMALY vs NORMAL, and what this "
    "particular scene's label is under that rule. Use ONLY the "
    "attributes given. State it as a condition, e.g. "
    "'On a white background, circle+red scenes are ANOMALY; this scene "
    "is therefore ANOMALY.' "
    "Reply with ONLY the rule sentence, nothing else."
)


def generate_reflection(generator_view_text: str, prev_reflection: str = None,
                         critique: str = None) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if prev_reflection is None:
        messages.append({"role": "user", "content": generator_view_text})
    else:
        messages.append({
            "role": "user",
            "content": (
                f"{generator_view_text}\n\n"
                f"Your previous proposed rule was: \"{prev_reflection}\"\n"
                f"A verifier checked it against the true outcome and rejected it "
                f"with this critique: \"{critique}\"\n"
                f"Propose a revised, corrected rule that addresses the critique."
            ),
        })
    reply = chat(messages, temperature=0.0)
    return reply.strip().strip('"')
