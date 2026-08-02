"""
Reflection Generator G_theta -- real-image version.

At j=0, the generator sees ONLY the image + a label-free text prompt
(core_visual.task_env.generator_view_text) -- it must actually look at
the image to determine shape/color/material/size, nothing is spelled
out in text. On refinement (j>0), it also gets its own prior guess and
the verifier's critique (still no label revealed directly, but the
critique will typically state the correct label since the verifier DOES
see it).
"""
from core_visual.llm_client import chat

SYSTEM_PROMPT = (
    "You are a careful visual pattern-analysis assistant. You are shown "
    "an image of a single 3D object on a plain background. The scene's "
    "true outcome label (ANOMALY or NORMAL) is NOT given to you -- you "
    "must infer it yourself from what you see, and propose a SHORT, "
    "CONCRETE, testable heuristic rule (one or two sentences) stating "
    "which visual attribute condition you believe determines ANOMALY "
    "vs NORMAL, and what this particular scene's label is under that "
    "rule. Describe only attributes you can actually see in the image "
    "(e.g. object shape, color, whether its surface looks glossy/"
    "metallic or matte/rubber-like, and its apparent size). State it as "
    "a condition, e.g. 'Glossy red cube scenes are ANOMALY; this scene "
    "is therefore ANOMALY.' Reply with ONLY the rule sentence, nothing else."
)


def generate_reflection(image_path: str, generator_view_text: str,
                         prev_reflection: str = None, critique: str = None) -> str:
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
                f"Propose a revised, corrected rule that addresses the critique. "
                f"The image is attached again for reference."
            ),
        })
    reply = chat(messages, temperature=0.0, images=[image_path])
    return reply.strip().strip('"')
