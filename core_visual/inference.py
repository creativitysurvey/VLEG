"""
Hybrid Episodic-Semantic Grounding inference -- real-image version.

DEVIATION #2 from the naive real-image design (see DEVIATIONS_LOG.md
#25): past examples are shown as TEXT-ONLY (their verified rule), NOT
their images. Only the TARGET combo's image is actually attached --
every call has exactly ONE image, regardless of memory bank size.

MITIGATION (see DEVIATIONS_LOG.md #27-29 for the problem, #30 for this
fix and its validation): predict_hybrid can optionally run a cross-
episode attribute-skew check (core_visual/debias.py) before calling the
model, and inject a warning into the prompt if every reflection sharing
a label happens to share an incidental value of some other attribute
(e.g. all ANOMALY examples being red). This is OFF by default
(debias=False) so existing call sites reproduce prior results exactly;
pass debias=True to enable it.
"""
import re
from core_visual.llm_client import chat
from core_visual.memory import MemoryBank
from core_visual.task_env import query_text
from core_visual.debias import detect_attribute_skew, format_skew_warning

SYSTEM_PROMPT_HYBRID = (
    "You classify new scenes as ANOMALY or NORMAL. You are given several "
    "verified rules distilled from past observations (no images of "
    "those past scenes are shown to you, only the rules), followed by "
    "ONE target image to classify. Use the rules and your own visual "
    "inspection of the target image to decide. "
    "Reply with ONLY one word: ANOMALY or NORMAL."
)


def predict_hybrid(bank: MemoryBank, target_combo: dict, k: int = None, debias: bool = False) -> str:
    """`k` is accepted for call-signature compatibility with the symbolic
    version but ignored -- ALL verified rules in the bank are used as
    text context. Set debias=True to enable the cross-episode
    attribute-skew warning (see module docstring and core_visual/debias.py)."""
    tuples = bank.tuples
    parts = [f"Verified rule {i}: {t.verified_reflection}" for i, t in enumerate(tuples, start=1)]

    system_prompt = SYSTEM_PROMPT_HYBRID
    if debias:
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


def predict(bank: MemoryBank, target_combo: dict, k: int = None, debias: bool = False) -> str:
    return predict_hybrid(bank, target_combo, k=k, debias=debias)
