"""
Hybrid Episodic-Semantic Grounding inference — Eq. III.13 / III.14 and
Algorithm 1, Phase 2 (lines 17-20).

P = PromptBuilder( [Episode_i (+) Verified_Rule_i]_{i in top-k} (+) Target )
y_hat = LLM(P)

Extended for Experiment 2 (Grounding Modality Ablation, Section IV-C):
supports Hybrid / Abstract-Only / Raw-Only prompt construction. All three
modalities use the SAME top-k retrieval (via reflection-embedding cosine
similarity, core/retrieval.py) — this is a deliberate methodological
choice, see DEVIATIONS_LOG.md #13: it holds retrieval quality constant
across conditions so the ablation isolates the effect of what's SHOWN
to the reasoning LLM, not differences in what got retrieved.
"""
import re
from core.llm_client import chat
from core.retrieval import retrieve_top_k
from core.memory import MemoryBank
from core.task_env import query_text

SYSTEM_PROMPT_HYBRID = (
    "You classify new scenes as ANOMALY or NORMAL. You are given several "
    "past examples, each with the raw observed scene and a verified rule "
    "that was validated against it, followed by a new scene to classify. "
    "Use the examples and rules to decide. "
    "Reply with ONLY one word: ANOMALY or NORMAL."
)

SYSTEM_PROMPT_ABSTRACT_ONLY = (
    "You classify new scenes as ANOMALY or NORMAL. You are given several "
    "verified rules distilled from past observations (the raw scenes "
    "themselves are NOT available to you), followed by a new scene to "
    "classify. Use the rules to decide. "
    "Reply with ONLY one word: ANOMALY or NORMAL."
)

SYSTEM_PROMPT_RAW_ONLY = (
    "You classify new scenes as ANOMALY or NORMAL. You are given several "
    "raw past observed scenes with their outcome labels (no distilled "
    "rules are available to you), followed by a new scene to classify. "
    "Infer the pattern yourself from the raw examples. "
    "Reply with ONLY one word: ANOMALY or NORMAL."
)


def build_hybrid_prompt(top_tuples, target_combo: dict) -> str:
    parts = [f"Example — {t.episode_text} Verified rule: {t.verified_reflection}" for t in top_tuples]
    parts.append(f"Target: {query_text(target_combo)}")
    return "\n\n".join(parts)


def build_abstract_only_prompt(top_tuples, target_combo: dict) -> str:
    parts = [f"Verified rule: {t.verified_reflection}" for t in top_tuples]
    parts.append(f"Target: {query_text(target_combo)}")
    return "\n\n".join(parts)


def build_raw_only_prompt(top_tuples, target_combo: dict) -> str:
    parts = [f"Example — {t.episode_text}" for t in top_tuples]
    parts.append(f"Target: {query_text(target_combo)}")
    return "\n\n".join(parts)


_MODALITY_CONFIG = {
    "hybrid": (build_hybrid_prompt, SYSTEM_PROMPT_HYBRID),
    "abstract_only": (build_abstract_only_prompt, SYSTEM_PROMPT_ABSTRACT_ONLY),
    "raw_only": (build_raw_only_prompt, SYSTEM_PROMPT_RAW_ONLY),
}


def predict_with_modality(bank: MemoryBank, target_combo: dict, modality: str, k: int = 3) -> str:
    if modality not in _MODALITY_CONFIG:
        raise ValueError(f"unknown modality {modality!r}, expected one of {list(_MODALITY_CONFIG)}")
    build_fn, system_prompt = _MODALITY_CONFIG[modality]
    top_tuples = retrieve_top_k(bank, target_combo, k=k)
    if not top_tuples:
        return "NORMAL"  # degenerate case, no memory to ground on
    prompt = build_fn(top_tuples, target_combo)
    reply = chat(
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": prompt}],
        temperature=0.0,
    )
    match = re.search(r"ANOMALY|NORMAL", reply.upper())
    return match.group(0) if match else "NORMAL"


def predict(bank: MemoryBank, target_combo: dict, k: int = 3) -> str:
    """Backward-compatible alias — hybrid grounding (used by Experiment 3)."""
    return predict_with_modality(bank, target_combo, "hybrid", k=k)
