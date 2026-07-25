"""
Retrieval — Eq. III.11 / III.12, implemented faithfully using a real
embedding model (nomic-embed-text via Ollama), now that one is available.

S(T, M_i) = cosine( phi(T), phi(r*_i) )
M_top = top-k tuples by S

Embeddings are computed once per verified reflection (cached on the
MemoryTuple) and once per query at retrieval time.
"""
import math
from core.memory import MemoryBank, MemoryTuple
from core.llm_client import embed
from core.task_env import query_text


def _cosine(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def ensure_embedding(tup: MemoryTuple) -> list:
    """Lazily compute + cache the embedding of a memory tuple's verified reflection."""
    if not hasattr(tup, "_embedding") or tup._embedding is None:
        tup._embedding = embed(tup.verified_reflection)
    return tup._embedding


def retrieve_top_k(bank: MemoryBank, target_combo: dict, k: int = 3):
    if not bank.tuples:
        return []
    target_vec = embed(query_text(target_combo))
    scored = []
    for t in bank.tuples:
        t_vec = ensure_embedding(t)
        score = _cosine(target_vec, t_vec)
        scored.append((score, t))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:k]]


# --- Raw-episode-embedding retrieval, added for Experiment 1 baselines
# (ICL and HDR both retrieve over RAW episodes, not verified reflections
# — they either have no reflections at all (ICL) or only a single global
# one shared across all queries (HDR), so retrieval must be keyed on the
# episode text itself instead of core.retrieval.retrieve_top_k's
# reflection-based scoring.

def retrieve_top_k_by_episode(bank: MemoryBank, target_combo: dict, k: int = 3):
    if not bank.tuples:
        return []
    from core.task_env import query_text
    target_vec = embed(query_text(target_combo))
    scored = []
    for t in bank.tuples:
        if not hasattr(t, "_episode_embedding") or t._episode_embedding is None:
            t._episode_embedding = embed(t.episode_text)
        score = _cosine(target_vec, t._episode_embedding)
        scored.append((score, t))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:k]]
