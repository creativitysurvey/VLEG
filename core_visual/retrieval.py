"""Retrieval -- real-image version. Identical mechanism to core/retrieval.py
(cosine similarity over verified-reflection embeddings) -- retrieval
still operates on TEXT (the reflections), not images, so this logic is
unchanged in substance, just imports from core_visual.llm_client."""
import math
from core_visual.memory import MemoryBank
from core_visual.llm_client import embed


def _cosine(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def ensure_embedding(tup) -> list:
    if not hasattr(tup, "_embedding") or tup._embedding is None:
        tup._embedding = embed(tup.verified_reflection)
    return tup._embedding


def retrieve_top_k(bank: MemoryBank, query_text: str, k: int = 3):
    if not bank.tuples:
        return []
    target_vec = embed(query_text)
    scored = []
    for t in bank.tuples:
        t_vec = ensure_embedding(t)
        score = _cosine(target_vec, t_vec)
        scored.append((score, t))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:k]]
