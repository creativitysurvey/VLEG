"""
Hybrid Memory Bank M — Eq. III.6.

M = { (e_i, r*_i) | exists j <= K s.t. V_phi(e_i, r_i^(j)) = 1 }
"""
from dataclasses import dataclass, field


@dataclass
class MemoryTuple:
    combo: dict            # attribute dict from task_env (incl. label)
    episode_text: str
    verified_reflection: str
    _embedding: list = None  # lazily populated cache, see core/retrieval.py


@dataclass
class MemoryBank:
    tuples: list = field(default_factory=list)

    def add(self, combo: dict, episode_text: str, verified_reflection: str):
        self.tuples.append(MemoryTuple(combo, episode_text, verified_reflection))

    def __len__(self):
        return len(self.tuples)
