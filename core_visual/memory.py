"""Hybrid Memory Bank -- real-image version. Stores image_path alongside
the verified reflection so Hybrid Grounding can show the image again at
inference time."""
from dataclasses import dataclass, field


@dataclass
class MemoryTuple:
    combo: dict
    image_path: str
    verified_reflection: str
    _embedding: list = None


@dataclass
class MemoryBank:
    tuples: list = field(default_factory=list)

    def add(self, combo: dict, image_path: str, verified_reflection: str):
        self.tuples.append(MemoryTuple(combo, image_path, verified_reflection))

    def __len__(self):
        return len(self.tuples)
