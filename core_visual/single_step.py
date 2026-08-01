"""
Single-step, UNVERIFIED reflection commitment -- real-image version.
Mirrors the symbolic pipeline's core/single_step.py: generates exactly
ONE reflection per episode and always commits it, no gate, no
discard -- the baseline VLEG's K-bounded verify-or-discard loop is
compared against in Experiment 5.
"""
from core_visual.generator import generate_reflection
from core_visual.task_env import generator_view_text
from core_visual.memory import MemoryBank


def single_step_commit(image_path: str, gtext: str) -> str:
    """One unverified attempt, always returned regardless of correctness."""
    return generate_reflection(image_path, gtext, prev_reflection=None, critique=None)


def build_single_step_bank(train_combos: list, log_prefix: str = "single_step") -> MemoryBank:
    bank = MemoryBank()
    for idx, combo in enumerate(train_combos):
        gtext = generator_view_text(combo)
        reflection = single_step_commit(combo["image_path"], gtext)
        bank.add(combo, combo["image_path"], reflection)
        print(f"  [{log_prefix}] episode {idx+1}/{len(train_combos)} "
              f"({combo['shape']}/{combo['color']}/{combo['material']}/{combo['size']}, "
              f"true={combo['label']}): COMMITTED (no gate) -> \"{reflection}\"")
    return bank
