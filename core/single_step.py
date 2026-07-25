"""
Single-Step Reflection baseline (paper's Reflexion-style comparator,
Section IV-F / Table I) — commits the initial candidate reflection
r_i^(0) DIRECTLY to memory with NO verification gate at all. This is
distinct from Experiment 3's "K=1" condition (core/refinement.py with
K=1), which still runs the Strict Verifier once and DISCARDS on failure.
Table I's baseline explicitly has no verification step ("N/A (No
Verification)"), matching the paper's IV-D prose description — see
DEVIATIONS_LOG.md #7.
"""
from core.generator import generate_reflection


def single_step_commit(generator_view_text: str) -> str:
    """Generate once, return the reflection unconditionally (no gate)."""
    return generate_reflection(generator_view_text, prev_reflection=None, critique=None)
