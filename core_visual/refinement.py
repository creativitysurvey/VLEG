"""
Iterative Cyclic Refinement loop -- real-image version. Same K-bounded
generate-verify-refine structure as core/refinement.py, but threads an
image_path through to both the Generator and Verifier.
"""
from dataclasses import dataclass, field
from core_visual.generator import generate_reflection
from core_visual.verifier import verify_reflection


@dataclass
class RefinementResult:
    verified: bool
    final_reflection: str
    iterations_used: int
    history: list = field(default_factory=list)


def refine_episode(image_path: str, generator_view_text: str, verifier_view_text: str, K: int) -> RefinementResult:
    prev_reflection = None
    critique = None
    history = []
    for j in range(K):
        reflection = generate_reflection(image_path, generator_view_text, prev_reflection, critique)
        valid, critique_out = verify_reflection(image_path, verifier_view_text, reflection)
        history.append((reflection, valid, critique_out))
        if valid:
            return RefinementResult(verified=True, final_reflection=reflection,
                                     iterations_used=j + 1, history=history)
        prev_reflection, critique = reflection, critique_out
    return RefinementResult(verified=False, final_reflection=None, iterations_used=K, history=history)
