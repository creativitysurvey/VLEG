"""
Iterative Cyclic Refinement loop — Algorithm 1, Phase 1 (lines 3-15).

for each e_i:
    j = 0; c = None; v = False
    while not v and j < K:
        r = Generator(e_i, prev_r, c)
        v, c = Verifier(e_i, r)
        j += 1
    if v: commit (e_i, r) to memory

NOTE (see DEVIATIONS_LOG.md #6): the Generator and Verifier see different
renderings of e_i — the Generator gets the redacted (no-label) view, the
Verifier gets the full view including ground-truth label. Both are
grounded in the same underlying combo/episode; this split is what creates
real risk of the Generator's first guess being wrong.
"""
from dataclasses import dataclass, field
from core.generator import generate_reflection
from core.verifier import verify_reflection


@dataclass
class RefinementResult:
    episode_text: str
    verified: bool
    final_reflection: str | None
    iterations_used: int
    history: list = field(default_factory=list)  # list of (reflection, valid, critique)


def refine_episode(generator_view_text: str, verifier_view_text: str, K: int) -> RefinementResult:
    prev_reflection = None
    critique = None
    history = []
    for j in range(K):
        reflection = generate_reflection(generator_view_text, prev_reflection, critique)
        valid, critique_out = verify_reflection(verifier_view_text, reflection)
        history.append((reflection, valid, critique_out))
        if valid:
            return RefinementResult(
                episode_text=verifier_view_text, verified=True,
                final_reflection=reflection, iterations_used=j + 1,
                history=history,
            )
        prev_reflection, critique = reflection, critique_out
    return RefinementResult(
        episode_text=verifier_view_text, verified=False,
        final_reflection=None, iterations_used=K, history=history,
    )
