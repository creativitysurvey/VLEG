"""
Experiment 5 — Diagnostic Assessment of Memory Quality (paper Section
IV-F, Table I).

Compares Single-Step Reflection (no verification gate, Reflexion-style)
against full VLEG (K=3) on two metrics, across Level 2 (16 combos, no
masking) and Level 3 (12 training combos, 4 masked):
  - Reflection Validity Rate: % of episodes that yield a committed
    reflection (N/A for Single-Step, since it always commits by
    definition; n_verified/n_total for VLEG).
  - Hallucination Propagation Rate: % of COMMITTED reflections flagged
    as hallucinated by an independent oracle auditor that knows the true
    ground-truth rule (core/oracle_verifier.py) — not V_phi, which only
    checks per-episode consistency and structurally cannot catch
    over-specified/spurious reflections (see DEVIATIONS_LOG.md #6-#9).

Run:
    python -m experiments.experiment5_memory_quality
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.llm_client import check_connection, MODEL_NAME, embed
from core.task_env import make_level3_split, all_combinations, episode_text, generator_view_text
from core.refinement import refine_episode
from core.single_step import single_step_commit
from core.oracle_verifier import audit_reflection

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def run_single_step(combos, log_prefix: str) -> list[dict]:
    """Returns list of {combo, reflection} for ALL combos (always commits)."""
    committed = []
    for idx, combo in enumerate(combos):
        gtext = generator_view_text(combo)
        reflection = single_step_commit(gtext)
        committed.append({"combo": combo, "reflection": reflection})
        print(f"  [{log_prefix}] episode {idx+1}/{len(combos)}: COMMITTED (no gate) -> \"{reflection}\"")
    return committed


def run_vleg(combos, K: int, log_prefix: str) -> tuple[list[dict], dict]:
    """Returns (list of {combo, reflection} for VERIFIED-only, stats dict)."""
    committed = []
    n_verified = 0
    for idx, combo in enumerate(combos):
        etext = episode_text(combo)
        gtext = generator_view_text(combo)
        result = refine_episode(gtext, etext, K=K)
        if result.verified:
            committed.append({"combo": combo, "reflection": result.final_reflection})
            n_verified += 1
            print(f"  [{log_prefix}] episode {idx+1}/{len(combos)}: VERIFIED after "
                  f"{result.iterations_used} iter(s) -> \"{result.final_reflection}\"")
        else:
            print(f"  [{log_prefix}] episode {idx+1}/{len(combos)}: DISCARDED after {K} iter(s)")
    stats = {"n_total": len(combos), "n_verified": n_verified,
              "validity_rate": n_verified / len(combos) if combos else 0.0}
    return committed, stats


def audit_hallucination_rate(committed: list[dict], log_prefix: str) -> dict:
    n_hallucinated = 0
    details = []
    for item in committed:
        hallucinated, reason = audit_reflection(item["reflection"])
        n_hallucinated += int(hallucinated)
        details.append({"reflection": item["reflection"], "hallucinated": hallucinated, "reason": reason})
        flag = "HALLUCINATED" if hallucinated else "ok"
        print(f"  [{log_prefix}] oracle audit: [{flag}] \"{item['reflection']}\" — {reason}")
    rate = n_hallucinated / len(committed) if committed else 0.0
    return {"n_committed": len(committed), "n_hallucinated": n_hallucinated,
            "hallucination_rate": rate, "details": details}


def run_condition(level_name: str, combos, strategy: str, K: int = 3) -> dict:
    print(f"\n=== {level_name} — {strategy} ===")
    t0 = time.time()
    if strategy == "single_step":
        committed = run_single_step(combos, log_prefix=f"{level_name}_{strategy}")
        validity_rate = None  # N/A — always commits, not a meaningful gate metric
    else:
        committed, vstats = run_vleg(combos, K=K, log_prefix=f"{level_name}_{strategy}")
        validity_rate = vstats["validity_rate"]
    t1 = time.time()

    print(f"-- Auditing {len(committed)} committed reflections against oracle rule --")
    audit = audit_hallucination_rate(committed, log_prefix=f"{level_name}_{strategy}")
    t2 = time.time()

    print(f"{level_name} / {strategy}: Validity Rate = "
          f"{'N/A' if validity_rate is None else f'{validity_rate*100:.1f}%'}, "
          f"Hallucination Propagation Rate = {audit['hallucination_rate']*100:.1f}% "
          f"({audit['n_hallucinated']}/{audit['n_committed']})")

    return {
        "level": level_name, "strategy": strategy,
        "validity_rate": validity_rate,
        "hallucination_rate": audit["hallucination_rate"],
        "n_committed": audit["n_committed"], "n_hallucinated": audit["n_hallucinated"],
        "consolidation_time_sec": round(t1 - t0, 1),
        "audit_time_sec": round(t2 - t1, 1),
        "details": audit["details"],
    }


def main():
    print(f"Checking Ollama connection (model={MODEL_NAME!r})...")
    check_connection()

    level2_combos = all_combinations()               # 16 combos, no masking
    level3_train_combos, _ = make_level3_split(mask_count=4, seed=42)  # 12 combos

    results = []
    results.append(run_condition("Level2", level2_combos, "single_step"))
    results.append(run_condition("Level2", level2_combos, "VLEG", K=3))
    results.append(run_condition("Level3", level3_train_combos, "single_step"))
    results.append(run_condition("Level3", level3_train_combos, "VLEG", K=3))

    print(f"\n\n{'='*78}")
    print(f"{'Task Level':<10} {'Strategy':<14} {'Validity Rate':<15} {'Hallucination Rate':<20}")
    print('='*78)
    for r in results:
        vr = "N/A (No Verification)" if r["validity_rate"] is None else f"{r['validity_rate']*100:.1f}%"
        hr = f"{r['hallucination_rate']*100:.1f}%"
        print(f"{r['level']:<10} {r['strategy']:<14} {vr:<15} {hr:<20}")

    out_path = OUTPUT_DIR / "experiment5_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
