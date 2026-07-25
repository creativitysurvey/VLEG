"""
Experiment 4 — Parameter Sensitivity and Overhead Analysis (paper
Section IV-E).

Sweeps K in {1, 2, 3, 5}, builds a VLEG memory bank at each K for both
Level 2 and Level 3, evaluates hybrid-grounding Task Accuracy, and
tracks the token overhead consumed strictly during the CONSOLIDATION
phase (Generator + Verifier calls), matching the paper's definition:
"total number of prompt and completion tokens consumed per consolidated
episodic memory."

Run:
    python -m experiments.experiment4_parameter_sensitivity
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.llm_client import check_connection, MODEL_NAME, reset_token_usage, get_token_usage
from core.task_env import make_level3_split, all_combinations, episode_text, generator_view_text
from core.refinement import refine_episode
from core.memory import MemoryBank
from core.inference import predict_with_modality

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

K_VALUES = [1, 2, 3, 5]


def build_vleg_bank(train_combos, K: int, log_prefix: str) -> MemoryBank:
    bank = MemoryBank()
    for idx, combo in enumerate(train_combos):
        etext = episode_text(combo)
        gtext = generator_view_text(combo)
        result = refine_episode(gtext, etext, K=K)
        if result.verified:
            bank.add(combo, etext, result.final_reflection)
        print(f"  [{log_prefix}] episode {idx+1}/{len(train_combos)}: "
              f"{'VERIFIED (' + str(result.iterations_used) + ' iter)' if result.verified else 'DISCARDED'}")
    return bank


def evaluate_hybrid(bank: MemoryBank, test_combos) -> float:
    correct = 0
    for combo in test_combos:
        pred = predict_with_modality(bank, combo, "hybrid", k=3)
        correct += int(pred == combo["label"])
    return correct / len(test_combos) if test_combos else 0.0


def run_condition(level_name: str, train_combos, test_combos, K: int) -> dict:
    print(f"\n=== {level_name} — K={K} ===")
    reset_token_usage()
    t0 = time.time()
    bank = build_vleg_bank(train_combos, K=K, log_prefix=f"{level_name}_K{K}")
    t1 = time.time()
    consolidation_usage = get_token_usage()  # tokens used ONLY during consolidation

    accuracy = evaluate_hybrid(bank, test_combos)
    t2 = time.time()

    total_tokens = consolidation_usage["prompt_tokens"] + consolidation_usage["completion_tokens"]
    tokens_per_episode = total_tokens / len(train_combos) if train_combos else 0.0

    print(f"  [{level_name} K={K}] bank={len(bank)}/{len(train_combos)}  "
          f"accuracy={accuracy*100:.1f}%  "
          f"consolidation_tokens={total_tokens} ({tokens_per_episode:.0f}/episode)  "
          f"consolidation_time={t1-t0:.1f}s")

    return {
        "level": level_name, "K": K,
        "bank_size": len(bank), "n_train": len(train_combos),
        "accuracy": accuracy,
        "consolidation_prompt_tokens": consolidation_usage["prompt_tokens"],
        "consolidation_completion_tokens": consolidation_usage["completion_tokens"],
        "consolidation_total_tokens": total_tokens,
        "consolidation_calls": consolidation_usage["calls"],
        "tokens_per_episode": tokens_per_episode,
        "consolidation_time_sec": round(t1 - t0, 1),
        "eval_time_sec": round(t2 - t1, 1),
    }


def main():
    print(f"Checking Ollama connection (model={MODEL_NAME!r})...")
    check_connection()

    level2_combos = all_combinations()
    level3_train, level3_test = make_level3_split(mask_count=4, seed=42)

    results = []
    for K in K_VALUES:
        results.append(run_condition("Level2", level2_combos, level2_combos, K))
        results.append(run_condition("Level3", level3_train, level3_test, K))

    print(f"\n\n{'='*90}")
    print(f"{'Level':<8} {'K':<4} {'Accuracy':<10} {'Total tokens':<14} {'Tokens/episode':<16} {'Time(s)':<8}")
    print('='*90)
    for r in results:
        print(f"{r['level']:<8} {r['K']:<4} {r['accuracy']*100:>7.1f}%   "
              f"{r['consolidation_total_tokens']:>11}   {r['tokens_per_episode']:>13.0f}   "
              f"{r['consolidation_time_sec']:>6.1f}")

    out_path = OUTPUT_DIR / "experiment4_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
