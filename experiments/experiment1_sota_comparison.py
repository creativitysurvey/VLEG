"""
Experiment 1 — SOTA Comparative Analysis (paper Section IV-B).

Compares VLEG (K=3, hybrid grounding) against three baselines — HDR,
Reflexion, ICL — across Level 1, Level 2, and Level 3 of the
Experience-Transfer task. See core/baselines.py and DEVIATIONS_LOG.md
#14-16 for implementation choices and known limitations (esp. #15: our
oracle rule may not stress-test HDR's claimed weakness as intended).

Run:
    python -m experiments.experiment1_sota_comparison
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.llm_client import check_connection, MODEL_NAME
from core.task_env import (
    make_level3_split, all_combinations, level1_combinations,
    episode_text, generator_view_text,
)
from core.refinement import refine_episode
from core.memory import MemoryBank
from core.inference import predict_with_modality
from core.baselines import (
    build_icl_bank, icl_predict,
    build_reflexion_bank, reflexion_predict,
    hdr_generate_global_hypothesis, hdr_predict,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def build_vleg_bank(train_combos, K: int, log_prefix: str) -> MemoryBank:
    bank = MemoryBank()
    for idx, combo in enumerate(train_combos):
        etext = episode_text(combo)
        gtext = generator_view_text(combo)
        result = refine_episode(gtext, etext, K=K)
        if result.verified:
            bank.add(combo, etext, result.final_reflection)
        print(f"  [{log_prefix}] episode {idx+1}/{len(train_combos)}: "
              f"{'VERIFIED' if result.verified else 'DISCARDED'}")
    return bank


def eval_accuracy(predict_fn, test_combos) -> float:
    correct = 0
    for combo in test_combos:
        pred = predict_fn(combo)
        correct += int(pred == combo["label"])
    return correct / len(test_combos) if test_combos else 0.0


def run_level(level_name: str, train_combos, test_combos) -> dict:
    print(f"\n\n{'='*70}\n{level_name}: {len(train_combos)} train, {len(test_combos)} test\n{'='*70}")
    results = {}

    print(f"\n--- VLEG (K=3) ---")
    t0 = time.time()
    vleg_bank = build_vleg_bank(train_combos, K=3, log_prefix=f"{level_name}_VLEG")
    vleg_acc = eval_accuracy(lambda c: predict_with_modality(vleg_bank, c, "hybrid", k=3), test_combos)
    print(f"  VLEG accuracy: {vleg_acc*100:.1f}%  (bank {len(vleg_bank)}/{len(train_combos)})  "
          f"[{time.time()-t0:.1f}s]")
    results["VLEG"] = {"accuracy": vleg_acc, "bank_size": len(vleg_bank)}

    print(f"\n--- HDR (global hypothesis, K=3) ---")
    t0 = time.time()
    hdr_result = hdr_generate_global_hypothesis(train_combos, K=3, log_prefix=f"{level_name}_HDR")
    hdr_raw_bank = build_icl_bank(train_combos)  # HDR retrieves raw episodes for hybrid grounding
    hdr_acc = eval_accuracy(lambda c: hdr_predict(hdr_result.hypothesis, hdr_raw_bank, c, k=3), test_combos)
    print(f"  HDR accuracy: {hdr_acc*100:.1f}%  (hypothesis verified={hdr_result.verified})  "
          f"[{time.time()-t0:.1f}s]")
    print(f"  Final global hypothesis: \"{hdr_result.hypothesis}\"")
    results["HDR"] = {"accuracy": hdr_acc, "verified": hdr_result.verified, "hypothesis": hdr_result.hypothesis}

    print(f"\n--- Reflexion (single-step, unverified) ---")
    t0 = time.time()
    reflexion_bank = build_reflexion_bank(train_combos, log_prefix=f"{level_name}_Reflexion")
    reflexion_acc = eval_accuracy(lambda c: reflexion_predict(reflexion_bank, c, k=3), test_combos)
    print(f"  Reflexion accuracy: {reflexion_acc*100:.1f}%  [{time.time()-t0:.1f}s]")
    results["Reflexion"] = {"accuracy": reflexion_acc}

    print(f"\n--- ICL (raw episodic retention) ---")
    t0 = time.time()
    icl_bank = build_icl_bank(train_combos)
    icl_acc = eval_accuracy(lambda c: icl_predict(icl_bank, c, k=3), test_combos)
    print(f"  ICL accuracy: {icl_acc*100:.1f}%  [{time.time()-t0:.1f}s]")
    results["ICL"] = {"accuracy": icl_acc}

    return {"level": level_name, "n_train": len(train_combos), "n_test": len(test_combos),
            "results": results}


def main():
    print(f"Checking Ollama connection (model={MODEL_NAME!r})...")
    check_connection()

    level1_combos = level1_combinations()
    level2_combos = all_combinations()
    level3_train, level3_test = make_level3_split(mask_count=4, seed=42)

    out_path = OUTPUT_DIR / "experiment1_results.json"

    def save_partial(results):
        out_path.write_text(json.dumps(results, indent=2))
        print(f"  [checkpoint] partial results ({len(results)} level(s)) written to {out_path}")

    all_results = []
    all_results.append(run_level("Level1", level1_combos, level1_combos))
    save_partial(all_results)
    all_results.append(run_level("Level2", level2_combos, level2_combos))
    save_partial(all_results)
    all_results.append(run_level("Level3", level3_train, level3_test))
    save_partial(all_results)

    print(f"\n\n{'='*70}\nSUMMARY — Task Accuracy (%)\n{'='*70}")
    print(f"{'Level':<8} {'VLEG':<8} {'HDR':<8} {'Reflexion':<11} {'ICL':<8}")
    for r in all_results:
        res = r["results"]
        print(f"{r['level']:<8} {res['VLEG']['accuracy']*100:>6.1f}%  "
              f"{res['HDR']['accuracy']*100:>6.1f}%  "
              f"{res['Reflexion']['accuracy']*100:>9.1f}%  "
              f"{res['ICL']['accuracy']*100:>6.1f}%")

    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
