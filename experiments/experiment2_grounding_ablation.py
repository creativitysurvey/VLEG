"""
Experiment 2 — Component Isolation (Grounding Modality Ablation), paper
Section IV-C.

Builds ONE VLEG (K=3) memory bank per task level (Level 2: 16 combos,
no masking; Level 3: 12 training combos, 4 masked, evaluated on all 16).
Then evaluates Task Accuracy under three grounding modalities — Hybrid,
Abstract-Only, Raw-Only — using the SAME retrieved top-k tuples for all
three (see core/inference.py and DEVIATIONS_LOG.md #13 for why: this
isolates the effect of what's shown to the reasoning LLM, holding
retrieval quality constant).

Run:
    python -m experiments.experiment2_grounding_ablation
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.llm_client import check_connection, MODEL_NAME
from core.task_env import make_level3_split, all_combinations, episode_text, generator_view_text
from core.refinement import refine_episode
from core.memory import MemoryBank
from core.inference import predict_with_modality

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

MODALITIES = ["hybrid", "abstract_only", "raw_only"]


def build_vleg_bank(train_combos, K: int, log_prefix: str) -> MemoryBank:
    bank = MemoryBank()
    for idx, combo in enumerate(train_combos):
        etext = episode_text(combo)
        gtext = generator_view_text(combo)
        result = refine_episode(gtext, etext, K=K)
        if result.verified:
            bank.add(combo, etext, result.final_reflection)
            print(f"  [{log_prefix}] episode {idx+1}/{len(train_combos)}: VERIFIED -> "
                  f"\"{result.final_reflection}\"")
        else:
            print(f"  [{log_prefix}] episode {idx+1}/{len(train_combos)}: DISCARDED")
    print(f"  [{log_prefix}] memory bank size: {len(bank)}/{len(train_combos)}")
    return bank


def evaluate_modality(bank: MemoryBank, test_combos, modality: str, log_prefix: str) -> dict:
    correct = 0
    per_item = []
    for combo in test_combos:
        pred = predict_with_modality(bank, combo, modality, k=3)
        is_correct = (pred == combo["label"])
        correct += int(is_correct)
        per_item.append({
            "combo": {k: v for k, v in combo.items() if k != "masked_at_train"},
            "masked_at_train": combo.get("masked_at_train", False),
            "predicted": pred, "correct": is_correct,
        })
    accuracy = correct / len(test_combos) if test_combos else 0.0
    print(f"  [{log_prefix}] modality={modality}: accuracy={accuracy*100:.1f}% ({correct}/{len(test_combos)})")
    return {"modality": modality, "accuracy": accuracy, "n_correct": correct,
            "n_total": len(test_combos), "per_item": per_item}


def run_level(level_name: str, train_combos, test_combos) -> dict:
    print(f"\n=== {level_name}: building VLEG (K=3) memory bank from {len(train_combos)} episodes ===")
    t0 = time.time()
    bank = build_vleg_bank(train_combos, K=3, log_prefix=level_name)
    t1 = time.time()

    print(f"\n=== {level_name}: evaluating {len(test_combos)} test queries under 3 modalities ===")
    modality_results = {}
    for modality in MODALITIES:
        modality_results[modality] = evaluate_modality(bank, test_combos, modality, log_prefix=level_name)
    t2 = time.time()

    # Break out masked-vs-seen accuracy per modality, if applicable (Level 3 only)
    masked_combos_exist = any(c.get("masked_at_train") for c in test_combos)
    breakdown = {}
    if masked_combos_exist:
        for modality, res in modality_results.items():
            masked_items = [i for i in res["per_item"] if i["masked_at_train"]]
            seen_items = [i for i in res["per_item"] if not i["masked_at_train"]]
            masked_acc = sum(i["correct"] for i in masked_items) / len(masked_items) if masked_items else None
            seen_acc = sum(i["correct"] for i in seen_items) / len(seen_items) if seen_items else None
            breakdown[modality] = {"masked_accuracy": masked_acc, "seen_accuracy": seen_acc}
            print(f"  [{level_name}] modality={modality}: masked-combo acc={masked_acc*100:.1f}%  "
                  f"seen-combo acc={seen_acc*100:.1f}%")

    return {
        "level": level_name,
        "bank_size": len(bank), "n_train": len(train_combos),
        "consolidation_time_sec": round(t1 - t0, 1), "eval_time_sec": round(t2 - t1, 1),
        "modality_results": {m: {"accuracy": r["accuracy"], "n_correct": r["n_correct"], "n_total": r["n_total"]}
                              for m, r in modality_results.items()},
        "masked_vs_seen_breakdown": breakdown,
        "raw_results": {m: r for m, r in modality_results.items()},
    }


def main():
    print(f"Checking Ollama connection (model={MODEL_NAME!r})...")
    check_connection()

    level2_combos = all_combinations()  # 16 combos, no masking; train == test
    level3_train, level3_test = make_level3_split(mask_count=4, seed=42)  # 12 train, 16 test (4 masked)

    out_path = OUTPUT_DIR / "experiment2_results.json"

    def save_partial(results):
        out_path.write_text(json.dumps(results, indent=2))
        print(f"  [checkpoint] partial results ({len(results)} level(s)) written to {out_path}")

    results = []
    results.append(run_level("Level2", level2_combos, level2_combos))
    save_partial(results)
    results.append(run_level("Level3", level3_train, level3_test))
    save_partial(results)

    print(f"\n\n{'='*80}")
    print(f"{'Level':<8} {'Hybrid':<10} {'Abstract-Only':<16} {'Raw-Only':<10}")
    print('='*80)
    for r in results:
        h = r["modality_results"]["hybrid"]["accuracy"] * 100
        a = r["modality_results"]["abstract_only"]["accuracy"] * 100
        raw = r["modality_results"]["raw_only"]["accuracy"] * 100
        print(f"{r['level']:<8} {h:>7.1f}%   {a:>13.1f}%   {raw:>7.1f}%")

    print(f"\n{'Level':<8} {'Modality':<15} {'Masked acc':<12} {'Seen acc':<10}")
    for r in results:
        if not r["masked_vs_seen_breakdown"]:
            continue
        for modality, b in r["masked_vs_seen_breakdown"].items():
            ma = f"{b['masked_accuracy']*100:.1f}%" if b['masked_accuracy'] is not None else "N/A"
            sa = f"{b['seen_accuracy']*100:.1f}%" if b['seen_accuracy'] is not None else "N/A"
            print(f"{r['level']:<8} {modality:<15} {ma:<12} {sa:<10}")

    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
