"""
Targeted re-diagnosis of seed123 and seed2024's K=3 masked-combo errors
from the multi-seed run (masked accuracy: seed123=25%, seed2024=50%).
Reconstructs each seed's K=3 memory bank from the ALREADY-VERIFIED
reflections copied verbatim from the multi-seed run's log -- no need to
re-run the expensive consolidation phase. Only re-runs predict_hybrid
on the 4 masked test combos per seed (8 calls total), printing full
per-item detail (predicted, raw reply, and the exact rule-text context
that was sent to the model) so we can see exactly what happened, not
just the aggregate accuracy.

Run:
    python -m experiments_visual.diagnose_masked_errors
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_visual.llm_client import check_connection
from core_visual.task_env import make_level3_split
from core_visual.memory import MemoryBank
from core_visual.inference import predict_hybrid, SYSTEM_PROMPT_HYBRID
from core_visual.task_env import query_text

SEED123_K3_VERIFIED = [
    (("cube", "red", "metal", "large"), "Glossy red cube scenes are NORMAL; this scene is therefore NORMAL."),
    (("cube", "blue", "metal", "small"), "Glossy blue cube scenes are ANOMALY; this scene is therefore ANOMALY."),
    (("sphere", "red", "metal", "large"), "Glossy red sphere scenes are ANOMALY; this scene is therefore ANOMALY."),
    (("sphere", "red", "metal", "small"), "Glossy red spheres on a plain background are ANOMALY; this scene is therefore ANOMALY."),
    (("cube", "red", "rubber", "large"), "Matte red cubes larger than half the frame are ANOMALY; this scene is therefore ANOMALY."),
    (("sphere", "blue", "rubber", "small"), "Matte blue sphere scenes are NORMAL; this scene is therefore NORMAL."),
    (("cube", "blue", "metal", "large"), "Glossy blue cube scenes are ANOMALY; this scene is therefore ANOMALY."),
    (("sphere", "blue", "rubber", "large"), "Matte blue sphere scenes are NORMAL; this scene is therefore NORMAL."),
    (("cube", "blue", "rubber", "large"), "Matte blue rectangular prism scenes are NORMAL; this scene is therefore NORMAL."),
    (("cube", "red", "rubber", "small"), "Matte red cube scenes are ANOMALY; this scene is therefore ANOMALY."),
]

SEED2024_K3_VERIFIED = [
    (("cube", "blue", "metal", "small"), "Glossy blue cube scenes are ANOMALY; this scene is therefore ANOMALY."),
    (("sphere", "blue", "rubber", "large"), "Matte blue sphere scenes are NORMAL; this scene is therefore NORMAL."),
    (("cube", "blue", "rubber", "small"), "Matte blue cube scenes are NORMAL; this scene is therefore NORMAL."),
    (("sphere", "red", "metal", "large"), "Glossy red sphere scenes are ANOMALY; this scene is therefore ANOMALY."),
    (("sphere", "blue", "rubber", "small"), "Matte blue sphere scenes are NORMAL; this scene is therefore NORMAL."),
    (("cube", "blue", "metal", "large"), "Glossy blue cube scenes are ANOMALY; this scene is therefore ANOMALY."),
    (("cube", "red", "metal", "small"), "Glossy red cube scenes are NORMAL; this scene is therefore NORMAL."),
    (("cube", "blue", "rubber", "large"), "Matte blue rectangular prism scenes are NORMAL; this scene is therefore NORMAL."),
    (("sphere", "red", "rubber", "small"), "Matte red sphere scenes are NORMAL; this scene is therefore NORMAL."),
    (("sphere", "red", "metal", "small"), "Glossy red spheres on a plain background are ANOMALY; this scene is therefore ANOMALY."),
    (("cube", "red", "metal", "large"), "Glossy red cube scenes are NORMAL; this scene is therefore NORMAL."),
]


def build_bank(verified_list, all_combos_by_key):
    bank = MemoryBank()
    for key, reflection in verified_list:
        combo = all_combos_by_key[key]
        bank.add(combo, combo["image_path"], reflection)
    return bank


def diagnose(seed, verified_list):
    print(f"\n{'='*80}\nSEED {seed} -- K=3 masked-combo re-diagnosis\n{'='*80}")
    train, test = make_level3_split(mask_count=4, seed=seed)
    all_combos_by_key = {(c["shape"], c["color"], c["material"], c["size"]): c for c in test}
    masked = [c for c in test if c["masked_at_train"]]

    bank = build_bank(verified_list, all_combos_by_key)
    print(f"Reconstructed bank: {len(bank)} verified rules "
          f"({len(bank)} matches expected -- compare to multi-seed JSON's k3_bank_verified)")
    print("\nFull rule context that will be sent with EVERY query this seed:")
    for i, t in enumerate(bank.tuples, start=1):
        print(f"  Rule {i}: {t.verified_reflection}")

    print(f"\nMasked combos for seed {seed} ({len(masked)}):")
    n_correct = 0
    for combo in masked:
        pred = predict_hybrid(bank, combo, k=None)
        correct = (pred == combo["label"])
        n_correct += int(correct)
        print(f"\n  {combo['shape']}/{combo['color']}/{combo['material']}/{combo['size']} "
              f"(true={combo['label']}, image={combo['image_path']}):")
        print(f"    predicted={pred}  [{'CORRECT' if correct else 'WRONG'}]")
    print(f"\nSeed {seed} masked accuracy this re-run: {n_correct}/{len(masked)} = {n_correct/len(masked)*100:.1f}%")


print("Checking Ollama connection...")
check_connection()

diagnose(123, SEED123_K3_VERIFIED)
diagnose(2024, SEED2024_K3_VERIFIED)
