"""
Diagnostic: reconstruct the K=1 and K=3 memory banks from the ALREADY-
VERIFIED reflections pasted from a completed run's terminal log (no need
to re-run the expensive Generator/Verifier consolidation loop), then run
retrieval + prediction on just the 4 masked test combos with full
visibility into which memory tuples got retrieved and their similarity
scores. This isolates whether the K=3-worse-on-masked result traces back
to retrieval pulling in spurious near-neighbors.

Run:
    python -m experiments.diagnose_masked_retrieval
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.task_env import make_level3_split, episode_text
from core.memory import MemoryBank
from core.retrieval import retrieve_top_k, _cosine
from core.llm_client import embed
from core.inference import predict, build_hybrid_prompt

train, test = make_level3_split()
masked = [c for c in test if c["masked_at_train"]]

# Verified reflections copied verbatim from the pasted run log.
# Keys are 1-indexed episode numbers matching train[i-1].
K1_VERIFIED = {
    6: "On a white background, striped large shapes are ANOMALY; this scene is therefore ANOMALY.",
    7: "On a black background, square+blue scenes are ANOMALY; this scene is therefore ANOMALY.",
    8: "On a black background, large striped scenes are ANOMALY; this scene is therefore ANOMALY.",
    10: "On a black background, large striped scenes are ANOMALY; this scene is therefore ANOMALY.",
    11: "On a white background, circle+red scenes are ANOMALY; this scene is therefore ANOMALY.",
}

K3_VERIFIED = {
    1: "On a white background, large striped scenes are NORMAL; this scene is therefore NORMAL.",
    2: "On a white background, large striped scenes are NORMAL; this scene is therefore NORMAL.",
    3: "On a white background, large solid blue square scenes are NORMAL; this scene is therefore NORMAL.",
    4: "On a white background, small solid-colored square scenes are NORMAL; this scene is therefore NORMAL.",
    5: "On a black background, large solid-colored circle scenes are NORMAL; this scene is therefore NORMAL.",
    6: "On a white background, striped large shapes are ANOMALY; this scene is therefore ANOMALY.",
    7: "On a black background, square+blue scenes are ANOMALY; this scene is therefore ANOMALY.",
    8: "On a black background, large striped scenes are ANOMALY; this scene is therefore ANOMALY.",
    9: "On a white background, blue solid small circle scenes are NORMAL; this scene is therefore NORMAL.",
    10: "On a black background, large striped scenes are ANOMALY; this scene is therefore ANOMALY.",
    11: "On a white background, circle+red scenes are ANOMALY; this scene is therefore ANOMALY.",
    12: "On a black background, circle+red scenes are NORMAL; this scene is therefore NORMAL.",
}


def build_bank(verified_dict):
    bank = MemoryBank()
    for ep_num, reflection in verified_dict.items():
        combo = train[ep_num - 1]
        bank.add(combo, episode_text(combo), reflection)
    return bank


def diagnose(bank: MemoryBank, label: str):
    print(f"\n{'='*70}\n{label}  (bank size = {len(bank)})\n{'='*70}")
    correct = 0
    for combo in masked:
        query = (f"New scene: background={combo['background']}, shape={combo['shape']}, "
                 f"color={combo['color']}, texture={combo['texture']}, size={combo['size']}. "
                 f"Is this ANOMALY or NORMAL?")
        query_vec = embed(query)
        scored = sorted(
            ((_cosine(query_vec, embed(t.verified_reflection)), t) for t in bank.tuples),
            key=lambda x: x[0], reverse=True,
        )
        print(f"\nQuery: {combo} -> TRUE LABEL: {combo['label']}")
        print("  Top-3 retrieved by cosine similarity:")
        for score, t in scored[:3]:
            print(f"    score={score:.3f}  combo={ {k:v for k,v in t.combo.items() if k!='masked_at_train'} }")
            print(f"             reflection=\"{t.verified_reflection}\"")
        pred = predict(bank, combo, k=3)
        is_correct = (pred == combo["label"])
        correct += int(is_correct)
        print(f"  -> PREDICTION: {pred}  ({'CORRECT' if is_correct else 'WRONG'})")
    print(f"\n{label} masked-combo accuracy: {correct}/{len(masked)} = {correct/len(masked)*100:.1f}%")


if __name__ == "__main__":
    k1_bank = build_bank(K1_VERIFIED)
    k3_bank = build_bank(K3_VERIFIED)
    diagnose(k1_bank, "K=1 bank")
    diagnose(k3_bank, "K=3 bank")
