"""
Follow-up diagnostic: Experiment 1's ICL, Reflexion, and HDR all
collapsed to EXACTLY 32/32 NORMAL -- too clean to accept without
seeing the model's RAW reply text (not just the post-regex "predicted"
label the main script saved), since a silent parsing-failure default
("NORMAL" if the ANOMALY|NORMAL regex doesn't match) would look
identical in the aggregate to a genuine constant-answer collapse.

Re-runs a small number of fresh calls for each baseline against a
true-ANOMALY target (where a genuine collapse should visibly fail) and
prints the FULL raw reply, not just the extracted label.

Reuses the same seed=42 split and the cached VLEG/Reflexion banks are
NOT re-consolidated -- ICL needs no consolidation at all; Reflexion and
HDR are re-consolidated fresh here (cheap relative to the original run,
since we only need enough to test inference, not the full 32-item eval).

Run:
    python -m experiments_visual.diagnose_exp1_raw_replies
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_visual.llm_client import check_connection, chat
from core_visual.task_env import make_level3_split_large, query_text
from core_visual.baselines import (
    build_icl_bank, build_reflexion_bank, hdr_generate_global_hypothesis,
    SYSTEM_PROMPT_RAW_ONLY,
)
from core_visual.inference import SYSTEM_PROMPT_HYBRID

print("Checking Ollama connection...")
check_connection()

train_combos, test_items = make_level3_split_large(mask_count=4, seed=42, n_test_per_combo=2)
true_anomaly_target = next(c for c in test_items if c["label"] == "ANOMALY")
print(f"\nUsing a TRUE-ANOMALY target: {true_anomaly_target['shape']}/{true_anomaly_target['color']}/"
      f"{true_anomaly_target['material']}/{true_anomaly_target['size']} -> {true_anomaly_target['image_path']}")

print(f"\n{'='*70}\n=== ICL raw reply ===\n{'='*70}")
icl_bank = build_icl_bank(train_combos)
image_paths = [t.image_path for t in icl_bank.tuples]
parts = [f"[Image {i}] Past example, true label: {t.combo['label']}"
         for i, t in enumerate(icl_bank.tuples, start=1)]
parts.append(f"[Image {len(icl_bank.tuples)+1}] TARGET scene to classify (this is the last image). "
             f"{query_text(true_anomaly_target)}")
image_paths.append(true_anomaly_target["image_path"])
prompt = "\n\n".join(parts)
reply = chat([{"role": "system", "content": SYSTEM_PROMPT_RAW_ONLY},
              {"role": "user", "content": prompt}], temperature=0.0, images=image_paths)
print(f"RAW REPLY: {reply!r}")
print(f"Regex match: {re.search(r'ANOMALY|NORMAL', reply.upper())}")

print(f"\n{'='*70}\n=== Reflexion raw reply (fresh bank) ===\n{'='*70}")
reflexion_bank = build_reflexion_bank(train_combos, K=3, log_prefix="reflexion_diag")
parts = [f"Verified rule {i}: {t.verified_reflection}" for i, t in enumerate(reflexion_bank.tuples, start=1)]
parts.append(f"Target scene (image attached). {query_text(true_anomaly_target)}")
prompt = "\n\n".join(parts)
reply = chat([{"role": "system", "content": SYSTEM_PROMPT_HYBRID},
              {"role": "user", "content": prompt}], temperature=0.0,
             images=[true_anomaly_target["image_path"]])
print(f"RAW REPLY: {reply!r}")
print(f"Regex match: {re.search(r'ANOMALY|NORMAL', reply.upper())}")

print(f"\n{'='*70}\n=== HDR raw reply (fresh hypothesis) ===\n{'='*70}")
hdr_result = hdr_generate_global_hypothesis(train_combos, K=1, log_prefix="hdr_diag")  # K=1 for speed, just need a hypothesis to test inference
print(f"Hypothesis used: {hdr_result.hypothesis}")
prompt = f"Global rule: {hdr_result.hypothesis}\n\nTarget scene (image attached). {query_text(true_anomaly_target)}"
reply = chat([{"role": "system", "content": (
    "You classify new scenes as ANOMALY or NORMAL using a global rule "
    "and your own visual inspection of the target image. "
    "Reply with ONLY one word: ANOMALY or NORMAL.")},
             {"role": "user", "content": prompt}],
            temperature=0.0, images=[true_anomaly_target["image_path"]])
print(f"RAW REPLY: {reply!r}")
print(f"Regex match: {re.search(r'ANOMALY|NORMAL', reply.upper())}")
