"""
Diagnostic: Experiment 1's three baselines (ICL, Reflexion, HDR) all
landed on EXACTLY 50.0% overall/masked/seen accuracy -- too clean a
coincidence to accept without checking whether this is a genuine
constant-answer collapse (as found for Raw-Only, #39) or a silent
parsing-failure default (predict functions fall back to "NORMAL" if
the regex ANOMALY|NORMAL doesn't match the model's reply, which would
ALSO produce exactly 50% on this balanced test set, for a completely
different and much less interesting reason).

Run:
    python -m experiments_visual.diagnose_exp1_baseline_collapse
"""
import json
from collections import Counter
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parent.parent / "outputs_visual" / "experiment1_visual_results.json"

d = json.loads(OUT_PATH.read_text())
for method in ["ICL", "Reflexion", "HDR"]:
    items = d[method]["per_item"]
    preds = [i["predicted"] for i in items]
    print(f"\n=== {method} ===")
    print(f"Prediction distribution: {Counter(preds)}")
    print(f"First 8 predictions: {preds[:8]}")
    true_labels = [i["combo"]["label"] for i in items]
    print(f"True label distribution: {Counter(true_labels)}")
