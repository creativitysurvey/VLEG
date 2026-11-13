"""Supplementary real experiments & exact provenance extraction for pre-submission revision:
1. Inspect traffic/data.py for exact 19-feature schema and exclusion rationale.
2. Evaluate zero-LLM-query hybrid: VLEG-CR-Bank / VLEG-CR-NS + Shallow CART (depth=3) fallback across Pilot & 5 Formal seeds.
3. Run 'Label-Given Unverified Single-Step Reflection' control across all 160 support flows (5 seeds) to answer Fairness Critique #2.
4. Extract a real 4-stage rule evolution case study + exact prompt templates from saved checkpoints.
"""
import json
import statistics
import sys
import time
import numpy as np
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier

REPO = Path("/data/liwei/VLEG")
sys.path.insert(0, str(REPO))

from scripts.audit_explanations_and_provenance import parse_rule
from scripts.run_vleg_cr import FLOW_ROOT, OUT_ROOT, SEEDS, safe_applies, evaluate_rule_on_seed
from traffic.data import feature_text, labeled_text
from traffic.experiment import timed_call, save_json

print("=================================================================")
print("PART 1: Exact 19-Feature Schema & Serialization in traffic/data.py")
print("=================================================================")
data_py = (REPO / "traffic" / "data.py").read_text(encoding="utf-8")
print(data_py[:3500])

# Helper metrics
def calc_metrics(yt, yp):
    yt, yp = np.asarray(yt), np.asarray(yp)
    tp_m = int(np.sum((yt == 1) & (yp == 1)))
    fn_m = int(np.sum((yt == 1) & (yp != 1)))
    fp_m = int(np.sum((yt == 0) & (yp == 1)))
    tp_b = int(np.sum((yt == 0) & (yp == 0)))
    fn_b = int(np.sum((yt == 0) & (yp != 0)))
    fp_b = int(np.sum((yt == 1) & (yp == 0)))
    f1_m = 2 * tp_m / (2 * tp_m + fp_m + fn_m) if (2 * tp_m + fp_m + fn_m) else 0.0
    f1_b = 2 * tp_b / (2 * tp_b + fp_b + fn_b) if (2 * tp_b + fp_b + fn_b) else 0.0
    return {
        "acc": (tp_m + tp_b) / len(yt) * 100.0,
        "macro_f1": 50.0 * (f1_m + f1_b),
        "recall": tp_m / (tp_m + fn_m) * 100.0 if (tp_m + fn_m) else 0.0,
        "fpr": fp_m / (tp_b + fn_b) * 100.0 if (tp_b + fn_b) else 0.0,
    }

# Check existing tabular baseline script if present to match exact encoding/seed
tab_scripts = list((REPO / "scripts").glob("*tree*")) + list((REPO / "scripts").glob("*tabular*")) + list((REPO / "scripts").glob("*baseline*")) + list((REPO / "scripts").glob("*ml*"))
print("\nExisting ML baseline scripts:", [p.name for p in tab_scripts])
for p in tab_scripts:
    print(f"--- {p.name} (head) ---")
    print(p.read_text(encoding="utf-8")[:1800])

print("\n=================================================================")
print("PART 2: Zero-LLM-Cost Hybrid: VLEG-CR Rules + Shallow CART Fallback")
print("=================================================================")

def encode_rows(mem_rows, test_rows):
    cat_cols = ["proto", "service", "state"]
    num_cols = sorted([k for k in mem_rows[0]["features"].keys() if k not in cat_cols])
    # One-hot over union of categories observed in memory (and handle unseen as all-zeros)
    cat_vocab = {c: sorted({str(r["features"][c]) for r in mem_rows}) for c in cat_cols}
    def vec(row):
        f = row["features"]
        out = [float(f[k]) for k in num_cols]
        for c in cat_cols:
            val = str(f[c])
            out.extend([1.0 if val == v else 0.0 for v in cat_vocab[c]])
        return out
    X_tr = np.array([vec(r) for r in mem_rows], dtype=np.float64)
    y_tr = np.array([1 if r["label"] == "MALICIOUS" else 0 for r in mem_rows], dtype=np.int64)
    X_te = np.array([vec(r) for r in test_rows], dtype=np.float64)
    y_te = np.array([1 if r["label"] == "MALICIOUS" else 0 for r in test_rows], dtype=np.int64)
    return X_tr, y_tr, X_te, y_te

def eval_cart_and_hybrid(stage_name, seed_int, min_sup=3, scope="bank"):
    bd = json.loads((FLOW_ROOT / stage_name / "prepared.json").read_text())
    st = json.loads((FLOW_ROOT / stage_name / "checkpoint.json").read_text())
    cr = json.loads((OUT_ROOT / stage_name / "checkpoint_cr.json").read_text())
    seed_rows = {r["id"]: r for r in bd["memory"]}

    X_tr, y_tr, X_te, y_te = encode_rows(bd["memory"], bd["test"])
    clf_d3 = DecisionTreeClassifier(max_depth=3, random_state=seed_int)
    clf_d3.fit(X_tr, y_tr)
    cart_preds = clf_d3.predict(X_te)

    comp = {}
    for m in st["memory"]:
        mid = m["id"]
        src_row = seed_rows[mid]
        txt = cr["memory"][mid]["vleg_cr"]
        if txt is None:
            continue
        atoms, _ = parse_rule(txt, src_row["features"])
        if atoms is None or not safe_applies(atoms, src_row["features"])[0]:
            continue
        lbl = 1 if src_row["label"] == "MALICIOUS" else 0
        pos_sup, ce_cnt = 0, 0
        for hid, r2 in seed_rows.items():
            if hid == mid:
                continue
            if safe_applies(atoms, r2["features"])[0]:
                if (1 if r2["label"] == "MALICIOUS" else 0) == lbl:
                    pos_sup += 1
                else:
                    ce_cnt += 1
        comp[mid] = {"atoms": atoms, "label": lbl, "pos_sup": pos_sup, "ce": ce_cnt}

    hyb_preds = []
    rule_used = 0
    rule_correct = 0
    for idx, r in enumerate(bd["test"]):
        tid = r["id"]
        cands = st["retrieval"][tid]["ids"][:3] if scope == "retrieved" else list(comp.keys())
        votes = {0: 0, 1: 0}
        for mid in cands:
            if mid in comp and comp[mid]["ce"] == 0 and comp[mid]["pos_sup"] >= min_sup:
                if safe_applies(comp[mid]["atoms"], r["features"])[0]:
                    votes[comp[mid]["label"]] += 1
        if scope == "retrieved":
            fired = (votes[0] > 0) != (votes[1] > 0)
            pred_rule = 1 if votes[1] > 0 else 0
        else:
            fired = votes[1] != votes[0]
            pred_rule = 1 if votes[1] > votes[0] else 0

        if fired:
            hyb_preds.append(pred_rule)
            rule_used += 1
            if pred_rule == y_te[idx]:
                rule_correct += 1
        else:
            hyb_preds.append(int(cart_preds[idx]))

    return calc_metrics(y_te, cart_preds), calc_metrics(y_te, hyb_preds), rule_used / len(y_te) * 100.0, (rule_correct / max(1, rule_used) * 100.0)

# Pilot screening for CART hybrid
for scp, sup in [("retrieved", 0), ("bank", 0), ("bank", 2), ("bank", 3), ("bank", 5)]:
    c_m, h_m, ru, r_acc = eval_cart_and_hybrid("pilot_20261012", 20261012, min_sup=sup, scope=scp)
    print(f"Pilot Hybrid [{scp:9s} | sup>={sup}] Macro-F1={h_m['macro_f1']:.2f}% (CART d3={c_m['macro_f1']:.2f}%) | Acc={h_m['acc']:.2f}% | Rec={h_m['recall']:.2f}% | FPR={h_m['fpr']:.2f}% | RuleDirect={ru:.1f}%")

for scp, sup in [("retrieved", 0), ("bank", 0), ("bank", 2), ("bank", 3), ("bank", 5)]:
    c_list, h_list, ru_list, racc_list = [], [], [], []
    for s in SEEDS:
        cm, hm, ru, racc = eval_cart_and_hybrid(f"formal_{s}", s, min_sup=sup, scope=scp)
        c_list.append(cm); h_list.append(hm); ru_list.append(ru); racc_list.append(racc)
    print(
        f"Formal Hybrid [{scp:9s} | sup>={sup}] "
        f"Acc={statistics.mean([x['acc'] for x in h_list]):.2f}+/-{statistics.stdev([x['acc'] for x in h_list]):.2f}% | "
        f"Macro-F1={statistics.mean([x['macro_f1'] for x in h_list]):.2f}+/-{statistics.stdev([x['macro_f1'] for x in h_list]):.2f}% | "
        f"Rec={statistics.mean([x['recall'] for x in h_list]):.2f}+/-{statistics.stdev([x['recall'] for x in h_list]):.2f}% | "
        f"FPR={statistics.mean([x['fpr'] for x in h_list]):.2f}+/-{statistics.stdev([x['fpr'] for x in h_list]):.2f}% | "
        f"RuleDirect={statistics.mean(ru_list):.1f}% (RuleSubsetAcc={statistics.mean(racc_list):.2f}%)"
    )

print("\n=================================================================")
print("PART 3: Extract Complete 4-Stage Case Study from Formal Seed 20261020")
print("=================================================================")
bd0 = json.loads((FLOW_ROOT / "formal_20261020" / "prepared.json").read_text())
st0 = json.loads((FLOW_ROOT / "formal_20261020" / "checkpoint.json").read_text())
cr0 = json.loads((OUT_ROOT / "formal_20261020" / "checkpoint_cr.json").read_text())
rows0 = {r["id"]: r for r in bd0["memory"]}
for m in st0["memory"]:
    mid = m["id"]
    cm = cr0["memory"][mid]
    if cm["label"] == "MALICIOUS" and cm["orig_within_ce"] > 0 and cm["converged_zero_ce"]:
        print("Case Memory ID:", mid)
        print("Raw labeled_text:\n", m["raw"])
        print("j=0 Unverified single_step:\n", m["single_step"])
        print("Stage-A VLEG (source-verified, orig_within_ce =", cm["orig_within_ce"], "):\n", m["vleg"])
        print("Stage-B VLEG-CR history step 0:\n", json.dumps(cm["cr_history"], indent=2))
        print("Final VLEG-CR (CE_seed=0):\n", cm["vleg_cr"])
        break

print("\n=================================================================")
print("PART 4: Label-Informed Unverified Single-Step Control (160 episodes)")
print("=================================================================")
LABEL_GIVEN_GEN = (
    "Analyze a labeled historical network flow. Write a concise, testable local heuristic "
    "of the form 'if <condition> then label = BENIGN/MALICIOUS' supporting its known Historical label. "
    "Do not invent payload contents, topology, host roles or fields not supplied. "
    "A single example cannot establish a general causal rule. Keep the response under 700 characters."
)
ctrl_path = OUT_ROOT / "label_informed_single_step_audit.json"
if ctrl_path.exists():
    ctrl_data = json.loads(ctrl_path.read_text())
else:
    ctrl_data = {}

all_hist = {}
for s in SEEDS:
    bd = json.loads((FLOW_ROOT / f"formal_{s}" / "prepared.json").read_text())
    for r in bd["memory"]:
        all_hist[r["id"]] = r

for s in SEEDS:
    sname = f"formal_{s}"
    bd = json.loads((FLOW_ROOT / sname / "prepared.json").read_text())
    s_rows = {r["id"]: r for r in bd["memory"]}
    if sname not in ctrl_data:
        ctrl_data[sname] = {}
    for idx, r in enumerate(bd["memory"], 1):
        mid = r["id"]
        if mid in ctrl_data[sname]:
            continue
        refl, cost = timed_call(LABEL_GIVEN_GEN, labeled_text(r), "llama3.3:70b", 32768)
        ev = evaluate_rule_on_seed(refl, r, s_rows)
        # Also check parseable & cross-seed CE
        atoms, err = parse_rule(refl, r["features"])
        src_true = safe_applies(atoms, r["features"])[0] if atoms is not None else False
        within_ce = sum(1 for hid, r2 in s_rows.items() if hid != mid and r2["label"] != r["label"] and atoms is not None and safe_applies(atoms, r2["features"])[0])
        cross_ce = sum(1 for hid, r2 in all_hist.items() if hid != mid and r2["label"] != r["label"] and atoms is not None and safe_applies(atoms, r2["features"])[0])
        ctrl_data[sname][mid] = {
            "label": r["label"], "reflection": refl, "parseable": atoms is not None,
            "src_true": src_true, "ok_zero_ce": ev["ok"], "reason": ev["reason"],
            "within_ce": within_ce, "cross_ce": cross_ce, "cost": cost
        }
        if idx % 16 == 0:
            save_json(ctrl_path, ctrl_data)
            print(f"  [Label-Given Control | {sname}] {idx}/32 done", flush=True)
    save_json(ctrl_path, ctrl_data)

recs = [v for s in SEEDS for v in ctrl_data[f"formal_{s}"].values()]
n_parse = sum(1 for x in recs if x["parseable"])
n_src_true = sum(1 for x in recs if x["parseable"] and x["src_true"])
n_zero_within = sum(1 for x in recs if x["ok_zero_ce"])
n_within_gt0 = sum(1 for x in recs if x["parseable"] and x["src_true"] and x["within_ce"] > 0)
n_cross_gt0 = sum(1 for x in recs if x["parseable"] and x["src_true"] and x["cross_ce"] > 0)
print(f"Label-Given Unverified Control Summary (160 episodes):")
print(f"  Parseable syntax:              {n_parse}/160 ({n_parse/1.6:.1f}%)")
print(f"  Antecedent TRUE on source:     {n_src_true}/160 ({n_src_true/1.6:.1f}%)")
print(f"  Strict valid & CE_seed=0:      {n_zero_within}/160 ({n_zero_within/1.6:.1f}%)")
print(f"  Within-seed CE > 0 (of valid): {n_within_gt0}/{n_src_true} ({100*n_within_gt0/max(1,n_src_true):.1f}%)")
print(f"  Cross-seed CE > 0 (of valid):  {n_cross_gt0}/{n_src_true} ({100*n_cross_gt0/max(1,n_src_true):.1f}%)")
