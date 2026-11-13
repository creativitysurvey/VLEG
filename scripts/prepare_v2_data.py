"""Align empirical metrics strictly with fixed Algorithm 1 (outer `if v = 1` gate)
and print symbolic section numbers for lossless 2-page condensation."""
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
from scripts.run_vleg_cr import FLOW_ROOT, OUT_ROOT, SEEDS, safe_applies
from traffic.data import feature_text
from traffic.experiment import INFERENCE, timed_call, save_json

def calc_all(yt, yp):
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
        "invalid": int(np.sum((yp != 0) & (yp != 1))),
    }

def encode_rows(mem_rows, test_rows):
    cat_cols = ["proto", "service", "state"]
    num_cols = sorted([k for k in mem_rows[0]["features"].keys() if k not in cat_cols])
    cat_vocab = {c: sorted({str(r["features"][c]) for r in mem_rows}) for c in cat_cols}
    def vec(row):
        f = row["features"]
        out = [float(f[k]) for k in num_cols]
        for c in cat_cols:
            val = str(f[c])
            out.extend([1.0 if val == v else 0.0 for v in cat_vocab[c]])
        return out
    return (
        np.array([vec(r) for r in mem_rows], dtype=np.float64),
        np.array([1 if r["label"] == "MALICIOUS" else 0 for r in mem_rows], dtype=np.int64),
        np.array([vec(r) for r in test_rows], dtype=np.float64),
        np.array([1 if r["label"] == "MALICIOUS" else 0 for r in test_rows], dtype=np.int64),
    )

# 1. Phase 1 strict v=1 audit (exclude the 2 episodes where Stage-A om["vleg"] is None)
all_hist = {}
for s in SEEDS:
    bd = json.loads((FLOW_ROOT / f"formal_{s}" / "prepared.json").read_text())
    for r in bd["memory"]:
        all_hist[r["id"]] = r

ben_ok, mal_ok, cross_ce_any, mal_zero_cross = 0, 0, 0, 0
extra_calls, extra_toks, extra_secs = [], [], []
unparseable_stage_a = 0

for s in SEEDS:
    stage = f"formal_{s}"
    st = json.loads((FLOW_ROOT / stage / "checkpoint.json").read_text())
    cr = json.loads((OUT_ROOT / stage / "checkpoint_cr.json").read_text())
    om_map = {m["id"]: m for m in st["memory"]}
    c_calls, c_toks, c_secs = 0, 0, 0.0
    for mid, cm in cr["memory"].items():
        om = om_map[mid]
        if om["vleg"] is None:
            continue  # Strictly excluded by outer `if v = 1` gate!
        a_atoms, _ = parse_rule(om["vleg"], all_hist[mid]["features"])
        if a_atoms is None:
            unparseable_stage_a += 1
        c_calls += cm["extra_cost"]["calls"]
        c_toks += cm["extra_cost"]["prompt_tokens"] + cm["extra_cost"]["completion_tokens"]
        c_secs += cm["extra_cost"]["seconds"]
        txt = cm["vleg_cr"]
        if txt is None:
            continue
        src_row = all_hist[mid]
        if src_row["label"] == "BENIGN":
            ben_ok += 1
        else:
            mal_ok += 1
        atoms, _ = parse_rule(txt, src_row["features"])
        c_ce = sum(1 for hid, r2 in all_hist.items() if hid != mid and r2["label"] != src_row["label"] and safe_applies(atoms, r2["features"])[0])
        if c_ce > 0:
            cross_ce_any += 1
        elif src_row["label"] == "MALICIOUS":
            mal_zero_cross += 1
    extra_calls.append(c_calls); extra_toks.append(c_toks); extra_secs.append(c_secs)

tot_ok = ben_ok + mal_ok
print("=== 1. Strict `v=1` Phase 1 Consolidation Audit (158 Stage-A Admitted / 160 Total) ===")
print(f"  Stage-A admitted: 158/160 (unparseable among 158: {unparseable_stage_a} -> 157 parseable)")
print(f"  Stage-B converged CE_seed=0 (of 158): {tot_ok}/158 ({100*tot_ok/158:.1f}%) | of 160: {tot_ok}/160 ({tot_ok/1.6:.1f}%)")
print(f"  Benign CE_seed=0: {ben_ok}/80 ({ben_ok/0.8:.1f}%) | Malicious CE_seed=0: {mal_ok}/78 admitted ({100*mal_ok/78:.1f}%), {mal_ok}/80 total ({mal_ok/0.8:.1f}%)")
print(f"  Unseen Cross-Seed CE > 0: {cross_ce_any}/{tot_ok} ({100*cross_ce_any/tot_ok:.1f}%) | Malicious 0 Cross-Seed CE: {mal_zero_cross}/{mal_ok} ({100*mal_zero_cross/mal_ok:.1f}%)")
print(f"  Extra CR cost/seed (strict v=1): calls={statistics.mean(extra_calls):.1f}, tokens={statistics.mean(extra_toks):.0f}, sec={statistics.mean(extra_secs):.1f}")

# 2. Update prompt-only vleg_cr strictly for the few queries that retrieved a v=0 episode
strict_cache_path = OUT_ROOT / "strict_v1_prompt_patch.json"
strict_patch = json.loads(strict_cache_path.read_text()) if strict_cache_path.exists() else {}

for stage in ["pilot_20261012"] + [f"formal_{s}" for s in SEEDS]:
    bd = json.loads((FLOW_ROOT / stage / "prepared.json").read_text())
    st = json.loads((FLOW_ROOT / stage / "checkpoint.json").read_text())
    cr = json.loads((OUT_ROOT / stage / "checkpoint_cr.json").read_text())
    om_map = {m["id"]: m for m in st["memory"]}
    v0_ids = {mid for mid, m in om_map.items() if m["vleg"] is None}
    if not v0_ids:
        continue
    if stage not in strict_patch:
        strict_patch[stage] = {}
    affected = [r for r in bd["test"] if any(i in v0_ids for i in st["retrieval"][r["id"]]["ids"][:3])]
    need_calls = [r for r in affected if r["id"] not in strict_patch[stage]]
    print(f"  [{stage}] v=0 memory IDs: {len(v0_ids)}, affected queries: {len(affected)}, new LLM calls needed: {len(need_calls)}", flush=True)
    for r in need_calls:
        tid = r["id"]
        rids = st["retrieval"][tid]["ids"][:3]
        blocks = []
        for mid in rids:
            if mid in v0_ids:
                continue  # Strict v=1 exclusion!
            refl = cr["memory"][mid]["vleg_cr"]
            blocks.append(om_map[mid]["raw"] + ("\nReflection: " + refl if refl is not None else ""))
        prompt_str = "\n\n".join(blocks) + "\n\nTarget flow features: " + feature_text(r)
        reply, cost = timed_call(INFERENCE, prompt_str, "llama3.3:70b", 32768)
        pred = reply.strip().upper()
        if pred not in ("BENIGN", "MALICIOUS"):
            pred = "INVALID"
        strict_patch[stage][tid] = {"prediction": pred, "reply": reply, **cost}
    save_json(strict_cache_path, strict_patch)

# 3. Compute complete strict v=1 metrics across Pilot and 5 Formal seeds
def eval_all_methods(stage, seed_int):
    bd = json.loads((FLOW_ROOT / stage / "prepared.json").read_text())
    st = json.loads((FLOW_ROOT / stage / "checkpoint.json").read_text())
    pr = json.loads((FLOW_ROOT / stage / "predictions.json").read_text())
    cr = json.loads((OUT_ROOT / stage / "checkpoint_cr.json").read_text())
    seed_rows = {r["id"]: r for r in bd["memory"]}
    om_map = {m["id"]: m for m in st["memory"]}
    pmap = {(r["method"], r["id"]): (1 if r["prediction"] == "MALICIOUS" else (0 if r["prediction"] == "BENIGN" else -1)) for r in pr}
    for k, r in cr["predictions"].items():
        if k.startswith("vleg_cr:"):
            tid = r["id"]
            if stage in strict_patch and tid in strict_patch[stage]:
                p_str = strict_patch[stage][tid]["prediction"]
            else:
                p_str = r["prediction"]
            pmap[("vleg_cr_strict", tid)] = 1 if p_str == "MALICIOUS" else (0 if p_str == "BENIGN" else -1)

    X_tr, y_tr, X_te, y_te = encode_rows(bd["memory"], bd["test"])
    clf_d3 = DecisionTreeClassifier(max_depth=3, random_state=seed_int).fit(X_tr, y_tr)
    cart_preds = clf_d3.predict(X_te)

    comp = {}
    for m in st["memory"]:
        mid = m["id"]
        if m["vleg"] is None:
            continue  # Strict v=1 gate!
        src_row = seed_rows[mid]
        txt = cr["memory"][mid]["vleg_cr"]
        if txt is None:
            continue
        atoms, _ = parse_rule(txt, src_row["features"])
        if atoms is None or not safe_applies(atoms, src_row["features"])[0]:
            continue
        lbl = 1 if src_row["label"] == "MALICIOUS" else 0
        pos_sup = sum(1 for hid, r2 in seed_rows.items() if hid != mid and (1 if r2["label"] == "MALICIOUS" else 0) == lbl and safe_applies(atoms, r2["features"])[0])
        comp[mid] = {"atoms": atoms, "label": lbl, "pos_sup": pos_sup}

    yt = [1 if r["label"] == "MALICIOUS" else 0 for r in bd["test"]]
    y_raw = [pmap[("raw", r["id"])] for r in bd["test"]]
    y_cr_prompt = [pmap[("vleg_cr_strict", r["id"])] for r in bd["test"]]

    def run_ns(scope, min_sup, fallback):
        yp, used, sub_ok = [], 0, 0
        for idx, r in enumerate(bd["test"]):
            tid = r["id"]
            cands = st["retrieval"][tid]["ids"][:3] if scope == "retrieved" else list(comp.keys())
            votes = {0: 0, 1: 0}
            for mid in cands:
                if mid in comp and comp[mid]["pos_sup"] >= min_sup and safe_applies(comp[mid]["atoms"], r["features"])[0]:
                    votes[comp[mid]["label"]] += 1
            fired = ((votes[0] > 0) != (votes[1] > 0)) if scope == "retrieved" else (votes[1] != votes[0])
            if fired:
                used += 1
                if fallback == "active_prompt":
                    pred = pmap[("vleg_cr_strict", tid)]
                else:
                    pred = 1 if votes[1] > votes[0] else 0
                if pred == yt[idx]:
                    sub_ok += 1
                yp.append(pred)
            else:
                yp.append(int(cart_preds[idx]) if fallback == "cart_d3" else pmap[("raw", tid)])
        return yp, used / len(yt) * 100.0, sub_ok / max(1, used) * 100.0

    # Non-firing benign diagnostic
    nf_ben, nf_cr_fp, nf_raw_fp = 0, 0, 0
    r_cnts = []
    for r in bd["test"]:
        tid = r["id"]
        rids = [i for i in st["retrieval"][tid]["ids"][:3] if om_map[i]["vleg"] is not None]
        r_cnts.append(sum(1 for i in rids if i in comp))
        if r["label"] == "BENIGN":
            mal_fired = any(i in comp and comp[i]["label"] == 1 and safe_applies(comp[i]["atoms"], r["features"])[0] for i in rids)
            if not mal_fired:
                nf_ben += 1
                if pmap[("vleg_cr_strict", tid)] == 1: nf_cr_fp += 1
                if pmap[("raw", tid)] == 1: nf_raw_fp += 1

    return {
        "yt": yt, "raw": y_raw, "cr_prompt": y_cr_prompt,
        "act_gate": run_ns("retrieved", 0, "active_prompt"),
        "ns_top3": run_ns("retrieved", 0, "raw"),
        "bank_0": run_ns("bank", 0, "raw"),
        "bank_3": run_ns("bank", 3, "raw"),
        "ns_cart": run_ns("retrieved", 0, "cart_d3"),
        "bank3_cart": run_ns("bank", 3, "cart_d3"),
        "bank5_cart": run_ns("bank", 5, "cart_d3"),
        "nf": (nf_ben, nf_cr_fp, nf_raw_fp),
        "rule_disp": (sum(1 for c in r_cnts if c >= 1) / len(r_cnts) * 100.0, statistics.mean(r_cnts)),
    }

pilot_res = eval_all_methods("pilot_20261012", 20261012)
formal_res = [eval_all_methods(f"formal_{s}", s) for s in SEEDS]

tot_nf_ben = sum(x["nf"][0] for x in formal_res)
tot_nf_cr = sum(x["nf"][1] for x in formal_res)
tot_nf_raw = sum(x["nf"][2] for x in formal_res)
print(f"\n=== 2. Strict `v=1` Non-Firing Benign Diagnostic ===")
print(f"  Benign evals w/ 0 firing mal rules: {tot_nf_ben} | VLEG-CR FPR: {100*tot_nf_cr/tot_nf_ben:.2f}% vs Raw FPR: {100*tot_nf_raw/tot_nf_ben:.2f}%")
print(f"  Prompt rule display: >=1 rule = {statistics.mean([x['rule_disp'][0] for x in formal_res]):.2f}%, mean = {statistics.mean([x['rule_disp'][1] for x in formal_res]):.2f} rules/query")

print("\n=== 3. Strict `v=1` Pilot & Formal 5-Seed Summary ===")
rng = np.random.default_rng(20261013)
yt_arr = np.array([x["yt"] for x in formal_res])
yr_arr = np.array([x["raw"] for x in formal_res])

for key, title in [
    ("cr_prompt", "VLEG-CR (prompt-only)"),
    ("act_gate", "VLEG-CR Active-Prompt Gate"),
    ("ns_top3", "VLEG-CR-NS (top-3, sup>=0)"),
    ("bank_0", "VLEG-CR-Bank (sup>=0)"),
    ("bank_3", "VLEG-CR-Bank (Pilot sup>=3)"),
    ("ns_cart", "VLEG-CR-NS + CART(d=3)"),
    ("bank3_cart", "VLEG-CR-Bank(sup>=3) + CART(d=3)"),
    ("bank5_cart", "VLEG-CR-Bank(sup>=5) + CART(d=3)"),
]:
    p_yp = pilot_res[key] if key == "cr_prompt" else pilot_res[key][0]
    p_f1 = calc_all(pilot_res["yt"], p_yp)["macro_f1"]
    f_yps = [x[key] if key == "cr_prompt" else x[key][0] for x in formal_res]
    f_used = [0.0 if key == "cr_prompt" else x[key][1] for x in formal_res]
    f_sub = [0.0 if key == "cr_prompt" else x[key][2] for x in formal_res]
    m_list = [calc_all(x["yt"], yp) for x, yp in zip(formal_res, f_yps)]
    r_list = [calc_all(x["yt"], x["raw"]) for x in formal_res]
    yp_arr = np.array(f_yps)
    boots = [np.mean([calc_all(yt_arr[i, idx], yp_arr[i, idx])["macro_f1"] - calc_all(yt_arr[i, idx], yr_arr[i, idx])["macro_f1"] for i in range(5)]) for idx in (rng.integers(0, 1000, size=1000) for _ in range(2000))]
    ci = np.percentile(boots, [2.5, 97.5])
    inv = sum(m["invalid"] for m in m_list)
    print(
        f"  {title:32s} | Pilot F1={p_f1:.2f}% | "
        f"Acc={statistics.mean([m['acc'] for m in m_list]):.2f}+/-{statistics.stdev([m['acc'] for m in m_list]):.2f} | "
        f"F1={statistics.mean([m['macro_f1'] for m in m_list]):.2f}+/-{statistics.stdev([m['macro_f1'] for m in m_list]):.2f} | "
        f"Rec={statistics.mean([m['recall'] for m in m_list]):.2f}+/-{statistics.stdev([m['recall'] for m in m_list]):.2f} | "
        f"FPR={statistics.mean([m['fpr'] for m in m_list]):.2f}+/-{statistics.stdev([m['fpr'] for m in m_list]):.2f} | "
        f"Used={statistics.mean(f_used):.1f}+/-{statistics.stdev(f_used):.1f}% (SubAcc={statistics.mean(f_sub):.2f}%) | "
        f"dRaw={statistics.mean([a['macro_f1']-b['macro_f1'] for a,b in zip(m_list,r_list)]):+.2f} [{ci[0]:+.2f},{ci[1]:+.2f}] | Inv={inv}"
    )

print("\n=== 4. Symbolic Section Source from Electronics_submission/main.tex ===")
etex = (REPO / "Electronics_submission" / "main.tex").read_text(encoding="utf-8")
s_idx = etex.find(r"\subsection{Controlled Memory-Strategy Comparison")
e_idx = etex.find(r"\subsection{Public-Flow Benchmark on UNSW-NB15")
print(etex[s_idx:e_idx])
