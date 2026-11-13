"""Fast KV-cache-ordered formal 5-seed inference & statistical summary for VLEG-CR."""
import json
import statistics
import sys
import time
import numpy as np
from pathlib import Path

REPO_ROOT = Path("/data/liwei/VLEG")
sys.path.insert(0, str(REPO_ROOT))

from traffic.data import feature_text
from traffic.experiment import INFERENCE, timed_call, metrics, save_json
from scripts.run_vleg_cr import FLOW_ROOT, OUT_ROOT, SEEDS


def run_seed_fast(seed, model="llama3.3:70b", num_ctx=32768, context_chars=18000):
    stage_name = f"formal_{seed}"
    src_dir = FLOW_ROOT / stage_name
    out_dir = OUT_ROOT / stage_name
    ckpt_path = out_dir / "checkpoint_cr.json"

    bundle = json.loads((src_dir / "prepared.json").read_text())
    orig_state = json.loads((src_dir / "checkpoint.json").read_text())
    orig_preds = json.loads((src_dir / "predictions.json").read_text())
    cr_state = json.loads(ckpt_path.read_text())

    orig_mem_by_id = {m["id"]: m for m in orig_state["memory"]}
    orig_pred_map = {(r["method"], r["id"]): r for r in orig_preds}

    def build_cr_prompt(selected_ids, target_row):
        blocks = []
        n_rules = 0
        for mid in selected_ids:
            om = orig_mem_by_id[mid]
            cm = cr_state["memory"][mid]
            if om["vleg"] is None and cm["vleg_cr"] is None:
                continue
            refl = cm["vleg_cr"]
            if refl is not None:
                n_rules += 1
            blocks.append(om["raw"] + ("\nReflection: " + refl if refl is not None else ""))
        target_str = "\n\nTarget flow features: " + feature_text(target_row)
        return "\n\n".join(blocks) + target_str, len(blocks), n_rules

    reused = 0
    pending_calls = []
    rule_counts = []

    # Pass 1: Instant byte-exact reuse check against archived vleg / raw_on_verified / raw
    for row in bundle["test"]:
        tid = row["id"]
        pkey = f"vleg_cr:{tid}"
        rids = orig_state["retrieval"][tid]["ids"]
        prompt_str, eff_n, n_rules = build_cr_prompt(rids, row)
        rule_counts.append(n_rules)

        if pkey in cr_state["predictions"]:
            continue

        usable_vleg = [orig_mem_by_id[k] for k in rids if orig_mem_by_id[k]["vleg"] is not None]
        target_str = "\n\nTarget flow features: " + feature_text(row)
        vleg_prompt = "\n\n".join(m["raw"] + "\nReflection: " + m["vleg"] for m in usable_vleg) + target_str
        raw_adm_prompt = "\n\n".join(m["raw"] for m in usable_vleg) + target_str
        raw_full_prompt = "\n\n".join(orig_mem_by_id[k]["raw"] for k in rids) + target_str

        if prompt_str == vleg_prompt:
            arc = dict(orig_pred_map[("vleg", tid)])
            arc["method"] = "vleg_cr"
            arc["reused_from"] = "vleg"
            cr_state["predictions"][pkey] = arc
            reused += 1
        elif prompt_str == raw_adm_prompt:
            arc = dict(orig_pred_map[("raw_on_verified", tid)])
            arc["method"] = "vleg_cr"
            arc["reused_from"] = "raw_on_verified"
            cr_state["predictions"][pkey] = arc
            reused += 1
        elif prompt_str == raw_full_prompt:
            arc = dict(orig_pred_map[("raw", tid)])
            arc["method"] = "vleg_cr"
            arc["reused_from"] = "raw"
            cr_state["predictions"][pkey] = arc
            reused += 1
        else:
            pending_calls.append((tuple(rids), row, prompt_str, eff_n))

    save_json(ckpt_path, cr_state)

    # Sort pending queries lexicographically by retrieved memory IDs to maximize llama-server slot-0 KV-cache hits
    pending_calls.sort(key=lambda x: x[0])
    total_pending = len(pending_calls)
    q_with_any_rule = sum(1 for c in rule_counts if c >= 1)
    q_with_all3_rules = sum(1 for c in rule_counts if c == 3)
    print(
        f"\n=== [{stage_name}] Rule display coverage: >=1 rule on {q_with_any_rule}/1000 ({q_with_any_rule/10:.1f}%), "
        f"all 3 rules on {q_with_all3_rules}/1000 ({q_with_all3_rules/10:.1f}%), mean={statistics.mean(rule_counts):.2f} rules/query ===",
        flush=True,
    )
    print(f"=== [{stage_name}] Instant exact-prompt reused: {reused} | Pending LLM calls (KV-sorted): {total_pending} ===", flush=True)

    t0 = time.perf_counter()
    for idx, (_, row, prompt_str, eff_n) in enumerate(pending_calls, 1):
        tid = row["id"]
        pkey = f"vleg_cr:{tid}"
        if len(prompt_str) > context_chars:
            raise ValueError(f"Prompt exceeds context_chars ({len(prompt_str)} > {context_chars})")
        reply, cost = timed_call(INFERENCE, prompt_str, model, num_ctx)
        pred = reply.strip().upper()
        if pred not in ("BENIGN", "MALICIOUS"):
            pred = "INVALID"
        cr_state["predictions"][pkey] = {
            "id": tid, "method": "vleg_cr", "truth": row["label"],
            "attack_category": row["attack_category"], "group": row.get("group"),
            "prediction": pred, "reply": reply,
            "retrieved_ids": orig_state["retrieval"][tid]["ids"],
            "effective_examples": eff_n, "context_characters": len(prompt_str),
            "retrieval_seconds": orig_state["retrieval"][tid]["seconds"],
            "reused_from": None, **cost,
        }
        if idx % 20 == 0 or idx == total_pending:
            save_json(ckpt_path, cr_state)
        if idx % 50 == 0 or idx == total_pending:
            el = time.perf_counter() - t0
            rate = el / idx
            eta_m = (total_pending - idx) * rate / 60.0
            print(f"  [{stage_name}] {idx}/{total_pending} LLM calls ({rate:.2f}s/call, elapsed={el/60:.1f}m, ETA={eta_m:.1f}m)", flush=True)

    # Evaluate in exact original bundle["test"] order
    recs = [cr_state["predictions"][f"vleg_cr:{r['id']}"] for r in bundle["test"]]
    m_cr = metrics(recs)
    m_raw = metrics([orig_pred_map[("raw", r["id"])] for r in bundle["test"]])
    m_vleg = metrics([orig_pred_map[("vleg", r["id"])] for r in bundle["test"]])
    delta_raw = (m_cr["macro_f1"] - m_raw["macro_f1"]) * 100
    delta_vleg = (m_cr["macro_f1"] - m_vleg["macro_f1"]) * 100
    print(
        f">>> [{stage_name} DONE] VLEG-CR Macro-F1={m_cr['macro_f1']*100:.2f}% "
        f"(vs Raw {m_raw['macro_f1']*100:.2f}% [{delta_raw:+.2f} pp], vs VLEG {m_vleg['macro_f1']*100:.2f}% [{delta_vleg:+.2f} pp]) | "
        f"Acc={m_cr['accuracy']*100:.2f}% | Recall={m_cr['attack_recall']*100:.2f}% | FPR={m_cr['false_positive_rate']*100:.2f}%",
        flush=True,
    )
    return m_cr, m_raw, m_vleg, recs, [orig_pred_map[("raw", r["id"])] for r in bundle["test"]], rule_counts


def main():
    cr_list, raw_list, vleg_list = [], [], []
    all_cr_recs, all_raw_recs = [], []
    any_rule_pcts, all3_rule_pcts, mean_rules = [], [], []

    for s in SEEDS:
        m_cr, m_raw, m_vleg, recs_cr, recs_raw, r_cnts = run_seed_fast(s)
        cr_list.append(m_cr)
        raw_list.append(m_raw)
        vleg_list.append(m_vleg)
        all_cr_recs.append(recs_cr)
        all_raw_recs.append(recs_raw)
        any_rule_pcts.append(sum(1 for c in r_cnts if c >= 1) / 10.0)
        all3_rule_pcts.append(sum(1 for c in r_cnts if c == 3) / 10.0)
        mean_rules.append(statistics.mean(r_cnts))

    # Paired cluster bootstrap 95% CI across the 1,000 test fingerprints (synchronized across 5 seeds)
    rng = np.random.default_rng(20261013)
    n_q = 1000
    cr_truth = np.array([[1 if r["truth"] == "MALICIOUS" else 0 for r in recs] for recs in all_cr_recs])
    cr_pred = np.array([[1 if r["prediction"] == "MALICIOUS" else (0 if r["prediction"] == "BENIGN" else -1) for r in recs] for recs in all_cr_recs])
    raw_pred = np.array([[1 if r["prediction"] == "MALICIOUS" else (0 if r["prediction"] == "BENIGN" else -1) for r in recs] for recs in all_raw_recs])

    def fast_macro_f1(yt, yp):
        tp_m = np.sum((yt == 1) & (yp == 1))
        fn_m = np.sum((yt == 1) & (yp != 1))
        fp_m = np.sum((yt == 0) & (yp == 1))
        tp_b = np.sum((yt == 0) & (yp == 0))
        fn_b = np.sum((yt == 0) & (yp != 0))
        fp_b = np.sum((yt == 1) & (yp == 0))
        f1_m = 2 * tp_m / (2 * tp_m + fp_m + fn_m) if (2 * tp_m + fp_m + fn_m) else 0.0
        f1_b = 2 * tp_b / (2 * tp_b + fp_b + fn_b) if (2 * tp_b + fp_b + fn_b) else 0.0
        return 50.0 * (f1_m + f1_b)

    boot_diffs = []
    for _ in range(2000):
        idx = rng.integers(0, n_q, size=n_q)
        d = np.mean([fast_macro_f1(cr_truth[s, idx], cr_pred[s, idx]) - fast_macro_f1(cr_truth[s, idx], raw_pred[s, idx]) for s in range(5)])
        boot_diffs.append(d)
    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])

    print("\n======================================================================")
    print("         FINAL 5-SEED FORMAL BENCHMARK SUMMARY (VLEG-CR)              ")
    print("======================================================================")
    for label, arr in [("Raw flow retrieval", raw_list), ("Standard VLEG", vleg_list), ("VLEG-CR (Ours)", cr_list)]:
        acc = [x["accuracy"] * 100 for x in arr]
        f1 = [x["macro_f1"] * 100 for x in arr]
        rec = [x["attack_recall"] * 100 for x in arr]
        fpr = [x["false_positive_rate"] * 100 for x in arr]
        inv = sum(x["invalid_predictions"] for x in arr)
        print(
            f"  {label:20s} | Acc: {statistics.mean(acc):.2f} +/- {statistics.stdev(acc):.2f} | "
            f"Macro-F1: {statistics.mean(f1):.2f} +/- {statistics.stdev(f1):.2f} | "
            f"Recall: {statistics.mean(rec):.2f} +/- {statistics.stdev(rec):.2f} | "
            f"FPR: {statistics.mean(fpr):.2f} +/- {statistics.stdev(fpr):.2f} | Invalid: {inv}"
        )
    deltas = [(a["macro_f1"] - b["macro_f1"]) * 100 for a, b in zip(cr_list, raw_list)]
    print(f"\n  Paired Macro-F1 Delta (VLEG-CR - Raw): {statistics.mean(deltas):+.2f} pp (95% Bootstrap CI: [{ci_lo:+.2f}, {ci_hi:+.2f}] pp)")
    print(f"  Per-seed deltas (pp): {[round(d, 2) for d in deltas]}")
    print(f"  Rule coverage across queries: >=1 rule = {statistics.mean(any_rule_pcts):.2f}%, all 3 rules = {statistics.mean(all3_rule_pcts):.2f}%, mean = {statistics.mean(mean_rules):.2f}/3.00")

    summary_out = {
        "vleg_cr_mean_macro_f1": statistics.mean([x["macro_f1"] * 100 for x in cr_list]),
        "vleg_cr_sd_macro_f1": statistics.stdev([x["macro_f1"] * 100 for x in cr_list]),
        "vleg_cr_mean_acc": statistics.mean([x["accuracy"] * 100 for x in cr_list]),
        "vleg_cr_sd_acc": statistics.stdev([x["accuracy"] * 100 for x in cr_list]),
        "vleg_cr_mean_recall": statistics.mean([x["attack_recall"] * 100 for x in cr_list]),
        "vleg_cr_sd_recall": statistics.stdev([x["attack_recall"] * 100 for x in cr_list]),
        "vleg_cr_mean_fpr": statistics.mean([x["false_positive_rate"] * 100 for x in cr_list]),
        "vleg_cr_sd_fpr": statistics.stdev([x["false_positive_rate"] * 100 for x in cr_list]),
        "per_seed_macro_f1_delta_vs_raw": deltas,
        "paired_bootstrap_95ci_vs_raw": [float(ci_lo), float(ci_hi)],
        "queries_with_at_least_1_rule_pct": statistics.mean(any_rule_pcts),
        "queries_with_all_3_rules_pct": statistics.mean(all3_rule_pcts),
        "mean_rules_per_query": statistics.mean(mean_rules),
        "by_seed": [cr_list[i] for i in range(5)],
    }
    save_json(OUT_ROOT / "formal_vleg_cr_summary.json", summary_out)
    print(f"  Saved complete summary to {OUT_ROOT / 'formal_vleg_cr_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
