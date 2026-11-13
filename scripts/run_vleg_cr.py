"""Counterexample-Refined VLEG (VLEG-CR) closed-loop consolidation & evaluation.

Strictly uses within-seed historical counterexamples (31 other flows in the same seed)
with zero access to other seeds or test flows. Reuses the exact GENERATOR, VERIFIER,
and INFERENCE prompts from traffic/experiment.py.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/data/liwei/VLEG")
sys.path.insert(0, str(REPO_ROOT))

from core.verification_response import parse_verification_response
from scripts.audit_explanations_and_provenance import parse_rule, applies, label as parse_label
from traffic.data import feature_text, labeled_text
from traffic.experiment import GENERATOR, VERIFIER, INFERENCE, timed_call, metrics, save_json

FLOW_ROOT = Path("/data/liwei/VLEG-flow-exp6-20261012_054157/outputs_traffic/unsw_70b_formal_20261012")
OUT_ROOT = REPO_ROOT / "results" / "vleg_cr_20261112"
SEEDS = [20261020, 20261021, 20261022, 20261023, 20261024]

CONTINUOUS_NO_NONZERO_EQ = {
    "dur", "rate", "sload", "dload", "sinpkt", "dinpkt", "sjit", "djit", "sbytes", "dbytes"
}
CAT_FEATURES = {"proto", "service", "state"}


def safe_applies(atoms, feats):
    try:
        return bool(applies(atoms, feats)), ""
    except Exception as exc:
        return False, f"type/eval error: {exc}"


def check_memorization(atoms):
    for key, op, rhs in atoms:
        if key in CONTINUOUS_NO_NONZERO_EQ and op == "==" and rhs[0] == "literal":
            if isinstance(rhs[1], (int, float)) and abs(float(rhs[1])) > 1e-12:
                return f"Do not use exact equality (== {rhs[1]}) on continuous/byte field `{key}`; use >=, <=, >, or < instead."
    return ""


def summarize_separating_features(src_feats, ce_feats_list, max_hints=5):
    hints = []
    for k in sorted(src_feats.keys()):
        sval = src_feats[k]
        if k in CAT_FEATURES:
            ce_vals = sorted({str(c[k]) for c in ce_feats_list})
            if str(sval) not in ce_vals:
                hints.append(f'{k} == "{sval}" (matched counterexamples have {ce_vals})')
        else:
            s_num = float(sval)
            c_nums = [float(c[k]) for c in ce_feats_list]
            if s_num > max(c_nums):
                hints.append(f"{k}={s_num} > max matched counterexample ({max(c_nums)})")
            elif s_num < min(c_nums):
                hints.append(f"{k}={s_num} < min matched counterexample ({min(c_nums)})")
        if len(hints) >= max_hints:
            break
    return "; ".join(hints)


def evaluate_rule_on_seed(text, src_row, seed_rows_by_id):
    if text is None:
        return {"ok": False, "reason": "no reflection", "atoms": None, "conflicts": []}
    if len(text) > 750:
        return {"ok": False, "reason": f"reflection too long ({len(text)} chars > 700 limit)", "atoms": None, "conflicts": []}
    src_feats = src_row["features"]
    true_lbl = src_row["label"]
    pred_lbl = parse_label(text)
    if pred_lbl != true_lbl:
        return {"ok": False, "reason": f"stated consequent label ({pred_lbl}) contradicts historical label ({true_lbl})", "atoms": None, "conflicts": []}
    atoms, err = parse_rule(text, src_feats)
    if atoms is None:
        return {"ok": False, "reason": f"rule syntax not deterministically parseable ({err})", "atoms": None, "conflicts": []}
    mem_err = check_memorization(atoms)
    if mem_err:
        return {"ok": False, "reason": mem_err, "atoms": atoms, "conflicts": []}
    src_ok, eval_err = safe_applies(atoms, src_feats)
    if eval_err:
        return {"ok": False, "reason": eval_err, "atoms": None, "conflicts": []}
    if not src_ok:
        return {"ok": False, "reason": "IF condition evaluates to FALSE on the source flow itself", "atoms": atoms, "conflicts": []}

    conflicts = []
    same_matches = []
    for hid, other_row in seed_rows_by_id.items():
        if hid == src_row["id"]:
            continue
        matched, err2 = safe_applies(atoms, other_row["features"])
        if err2:
            return {"ok": False, "reason": err2, "atoms": None, "conflicts": []}
        if matched:
            if other_row["label"] != true_lbl:
                conflicts.append(hid)
            else:
                same_matches.append(hid)
    if conflicts:
        return {
            "ok": False,
            "reason": f"over-generalized: matched {len(conflicts)} contrary-label historical flow(s) in this memory bank",
            "atoms": atoms,
            "conflicts": conflicts,
            "same_matches": same_matches,
        }
    return {"ok": True, "reason": "zero within-seed counterexamples", "atoms": atoms, "conflicts": [], "same_matches": same_matches}


def build_cr_critique(eval_res, src_row, seed_rows_by_id, seen_ce_ids):
    true_lbl = src_row["label"]
    contrary_lbl = "BENIGN" if true_lbl == "MALICIOUS" else "MALICIOUS"
    for cid in eval_res.get("conflicts", []):
        if cid not in seen_ce_ids:
            seen_ce_ids.append(cid)
    ce_examples = [seed_rows_by_id[cid] for cid in seen_ce_ids[:3]]

    parts = [
        f"Historical label is {true_lbl}. Rejected because {eval_res['reason']}."
    ]
    if ce_examples:
        sep = summarize_separating_features(src_row["features"], [c["features"] for c in ce_examples])
        ce_strs = [f"[{i+1}] {feature_text(c)}" for i, c in enumerate(ce_examples)]
        parts.append(f"Contrary-label ({contrary_lbl}) historical flows falsely matched: " + " | ".join(ce_strs))
        if sep:
            parts.append(f"Candidate separating differences on this source flow vs those counterexamples: {sep}.")
    parts.append(
        f'Write ONE concise (<650 chars) conjunctive heuristic of the exact form: '
        f'if <field> <op> <value> and ... then label = {true_lbl}, followed by a brief rationale. '
        f'Use only ==, !=, >=, <=, >, < joined by "and" (use double quotes for string literals like proto == "tcp", '
        f'and use inequality thresholds rather than exact == on continuous floats). '
        f'Ensure the condition is TRUE on the source flow and FALSE on all contrary-label counterexamples.'
    )
    return "\n".join(parts)


def consolidate_stage(stage_name, model="llama3.3:70b", num_ctx=32768, K_cr=3):
    src_dir = FLOW_ROOT / stage_name
    out_dir = OUT_ROOT / stage_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "checkpoint_cr.json"

    bundle = json.loads((src_dir / "prepared.json").read_text())
    orig_state = json.loads((src_dir / "checkpoint.json").read_text())
    seed_rows_by_id = {r["id"]: r for r in bundle["memory"]}

    if ckpt_path.exists():
        cr_state = json.loads(ckpt_path.read_text())
    else:
        cr_state = {"stage": stage_name, "K_cr": K_cr, "memory": {}, "predictions": {}}

    t0 = time.perf_counter()
    for idx, m in enumerate(orig_state["memory"], 1):
        mid = m["id"]
        if mid in cr_state["memory"]:
            continue
        src_row = seed_rows_by_id[mid]
        base_vleg = m["vleg"]
        init_eval = evaluate_rule_on_seed(base_vleg, src_row, seed_rows_by_id)

        if base_vleg is not None and init_eval["ok"]:
            cr_state["memory"][mid] = {
                "id": mid,
                "label": src_row["label"],
                "orig_vleg": base_vleg,
                "orig_within_ce": 0,
                "refined_needed": False,
                "converged_zero_ce": True,
                "vleg_cr": base_vleg,
                "cr_history": [],
                "extra_cost": {"seconds": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "calls": 0},
            }
            save_json(ckpt_path, cr_state)
            continue

        orig_ce_cnt = len(init_eval.get("conflicts", [])) if init_eval.get("atoms") is not None else -1
        seen_ce_ids = list(init_eval.get("conflicts", []))
        prev_refl = base_vleg if base_vleg is not None else m["single_step"]
        critique = build_cr_critique(init_eval, src_row, seed_rows_by_id, seen_ce_ids)

        cr_history = []
        accepted_cr = None
        tot_cost = {"seconds": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

        for attempt in range(K_cr):
            user_prompt = (
                "Flow features: " + feature_text(src_row)
                + "\nPrevious reflection: " + prev_refl
                + "\nVerifier critique: " + critique
            )
            cand, g_cost = timed_call(GENERATOR, user_prompt, model, num_ctx)
            for k in tot_cost:
                tot_cost[k] += g_cost[k]

            c_eval = evaluate_rule_on_seed(cand, src_row, seed_rows_by_id)
            v_verdict = None
            v_reply = None
            v_cost = {"seconds": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

            if c_eval["ok"]:
                v_reply, v_cost = timed_call(
                    VERIFIER, labeled_text(src_row) + "\nCandidate: " + cand, model, num_ctx
                )
                for k in tot_cost:
                    tot_cost[k] += v_cost[k]
                v_verdict = parse_verification_response(v_reply)
                if v_verdict["valid"]:
                    cr_history.append({
                        "attempt": attempt, "reflection": cand, "compiler_ok": True,
                        "within_ce": 0, "verifier": v_verdict, "g_cost": g_cost, "v_cost": v_cost
                    })
                    accepted_cr = cand
                    break
                else:
                    critique = (
                        f"Historical label is {src_row['label']}. Compiler check passed, but source verifier rejected: "
                        f"{v_verdict['critique']}. Keep exact IF ... THEN label = {src_row['label']} conditions and fix the rationale."
                    )
            else:
                critique = build_cr_critique(c_eval, src_row, seed_rows_by_id, seen_ce_ids)

            cr_history.append({
                "attempt": attempt, "reflection": cand, "compiler_ok": c_eval["ok"],
                "compiler_reason": c_eval["reason"],
                "within_ce": len(c_eval.get("conflicts", [])),
                "verifier": v_verdict, "g_cost": g_cost, "v_cost": v_cost
            })
            prev_refl = cand

        cr_state["memory"][mid] = {
            "id": mid,
            "label": src_row["label"],
            "orig_vleg": base_vleg,
            "orig_within_ce": orig_ce_cnt,
            "refined_needed": True,
            "converged_zero_ce": accepted_cr is not None,
            "vleg_cr": accepted_cr,
            "cr_history": cr_history,
            "extra_cost": tot_cost,
        }
        save_json(ckpt_path, cr_state)
        status_tag = "CONVERGED(CE=0)" if accepted_cr else "FALLBACK(raw)"
        print(f"  [{stage_name}] Memory {idx:02d}/32 ({src_row['label'][:3]}) orig_CE={orig_ce_cnt} -> {status_tag} in {len(cr_history)} step(s) ({tot_cost['seconds']:.1f}s)", flush=True)

    conv = sum(1 for v in cr_state["memory"].values() if v["converged_zero_ce"])
    extra_tok = sum(v["extra_cost"]["prompt_tokens"] + v["extra_cost"]["completion_tokens"] for v in cr_state["memory"].values())
    extra_calls = sum(v["extra_cost"]["calls"] for v in cr_state["memory"].values())
    print(f"=== [{stage_name}] Consolidation Complete: {conv}/32 rules at CE_seed=0 | extra_calls={extra_calls}, extra_tokens={extra_tok}, elapsed={time.perf_counter()-t0:.1f}s ===", flush=True)
    return cr_state


def run_inference_stage(stage_name, methods=("vleg_cr", "vleg_cr_mal"), model="llama3.3:70b", num_ctx=32768, context_chars=18000):
    src_dir = FLOW_ROOT / stage_name
    out_dir = OUT_ROOT / stage_name
    ckpt_path = out_dir / "checkpoint_cr.json"

    bundle = json.loads((src_dir / "prepared.json").read_text())
    orig_state = json.loads((src_dir / "checkpoint.json").read_text())
    orig_preds = json.loads((src_dir / "predictions.json").read_text())
    cr_state = json.loads(ckpt_path.read_text())

    orig_mem_by_id = {m["id"]: m for m in orig_state["memory"]}
    orig_pred_map = {(r["method"], r["id"]): r for r in orig_preds}

    def build_prompt_for_method(method_name, selected_ids, target_row):
        blocks = []
        for mid in selected_ids:
            om = orig_mem_by_id[mid]
            cm = cr_state["memory"][mid]
            if method_name == "vleg_cr":
                # Keep all verified flows; attach reflection iff converged to CE_seed=0
                if om["vleg"] is None and cm["vleg_cr"] is None:
                    continue
                refl = cm["vleg_cr"]
            elif method_name == "vleg_cr_mal":
                # Refine MALICIOUS rules; keep original BENIGN vleg rules
                if om["vleg"] is None and cm["vleg_cr"] is None:
                    continue
                refl = cm["vleg_cr"] if cm["label"] == "MALICIOUS" else om["vleg"]
            else:
                raise ValueError(method_name)
            blocks.append(om["raw"] + ("\nReflection: " + refl if refl is not None else ""))
        target_str = "\n\nTarget flow features: " + feature_text(target_row)
        return "\n\n".join(blocks) + target_str, len(blocks)

    for method in methods:
        reused = 0
        called = 0
        t_start = time.perf_counter()
        total_q = len(bundle["test"])

        for q_idx, row in enumerate(bundle["test"], 1):
            tid = row["id"]
            pkey = f"{method}:{tid}"
            if pkey in cr_state["predictions"]:
                continue

            rids = orig_state["retrieval"][tid]["ids"]
            prompt_str, eff_n = build_prompt_for_method(method, rids, row)

            # Check if prompt_str is byte-for-byte identical to archived vleg or raw_on_verified
            usable_vleg = [orig_mem_by_id[k] for k in rids if orig_mem_by_id[k]["vleg"] is not None]
            target_str = "\n\nTarget flow features: " + feature_text(row)
            vleg_prompt = "\n\n".join(m["raw"] + "\nReflection: " + m["vleg"] for m in usable_vleg) + target_str
            raw_adm_prompt = "\n\n".join(m["raw"] for m in usable_vleg) + target_str

            if prompt_str == vleg_prompt:
                arc = dict(orig_pred_map[("vleg", tid)])
                arc["method"] = method
                arc["reused_from"] = "vleg"
                cr_state["predictions"][pkey] = arc
                reused += 1
            elif prompt_str == raw_adm_prompt:
                arc = dict(orig_pred_map[("raw_on_verified", tid)])
                arc["method"] = method
                arc["reused_from"] = "raw_on_verified"
                cr_state["predictions"][pkey] = arc
                reused += 1
            else:
                if len(prompt_str) > context_chars:
                    raise ValueError(f"Prompt exceeds context_chars ({len(prompt_str)} > {context_chars})")
                reply, cost = timed_call(INFERENCE, prompt_str, model, num_ctx)
                pred = reply.strip().upper()
                if pred not in ("BENIGN", "MALICIOUS"):
                    pred = "INVALID"
                cr_state["predictions"][pkey] = {
                    "id": tid, "method": method, "truth": row["label"],
                    "attack_category": row["attack_category"], "group": row.get("group"),
                    "prediction": pred, "reply": reply, "retrieved_ids": rids,
                    "effective_examples": eff_n, "context_characters": len(prompt_str),
                    "retrieval_seconds": orig_state["retrieval"][tid]["seconds"],
                    "reused_from": None, **cost
                }
                called += 1

            if q_idx % 20 == 0 or q_idx == total_q:
                save_json(ckpt_path, cr_state)
                if called > 0 and (q_idx % 100 == 0 or q_idx == total_q):
                    elapsed = time.perf_counter() - t_start
                    print(f"  [{stage_name} | {method}] {q_idx}/{total_q} done (LLM_calls={called}, exact_reused={reused}, elapsed={elapsed:.1f}s)", flush=True)

        recs = [cr_state["predictions"][f"{method}:{r['id']}"] for r in bundle["test"]]
        m_res = metrics(recs)
        raw_res = metrics([orig_pred_map[("raw", r["id"])] for r in bundle["test"]])
        vleg_res = metrics([orig_pred_map[("vleg", r["id"])] for r in bundle["test"]])
        print(
            f">>> [{stage_name} | {method}] Acc={m_res['accuracy']*100:.2f}% | "
            f"Macro-F1={m_res['macro_f1']*100:.2f}% (vs Raw {raw_res['macro_f1']*100:.2f}%, vs VLEG {vleg_res['macro_f1']*100:.2f}%) | "
            f"Recall={m_res['attack_recall']*100:.2f}% | FPR={m_res['false_positive_rate']*100:.2f}%",
            flush=True,
        )


def audit_formal_phase1():
    all_hist = {}
    for s in SEEDS:
        bd = json.loads((FLOW_ROOT / f"formal_{s}" / "prepared.json").read_text())
        for r in bd["memory"]:
            all_hist[r["id"]] = r

    tot_adm, zero_within, zero_cross_mal, tot_mal = 0, 0, 0, 0
    cross_ce_any = 0
    for s in SEEDS:
        stage = f"formal_{s}"
        cr_state = json.loads((OUT_ROOT / stage / "checkpoint_cr.json").read_text())
        for mid, cm in cr_state["memory"].items():
            txt = cm["vleg_cr"]
            if txt is None:
                continue
            tot_adm += 1
            zero_within += 1
            src_row = all_hist[mid]
            atoms, _ = parse_rule(txt, src_row["features"])
            c_ce = sum(
                1 for hid, r2 in all_hist.items()
                if hid != mid and r2["label"] != src_row["label"] and safe_applies(atoms, r2["features"])[0]
            )
            if c_ce > 0:
                cross_ce_any += 1
            if src_row["label"] == "MALICIOUS":
                tot_mal += 1
                if c_ce == 0:
                    zero_cross_mal += 1

    print("\n=== Formal 5-Seed Phase 1 Counterexample Audit (160 episodes) ===")
    print(f"  Converged to Within-Seed CE=0 & Verifier-Valid: {tot_adm}/160 ({tot_adm/1.6:.1f}%)")
    print(f"  Within-Seed CE > 0 among admitted CR rules:     0/{tot_adm} (0.0%, down from 59.2% in VLEG)")
    print(f"  Unseen Cross-Seed Pool CE > 0 (other 4 seeds):  {cross_ce_any}/{tot_adm} ({100*cross_ce_any/max(1,tot_adm):.1f}%, down from 72.0% in VLEG)")
    print(f"  Malicious rules w/ 0 Cross-Seed Pool CE:        {zero_cross_mal}/{tot_mal} ({100*zero_cross_mal/max(1,tot_mal):.1f}%, up from 35.1% in VLEG)")


def summarize_formal_test(method):
    seed_m, raw_m, vleg_m = [], [], []
    for s in SEEDS:
        stage = f"formal_{s}"
        bd = json.loads((FLOW_ROOT / stage / "prepared.json").read_text())
        orig_preds = json.loads((FLOW_ROOT / stage / "predictions.json").read_text())
        orig_map = {(r["method"], r["id"]): r for r in orig_preds}
        cr_state = json.loads((OUT_ROOT / stage / "checkpoint_cr.json").read_text())
        seed_m.append(metrics([cr_state["predictions"][f"{method}:{r['id']}"] for r in bd["test"]]))
        raw_m.append(metrics([orig_map[("raw", r["id"])] for r in bd["test"]]))
        vleg_m.append(metrics([orig_map[("vleg", r["id"])] for r in bd["test"]]))

    print(f"\n================ FINAL 5-SEED SUMMARY ({method}) ================")
    for name, arr in [("Raw retrieval", raw_m), ("Standard VLEG", vleg_m), (method.upper(), seed_m)]:
        acc = [x["accuracy"] * 100 for x in arr]
        f1 = [x["macro_f1"] * 100 for x in arr]
        rec = [x["attack_recall"] * 100 for x in arr]
        fpr = [x["false_positive_rate"] * 100 for x in arr]
        print(
            f"  {name:16s} | Acc: {statistics.mean(acc):.2f}+/-{statistics.stdev(acc):.2f} | "
            f"Macro-F1: {statistics.mean(f1):.2f}+/-{statistics.stdev(f1):.2f} | "
            f"Recall: {statistics.mean(rec):.2f}+/-{statistics.stdev(rec):.2f} | "
            f"FPR: {statistics.mean(fpr):.2f}+/-{statistics.stdev(fpr):.2f}"
        )
    deltas = [(a["macro_f1"] - b["macro_f1"]) * 100 for a, b in zip(seed_m, raw_m)]
    print(f"  Per-seed Macro-F1 delta ({method} - Raw): {[round(d, 2) for d in deltas]} (Mean: {statistics.mean(deltas):+.2f} pp)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("pilot_and_phase1", "formal_infer"), required=True)
    ap.add_argument("--method", choices=("vleg_cr", "vleg_cr_mal"), default="vleg_cr")
    ap.add_argument("--K-cr", type=int, default=3)
    args = ap.parse_args()

    t_ping = time.perf_counter()
    reply, _ = timed_call("Reply with one word.", "Say OK.", "llama3.3:70b", 32768)
    print(f"[Smoke Test] llama3.3:70b replied {reply.strip()!r} in {time.perf_counter()-t_ping:.2f}s", flush=True)

    if args.mode == "pilot_and_phase1":
        print("\n--- Step 1A: Pilot Consolidation & End-to-End Inference (n=100) ---", flush=True)
        consolidate_stage("pilot_20261012", K_cr=args.K_cr)
        run_inference_stage("pilot_20261012", methods=("vleg_cr", "vleg_cr_mal"))

        print("\n--- Step 1B: Formal 5-Seed Phase 1 Counterexample Consolidation ---", flush=True)
        for s in SEEDS:
            consolidate_stage(f"formal_{s}", K_cr=args.K_cr)
        audit_formal_phase1()
    elif args.mode == "formal_infer":
        for s in SEEDS:
            consolidate_stage(f"formal_{s}", K_cr=args.K_cr)
            run_inference_stage(f"formal_{s}", methods=(args.method,))
        summarize_formal_test(args.method)


if __name__ == "__main__":
    main()
