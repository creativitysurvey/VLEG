# VLEG: Verified Localized Episodic Grounding with Counterexample Rule Refinement for Auditable LLM-Assisted Network Traffic Analysis

Official code, deterministic predicate compiler, and benchmark audit logs for **Verified Localized Episodic Grounding (VLEG)**, **Counterexample-Refined consolidation (VLEG-CR)**, and **Neuro-Symbolic Rule Triage (VLEG-CR-NS / VLEG-CR-Bank)**.

## Key Benchmark Results (Deduplicated 19-Feature UNSW-NB15, 5 Seeds x 1,000 Test Fingerprints)

| Regime | Method | Accuracy (%) | Macro-F1 (%) | Attack Recall (%) | FPR (%) | Online LLM Calls Saved |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Few-Shot LLM (32 flows) | Raw flow retrieval | 65.08 +/- 2.93 | 63.92 +/- 2.54 | 71.39 +/- 8.44 | 38.13 +/- 6.85 | 0.0% |
| Few-Shot LLM (32 flows) | Unverified single-step reflection | 61.38 +/- 1.08 | 42.66 +/- 1.35 | 6.17 +/- 1.43 | 9.89 +/- 2.03 | 0.0% |
| Few-Shot LLM (32 flows) | VLEG (Stage-A source-verified, prompt) | 59.74 +/- 2.03 | 59.68 +/- 2.03 | 77.74 +/- 5.69 | 48.78 +/- 1.67 | 0.0% |
| Few-Shot LLM (32 flows) | VLEG-CG (within-seed CE prompt gate) | 64.66 +/- 2.83 | 63.55 +/- 2.49 | 70.27 +/- 8.14 | 37.86 +/- 7.16 | 0.0% |
| Few-Shot LLM (32 flows) | VLEG-CR (Stage-B refined, prompt-only) | 61.72 +/- 2.40 | 61.45 +/- 2.17 | 78.87 +/- 7.58 | 46.67 +/- 5.07 | 0.0% |
| **Neuro-Symbolic (Ours)** | **VLEG-CR-NS (shared top-3 + Raw LLM)** | **65.82 +/- 2.49** | **64.88 +/- 2.58** | **74.66 +/- 8.39** | **38.67 +/- 5.06** | **47.1%** |
| **Neuro-Symbolic (Ours)** | **VLEG-CR-Bank (Pilot tau_sup=3 + Raw LLM)** | **66.58 +/- 2.79** | **65.30 +/- 2.74** | **71.63 +/- 9.32** | **35.99 +/- 6.06** | **49.0%** |
| **Zero-LLM Hybrid (Ours)** | **VLEG-CR-NS + Shallow CART (depth=3)** | **68.50 +/- 2.69** | **67.57 +/- 3.08** | **79.35 +/- 13.64** | **37.01 +/- 5.72** | **100.0%** |
| **Zero-LLM Hybrid (Ours)** | **VLEG-CR-Bank (tau_sup=5) + CART (depth=3)** | **69.02 +/- 3.55** | **68.16 +/- 3.47** | **81.07 +/- 13.44** | **37.10 +/- 7.73** | **100.0%** |
| Few-Shot Tabular (32 flows) | Shallow Decision Tree (CART, depth=3) | 70.18 +/- 1.64 | 69.39 +/- 1.76 | 83.38 +/- 12.86 | 36.53 +/- 6.76 | N/A |
| Few-Shot Tabular (32 flows) | Gradient Boosting / GBDT (32-flow) | 66.60 +/- 4.46 | 65.84 +/- 4.60 | 78.04 +/- 10.27 | 39.22 +/- 5.52 | N/A |
| Full-Train Tabular (88,699) | Random Forest (Full-train upper bound) | 82.06 +/- 0.36 | 81.31 +/- 0.36 | 91.99 +/- 0.59 | 22.99 +/- 0.55 | N/A |

## Reproducing Experiments & Manuscript

```bash
python3 -u scripts/run_vleg_cr.py --mode pilot_and_phase1
python3 -u scripts/run_formal_vleg_cr_fast.py
python3 -u scripts/run_reviewer_supplements.py
python3 -u scripts/prepare_v2_data.py
python3 scripts/build_cmc_submission_v2.py
```
