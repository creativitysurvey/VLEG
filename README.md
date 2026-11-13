# VLEG: Verified Localized Episodic Grounding with Counterexample Rule Refinement for Auditable LLM-Assisted Network Traffic Analysis

Official code, deterministic predicate compiler, and benchmark audit logs for **Verified Localized Episodic Grounding (VLEG)**, **Counterexample-Refined consolidation (VLEG-CR)**, and **Neuro-Symbolic Rule Triage (VLEG-CR-NS / VLEG-CR-Bank)**.

- **Phase 1 Consolidation (5 formal seeds, 160 support flows):** Stage-A VLEG admits `158/160` source-verified episodes; Stage-B VLEG-CR raises verified zero-counterexample (`CE_seed = 0`) rule coverage from `40.0%` (`64/160`) to **`83.1%` (`133/160`, or `84.2%` of the `158` Stage-A admitted episodes)** and cuts unseen cross-seed counterexamples from `72.0%` to **`40.6%` (`54/133`)**.
- **Phase 2 Online Triage (5 x 1,000 deduplicated UNSW-NB15 test fingerprints, strict leak-free 32-flow support encoding):**

| Regime | Method | Accuracy (%) | Macro-F1 (%) | Attack Recall (%) | FPR (%) | Online LLM Calls Saved |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Few-Shot LLM (32 flows) | Raw flow retrieval | 65.08 +/- 2.93 | 63.92 +/- 2.54 | 71.39 +/- 8.44 | 38.13 +/- 6.85 | 0.0% |
| Few-Shot LLM (32 flows) | Unverified single-step reflection | 61.38 +/- 1.08 | 42.66 +/- 1.35 | 6.17 +/- 1.43 | 9.89 +/- 2.03 | 0.0% |
| Few-Shot LLM (32 flows) | VLEG (Stage-A source-verified, prompt) | 59.74 +/- 2.03 | 59.68 +/- 2.03 | 77.74 +/- 5.69 | 48.78 +/- 1.67 | 0.0% |
| Few-Shot LLM (32 flows) | VLEG-CG (within-seed CE prompt gate) | 64.66 +/- 2.83 | 63.55 +/- 2.49 | 70.27 +/- 8.14 | 37.86 +/- 7.16 | 0.0% |
| Few-Shot LLM (32 flows) | VLEG-CR (Stage-B refined, prompt-only) | 61.72 +/- 2.40 | 61.45 +/- 2.17 | 78.87 +/- 7.58 | 46.67 +/- 5.07 | 0.0% |
| **Neuro-Symbolic (Ours)** | **VLEG-CR-NS (top-3 + Raw LLM, tau_sup=0)** | **65.82 +/- 2.49** | **64.88 +/- 2.58** | **74.66 +/- 8.39** | **38.67 +/- 5.06** | **47.1%** |
| **Neuro-Symbolic (Ours)** | **VLEG-CR-Bank (Pilot tau_sup=3 + Raw LLM)** | **66.58 +/- 2.79** | **65.30 +/- 2.74** | **71.63 +/- 9.32** | **35.99 +/- 6.06** | **49.0%** |
| **Zero-LLM Hybrid (Ours)** | **VLEG-CR-Bank (fixed tau_sup=3) + CART (d=3)** | **67.92 +/- 2.99** | **67.05 +/- 2.95** | **79.17 +/- 12.32** | **37.80 +/- 6.70** | **100.0% (0 embed)** |
| **Zero-LLM Hybrid (Ours)** | **VLEG-CR-NS (tau_sup=0) + CART (d=3)** | **68.50 +/- 2.69** | **67.57 +/- 3.08** | **79.35 +/- 13.64** | **37.01 +/- 5.72** | **100.0%** |
| Few-Shot Tabular (32 flows) | Shallow Decision Tree (CART, depth=3) | 69.88 +/- 1.91 | 68.71 +/- 2.02 | 79.41 +/- 16.27 | 34.96 +/- 8.78 | N/A |
| Few-Shot Tabular (32 flows) | Gradient Boosting / GBDT (32-flow) | 67.90 +/- 4.15 | 67.12 +/- 4.32 | 79.76 +/- 11.62 | 38.13 +/- 5.86 | N/A |
| Full-Train Tabular (88,699) | Random Forest (Full-train upper bound) | 82.06 +/- 0.36 | 81.31 +/- 0.36 | 91.99 +/- 0.59 | 22.99 +/- 0.55 | N/A |

## Reproducing Experiments & Final Manuscript

```bash
python3 -u scripts/run_vleg_cr.py --mode pilot_and_phase1
python3 -u scripts/run_formal_vleg_cr_fast.py
python3 -u scripts/run_reviewer_supplements.py
python3 -u scripts/prepare_v2_data.py
python3 scripts/build_cmc_submission_v4.py
```
