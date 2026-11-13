# VLEG: Verified Localized Episodic Grounding with Counterexample Rule Refinement for Auditable LLM-Assisted Network Traffic Analysis

Official code, deterministic predicate compiler, and benchmark audit logs for **Verified Localized Episodic Grounding (VLEG)**, **Counterexample-Refined consolidation (VLEG-CR)**, and **Neuro-Symbolic Rule Triage (VLEG-CR-NS / VLEG-CR-Bank)**.

## Overview

Memory-augmented large language models (LLMs) can assist few-shot Security Operations Center (SOC) network intrusion triage without gradient updates, yet naive natural-language reflections suffer from three failure modes:
1. **Unverified Label Inversion:** Generating reflections from label-free flow telemetry mislabels 95.0% (76/80) of malicious UNSW-NB15 support flows as benign (6.17% attack recall).
2. **Cross-Flow Over-Generalization:** Single-episode verification (VLEG, K=3) restores 100% source-label agreement (158/160 admitted), but 59.2% (93/157) of compiled rules match contrary-label counterexamples within their 32-flow seed (72.0% across pooled seeds).
3. **Prompt Anchoring on Non-Firing Rules:** Unconditionally injecting rule text into LLM prompts inflates false positives on benign flows from 32.19% to 39.45% even when zero retrieved malicious rules evaluate to true.

**VLEG-CR-NS** closes this loop in two phases:
- **Phase 1 (Verified & Counterexample-Refined Consolidation):** Couples LLM source verification with a deterministic predicate compiler that feeds within-seed contrary-label counterexamples back to the generator (K_CR=3), raising verified zero-counterexample (CE_seed = 0) rule convergence from 40.0% (64/160) to **84.4% (135/160)** and reducing unseen cross-seed counterexamples from 72.0% to **40.7%**.
- **Phase 2 (Neuro-Symbolic Rule Triage):** Evaluates compiled CE_seed = 0 Boolean predicates deterministically in <0.1 ms before falling back to top-k raw-flow LLM retrieval, eliminating non-firing prompt anchoring and saving **47.1%-49.0%** of online 70B LLM inference calls.

## Key Benchmark Results (Deduplicated UNSW-NB15, 5 Seeds x 1,000 Test Fingerprints)

| Regime | Method | Accuracy (%) | Macro-F1 (%) | Attack Recall (%) | FPR (%) | Online LLM Calls Saved |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Few-Shot LLM (32 flows) | Raw flow retrieval | 65.08 +/- 2.93 | 63.92 +/- 2.54 | 71.39 +/- 8.44 | 38.13 +/- 6.85 | 0.0% |
| Few-Shot LLM (32 flows) | Unverified single-step reflection | 61.38 +/- 1.08 | 42.66 +/- 1.35 | 6.17 +/- 1.43 | 9.89 +/- 2.03 | 0.0% |
| Few-Shot LLM (32 flows) | VLEG (Stage-A source-verified, prompt) | 59.74 +/- 2.03 | 59.68 +/- 2.03 | 77.74 +/- 5.69 | 48.78 +/- 1.67 | 0.0% |
| Few-Shot LLM (32 flows) | VLEG-CG (within-seed CE prompt gate) | 64.66 +/- 2.83 | 63.55 +/- 2.49 | 70.27 +/- 8.14 | 37.86 +/- 7.16 | 0.0% |
| Few-Shot LLM (32 flows) | VLEG-CR (Stage-B refined, prompt-only) | 61.60 +/- 2.36 | 61.33 +/- 2.11 | 78.81 +/- 7.62 | 46.82 +/- 5.18 | 0.0% |
| **Neuro-Symbolic (Ours)** | **VLEG-CR-NS (shared top-3, tau_sup=0)** | **65.78 +/- 2.53** | **64.85 +/- 2.58** | **74.72 +/- 8.26** | **38.76 +/- 5.17** | **47.1%** |
| **Neuro-Symbolic (Ours)** | **VLEG-CR-Bank (Pilot tau_sup=3)** | **66.58 +/- 2.79** | **65.30 +/- 2.74** | **71.63 +/- 9.32** | **35.99 +/- 6.06** | **49.0%** |
| Few-Shot Tabular (32 flows) | Shallow Decision Tree (CART, depth=3) | 70.18 +/- 1.64 | 69.39 +/- 1.76 | 83.38 +/- 12.86 | 36.53 +/- 6.76 | N/A |
| Few-Shot Tabular (32 flows) | Random Forest (200 trees, 32-flow) | 68.86 +/- 1.25 | 68.42 +/- 1.23 | 87.60 +/- 10.86 | 40.66 +/- 5.66 | N/A |
| Full-Train Tabular (88,699) | Random Forest (Full-train upper bound) | 82.06 +/- 0.36 | 81.31 +/- 0.36 | 91.99 +/- 0.59 | 22.99 +/- 0.55 | N/A |

- Paired cluster bootstrap 95% CI vs. Raw flow retrieval: **[+0.13, +1.71] pp** for VLEG-CR-NS and **[+0.28, +2.60] pp** for VLEG-CR-Bank.

## Repository Structure

- `core/` - Ollama LLM/embedding client (`llama3.3:70b`, `qwen2.5:7b-instruct`, `nomic-embed-text`) and strict native-Boolean JSON verifier parser.
- `symbolic/` - 16-scene controlled symbolic sanity check (Levels 1-3, grounding modality ablations, and K-budget sweeps).
- `traffic/` - Deduplicated 19-feature UNSW-NB15 benchmark preparation and Phase 1 / Phase 2 evaluation pipeline.
- `scripts/audit_explanations_and_provenance.py` - Deterministic rule compiler and counterexample auditor.
- `scripts/run_vleg_cr.py` - Counterexample-guided inductive rule refinement (VLEG-CR) on pilot and formal 5-seed support banks.
- `scripts/run_formal_vleg_cr_fast.py` - KV-cache prefix-sorted formal 5-seed inference and paired bootstrap evaluation.
- `results/vleg_cr_20261112/` - Saved VLEG-CR consolidation checkpoints, predictions, and neuro-symbolic ablation summaries.
- `CMC_submission/` - Manuscript LaTeX source (`main.tex`), bibliography (`references.bib`), and compiled PDF (`main.pdf`).

## Reproducing VLEG-CR and Neuro-Symbolic Triage

```bash
# 1. Pilot screening + 5-seed counterexample-guided rule refinement (Phase 1 Stage B)
python3 -u scripts/run_vleg_cr.py --mode pilot_and_phase1

# 2. Formal 5-seed LLM evaluation & paired bootstrap summary
python3 -u scripts/run_formal_vleg_cr_fast.py

# 3. Build manuscript PDF
python3 scripts/build_cmc_submission.py
cd CMC_submission && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
```
