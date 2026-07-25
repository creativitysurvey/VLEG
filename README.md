# VLEG — Reduced-Order Reimplementation and Paper Validation

This repo is a real, runnable reimplementation of the paper *"Verified
Localized Episodic Grounding: Robust Memory Consolidation for Large
Language Models in Heterogeneous Environments"* (VLEG), built to check
whether the paper's five experiments (which originally reported
fabricated/illustrative numbers, not measured ones) actually hold up
when run for real.

**Short version of what we found:** some of VLEG's claims replicate
(iterative verification does improve memory coverage and accuracy over
an unverified baseline), and several do not (Hybrid Grounding
underperforms simpler alternatives specifically on the heterogeneous
condition it was designed for; the simplest baseline, ICL, beats VLEG
at every level tested; "K=3" is not a real plateau). The full,
itemized account of every deviation, bug, and finding — in the order
it was discovered — is in [`DEVIATIONS_LOG.md`](DEVIATIONS_LOG.md).
The paper itself, rewritten to match what was actually measured, is
[`final_aligned.tex`](final_aligned.tex) / `final_aligned.pdf`.

This is a **reduced-order** implementation: a local 7B-parameter model
(Qwen2.5-7B-Instruct via Ollama) instead of GPT-4o, symbolic attribute
tuples instead of rendered multi-modal scenes, and small exhaustively-
enumerated task populations (8-16 items) instead of 500-query samples.
See `DEVIATIONS_LOG.md` entries #1-#6 and #14-#16 for exactly what was
simplified and why.

## Repo layout

```
core/                      Shared library code
  task_env.py                 The symbolic Experience-Transfer environment
                               (Levels 1-3, oracle rule, train/test splits)
  llm_client.py                Ollama chat + embeddings wrapper, token tracking
  generator.py / verifier.py   VLEG's Reflection Generator (G_theta) and
                               Strict Verifier (V_phi)
  refinement.py                The K-bounded generate-verify-refine loop
  memory.py / retrieval.py     Memory bank + embedding-based top-k retrieval
  inference.py                 Hybrid / Abstract-Only / Raw-Only grounding
  single_step.py               Unverified single-step reflection (Reflexion-style)
  baselines.py                 ICL, Reflexion, and HDR baseline methods (Exp 1)
  oracle_verifier.py           Ground-truth-aware auditors used in Exp 5

experiments/                One script per experiment, each with a main()
  experiment1_sota_comparison.py     Exp 1 — VLEG vs. HDR vs. Reflexion vs. ICL
  experiment2_grounding_ablation.py  Exp 2 — Hybrid vs. Abstract-Only vs. Raw-Only
  experiment3_iterative_refinement.py Exp 3 — K=1 vs. K=3 ablation
  experiment4_parameter_sensitivity.py Exp 4 — K in {1,2,3,5}, token overhead
  experiment5_memory_quality.py      Exp 5 — label-correctness vs. spurious attribution
  experiment{1,2,3,4}_multi_seed.py  Robustness checks across masking seeds
  reaudit_experiment5_*.py           Follow-up re-audits of Exp 5's reflections
  diagnose_masked_retrieval.py       Retrieval diagnostic used during Exp 3
  sanity_check_verifier.py           Standalone sanity check of the Strict Verifier

run_all.py                  Runs Experiments 1-5 (single-seed versions) end to end
outputs/                     JSON results from every run — tracked in git, see below
DEVIATIONS_LOG.md            The full log: every deviation, bug, and finding
final_aligned.tex / .pdf     The paper, rewritten to match measured results
references.bib
environment.yml / .gitignore
```

## 1. Setup

```bash
# Check what models you already have
ollama list

# You need a chat model and an embedding model, e.g.:
ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text

# Make sure the server is running (usually starts automatically; if not):
ollama serve
```

If your model names differ from `qwen2.5:7b-instruct` / `nomic-embed-text`,
edit `core/llm_client.py` and change `MODEL_NAME` / `EMBED_MODEL_NAME` to
match exactly what `ollama list` shows.

No third-party Python packages are required — everything uses the
standard library only (`urllib`, `json`, etc.). Any Python 3.10+
interpreter works.

```bash
conda env create -f environment.yml
conda activate vleg-exp3
```

## 2. Run it

**Everything, in order** (~20-30 min on a local 7B model):
```bash
python run_all.py
```

**One experiment at a time** (e.g. Experiment 3):
```bash
python -m experiments.experiment3_iterative_refinement
```

Each script prints its progress live (episode-by-episode
verification, per-condition accuracy) and writes full results to
`outputs/<script_name>_results.json`.

**Multi-seed robustness checks** (more expensive — 15-20+ min each;
these are what turned several single-seed results into trustworthy
ones, see `DEVIATIONS_LOG.md` #9-#13):
```bash
python -m experiments.experiment1_multi_seed
python -m experiments.experiment2_multi_seed
python -m experiments.experiment3_multi_seed
python -m experiments.experiment4_multi_seed
```

**Diagnostics** (cheap, standalone — useful if a result looks
degenerate and you want to isolate why):
```bash
python -m experiments.sanity_check_verifier          # does V_phi actually reject bad reflections?
python -m experiments.diagnose_masked_retrieval       # what gets retrieved for masked queries?
python -m experiments.reaudit_experiment5_ground_truth # label-correct vs. spurious attribution, decomposed
```

## 3. What each experiment tests

| # | Question | Where the real answer diverges from the paper |
|---|----------|------------------------------------------------|
| 1 | Does VLEG beat HDR / Reflexion / ICL? | ICL beat VLEG at every level tested; HDR is bimodal (perfect when the rule is globally learnable, weak when it's conditional) rather than uniformly weak |
| 2 | Is Hybrid Grounding better than Abstract-Only / Raw-Only? | Hybrid wins on Level 2 but is the *worst* of the three on Level 3 / masked combinations — the condition it was designed for |
| 3 | Does the K=1→K=3 verification loop help? | Yes, modestly (+11.25pp overall, +10pp on masked combos, pooled over 5 seeds) — but via memory coverage recovery, not reflection quality |
| 4 | Does accuracy plateau at K=3, and does token cost scale linearly? | Neither: Level 2 keeps improving through K=5; Level 3 plateaus in only 1 of 3 seeds; token cost is concave, not linear |
| 5 | Does verification reduce hallucination? | It drives label-correctness to 100%, but has no measurable effect on spurious attribute citation (~80-87% either way) |

See `DEVIATIONS_LOG.md` for the full numbers, methodology notes, and
the reasoning behind every reduced-order substitution.

## 4. If something breaks

Paste me:
- The full terminal output (especially any `OllamaError` message).
- The output of `ollama list`.
- If predictions look degenerate (e.g. always ANOMALY or always
  NORMAL), paste a few lines of the verified reflections printed
  during consolidation — that usually tells us whether the Generator
  or Verifier prompt needs tuning for your particular model.

Expect several round trips here — this is meant to be debugged
interactively against your real output, not run once and trusted
blindly.

## Putting this in git

```bash
cd vleg_exp3
git init
git add .
git commit -m "VLEG reduced-order reimplementation: Experiments 1-5, deviations log, aligned paper"
git remote add origin <your-repo-url>
git branch -M main
git push -u origin main
```

`outputs/*.json` is deliberately tracked (not gitignored) — those files
are the actual measured results `DEVIATIONS_LOG.md` and
`final_aligned.tex` cite, so the repo carries its own evidence rather
than just the code that could someday reproduce it.
