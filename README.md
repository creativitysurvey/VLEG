# VLEG Experiment 3 — Iterative Refinement Ablation

Real, runnable implementation of the paper's Experiment 3 (Section IV-D):
compares **K=1** (single-step reflection, no verification — mimics the
Reflexion-style baseline) against **K=3** (full VLEG cyclic
generate-verify-refine loop) on a Level 3 Experience-Transfer task
(masked object combinations).

This is a **reduced-order** implementation targeting a local Ollama
model instead of GPT-4o, and a symbolic text task instead of rendered
multi-modal scenes. See `DEVIATIONS_LOG.md` for the full, itemized list
of every place this diverges from the paper and why.

## 1. Setup

```bash
# Check what models you already have
ollama list

# If nothing suitable is listed, pull one (recommended default):
ollama pull llama3.1:8b

# Make sure the server is running (usually starts automatically; if not):
ollama serve
```

If you use a different model name than `llama3.1:8b`, edit
`core/llm_client.py` and change `MODEL_NAME` to match exactly what
`ollama list` shows.

No extra Python packages are required — everything uses the standard
library (`urllib`, `json`). Any Python 3.10+ interpreter works.

Optional conda env if you want isolation:

```bash
conda create -n vleg python=3.11 -y
conda activate vleg
```

## 2. Get the code onto your machine

If you're copying this from the sandbox by hand, create the files with
the same relative paths shown in this project (`core/`, `experiments/`,
`DEVIATIONS_LOG.md`, `README.md`). If you'd rather I hand you a single
archive or paste-able heredoc commands for each file, ask and I'll
generate them.

## 3. Run it

```bash
cd vleg_exp3
python -m experiments.experiment3_iterative_refinement
```

This will:
1. Ping Ollama with a smoke-test message.
2. Build the Level 3 train/test split (12 training combos, 16 test
   combos including the 4 masked ones).
3. Consolidate memory under K=1, then under K=3 — printing each
   episode's verification outcome as it goes (this will take a while;
   each episode does up to K generator+verifier round trips).
4. Evaluate both memory banks on all 16 test queries via hybrid
   grounding inference.
5. Print a summary table and write full per-item results to
   `outputs/experiment3_results.json`.

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

## Environment (conda)

No third-party packages are required — everything here uses the Python
standard library only. `environment.yml` exists to pin the Python
version.

```bash
conda env create -f environment.yml
conda activate vleg-exp3
```

## Putting this in git

```bash
cd vleg_exp3
git init
git add .
git commit -m "VLEG reduced-order reimplementation: Experiments 1-5, deviations log, aligned paper"

# Create an empty repo on GitHub/GitLab/etc first, then:
git remote add origin <your-repo-url>
git branch -M main
git push -u origin main
```

`outputs/*.json` is deliberately tracked (not gitignored) — those files
are the actual measured results `DEVIATIONS_LOG.md` and
`final_aligned.tex` cite, so the repo carries its own evidence rather
than just the code that could someday reproduce it. If you'd rather keep
the repo lean, add `outputs/` to `.gitignore` yourself before the first
commit — but then the specific numbers in `DEVIATIONS_LOG.md` won't be
independently checkable from the repo alone.
