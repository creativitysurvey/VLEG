# VLEG — Reduced-Order Reimplementation and Empirical Validation

VLEG ("Verified Localized Episodic Grounding: Robust Memory
Consolidation for Large Language Models in Heterogeneous Environments")
is a memory-consolidation framework: episodes are turned into
verified, distilled "reflections" by a K-bounded generate-verify-refine
loop, then retrieved and grounded at inference time. This repo is a
real, runnable implementation built to test VLEG's own claims —
originally motivated with illustrative numbers, not measured ones —
against actual experiments, at increasing scale, model capability, and
(eventually) data realism, and to report what holds up and what
doesn't.

**This repo now contains TWO parallel implementations:**
- **`core/` + `experiments/`** — the original, symbolic-attribute
  version (episodes are text tuples like "white background, red
  circle"). Runs on 7B and 70B local models.
- **`core_visual/` + `experiments_visual/`** — a real-image version
  built on CLEVR (Johnson et al. 2017), added after confirming
  real-scale multi-modal data and a local vision-capable model were
  both feasible. Runs on a 70B-class vision model (Qwen2.5-VL-72B).

**Short version of what we found**, across ~40 rounds of experiments,
bugs, and re-runs (full blow-by-blow, in discovery order, in
[`DEVIATIONS_LOG.md`](DEVIATIONS_LOG.md)):
- Several 7B-scale findings that looked like real architectural
  problems (ICL beating VLEG, Hybrid Grounding underperforming, a
  fragile K=3 "plateau") **reversed at 70B scale** — small-model
  artifacts, not properties of VLEG.
- One finding did **not** reverse at any scale tested: a per-episode
  verifier reliably fixes label-correctness but is structurally blind
  to over-specification (citing irrelevant attributes as necessary).
- Moving to real images (CLEVR) **replicated** VLEG's core memory-
  coverage mechanism but **did not replicate** the accuracy benefit —
  and led to discovering a new failure mode, **spurious cross-
  reflection correlation**: every consolidated example of one label
  happening to share an incidental attribute (e.g. color), teaching the
  model an incorrect dependency no single verified reflection states
  and no per-reflection audit can catch. This is causally confirmed via
  ablation, reproduced by deliberate (bidirectional) induction, and
  partially mitigated (a structural fix works; a prompt-level warning
  does not and can backfire).
- A real-image re-run of the SOTA comparison shows a concrete case
  where VLEG's stricter verifier (which checks visual accuracy, not
  just label correctness) prevents a failure mode that sinks all three
  baselines (ICL, Reflexion, HDR) to a collapsed constant prediction.

The paper itself, kept in sync with what was actually measured at every
step, is [`final_aligned.tex`](final_aligned.tex) / `final_aligned.pdf`.

## Repo layout

```
core/                        Symbolic-pipeline library code
  task_env.py                   The symbolic Experience-Transfer environment
                                 (Levels 1-3, oracle rule, train/test splits)
  llm_client.py                 Ollama chat + embeddings wrapper, token tracking
                                 MODEL_NAME here controls which model Experiments
                                 1-5 in experiments/ use (7B or 70B)
  generator.py / verifier.py    VLEG's Reflection Generator (G_theta) and
                                 Strict Verifier (V_phi)
  refinement.py                 The K-bounded generate-verify-refine loop
  memory.py / retrieval.py      Memory bank + embedding-based top-k retrieval
  inference.py                  Hybrid / Abstract-Only / Raw-Only grounding
  single_step.py                Unverified single-step reflection
  baselines.py                  ICL, Reflexion (K-bounded, fairness-fixed —
                                 see DEVIATIONS_LOG.md #32), and HDR
  oracle_verifier.py             Early LLM-based auditor for Exp 5, later found
                                 unreliable (see #12) -- kept for reference,
                                 superseded by the deterministic ground-truth +
                                 keyword checks used everywhere else

core_visual/                 Real-image (CLEVR) pipeline library code
  task_env.py                   Loads clevr_crops[_large]/index.json; oracle
                                 rule maps shape+color (causal), material
                                 (background-analog, flips the rule), size
                                 (distractor) onto real CLEVR attributes
  llm_client.py                 Separate Ollama client with image support
                                 (base64 encoding, images= kwarg). MODEL_NAME
                                 here controls experiments_visual/'s model
                                 (currently qwen2.5vl:72b)
  generator.py / verifier.py    Image-aware Generator/Verifier -- Verifier
                                 also checks whether a reflection's claimed
                                 visual attributes match the actual image
  refinement.py / memory.py / retrieval.py   Same roles as core/'s, adapted
                                 for images (retrieval is over reflection TEXT
                                 embeddings only, images aren't embedded)
  inference.py                  predict_hybrid / predict_abstract_only /
                                 predict_raw_only -- see the module docstring
                                 for why past-example IMAGES are deliberately
                                 NOT shown in Hybrid mode (multi-image
                                 attention collapse, #25)
  single_step.py                Unverified single-step commit
  baselines.py                  ICL, Reflexion, HDR adapted to this pipeline's
                                 image constraints -- see the module docstring,
                                 all three baselines are known to collapse to a
                                 constant prediction for three DIFFERENT
                                 diagnosed reasons (#41)
  debias.py                     Cross-episode attribute-skew detector +
                                 structural mitigation (#27-#31) for the
                                 spurious-correlation failure mode

experiments/                 Symbolic-pipeline experiment scripts
  experiment{1..5}_*.py           One script per experiment, run with
                                   `python -m experiments.<name>`
  experiment{1,2,3,4}_multi_seed.py   Multi-seed robustness checks
  reaudit_experiment5_*.py        Early re-audit attempts (superseded, kept
                                   for the historical record in the log)
  diagnose_masked_retrieval.py, sanity_check_verifier.py   Standalone checks

experiments_visual/          Real-image (CLEVR) pipeline experiment scripts
  experiment1_sota_comparison.py      VLEG vs ICL/Reflexion/HDR (pilot, n=32)
  experiment2_grounding_ablation.py   Hybrid vs Abstract-Only vs Raw-Only (n=48)
  experiment3_iterative_refinement.py K=1 vs K=3, LARGE SCALE (n=160)
  experiment3_multi_seed.py           K=1 vs K=3 across 5 seeds (n=16 each,
                                       the ORIGINAL pilot scale, not the n=160
                                       scale -- see DEVIATIONS_LOG.md #26 vs #27)
  experiment4_parameter_sensitivity.py  K in {1,2,3,5} sweep (pilot, n=48)
  experiment5_memory_quality.py       label-correctness + visual-attribute
                                       hallucination audit (single seed)
  experiment5_multi_seed.py           same, pooled across 3 seeds
  diagnose_*.py                       ~12 standalone diagnostics -- each one
                                       was written to answer a SPECIFIC
                                       question that came up mid-investigation;
                                       see DEVIATIONS_LOG.md for which entry
                                       each corresponds to (grep the filename)
  fill_sphere_blue_metal_gap.py       one-off retry script for a specific
                                       combo that repeatedly failed to
                                       consolidate at K=3

prepare_clevr.py             Builds clevr_crops[_large]/index.json from a
                              local CLEVR_v1.0 install -- see Section 2 below
run_all.py                   Runs symbolic Experiments 1-5 (single-seed) end
                              to end -- does NOT touch experiments_visual/
outputs/ , outputs_visual/   JSON results from every run -- tracked in git
DEVIATIONS_LOG.md            The full log: every deviation, bug, and finding,
                              in the order it was discovered (~40 entries)
final_aligned.tex / .pdf     The paper, kept in sync with measured results
references.bib
environment.yml / .gitignore
```

## 1. Setup

```bash
conda env create -f environment.yml
conda activate vleg-exp3
```

No third-party Python packages beyond what's in `environment.yml` are
required for the LLM-calling code itself (`urllib`, `json`, standard
library) — `Pillow` is needed only for `prepare_clevr.py`'s image
cropping.

### Models

This repo now uses **four** different Ollama models across the two
pipelines. Check what you have:

```bash
ollama list
```

Pull whatever's missing:

```bash
ollama pull qwen2.5:7b-instruct      # symbolic pipeline, small-model results
ollama pull llama3.3:70b             # symbolic pipeline, 70B cross-scale results (~42GB)
ollama pull qwen2.5vl:72b            # real-image pipeline (~48GB) -- REQUIRED for
                                      # anything in experiments_visual/, this is a
                                      # vision-capable model, the other three aren't
ollama pull nomic-embed-text         # embeddings, both pipelines
```

`qwen2.5:7b-instruct` and `llama3.3:70b` are both usable by
`core/llm_client.py` — which one actually runs depends on
`MODEL_NAME` at the top of that file; change it and re-run to switch
scale. `core_visual/llm_client.py` is a **separate file** with its own
`MODEL_NAME`, currently hardcoded to `qwen2.5vl:72b` — the symbolic and
real-image pipelines are not meant to share a model config, since the
real-image one specifically needs vision support.

Make sure the server is running:

```bash
ollama serve
# or, to keep it alive independent of your terminal session:
nohup ollama serve > ollama_serve.log 2>&1 &
disown
```

**Hardware**: `llama3.3:70b` and `qwen2.5vl:72b` both need a GPU with
roughly 45-55GB+ of free VRAM at their default Ollama quantization
(Q4_K_M). If the GPU is shared with other workloads, expect possible
instability — see the reproducibility caveats in Section 6.

## 2. Preparing the real-image (CLEVR) data

**Only needed for `experiments_visual/`** — skip this section if you
only want to run the symbolic pipeline.

```bash
# Download CLEVR (this is large, ~18GB)
wget https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip
unzip CLEVR_v1.0.zip
```

`prepare_clevr.py` crops single objects out of CLEVR's multi-object
scenes (CLEVR's metadata gives each object's pixel-space center, not a
bounding box, so candidates are ranked by distance to the nearest other
object in the same scene and the most isolated instances are kept —
see the script's docstring and `DEVIATIONS_LOG.md` #22-#23 for the full
methodology and its known limitations). It needs to be run **twice**,
producing two separate crop sets used by different experiments:

```bash
# Small set: 1 image per combo (16 total) -- used by early pilot
# diagnostics and experiment3_multi_seed.py's original-scale check
python prepare_clevr.py --clevr-root ~/path/to/CLEVR_v1.0 --split train \
    --out-dir clevr_crops --n-per-combo 1

# Large set: 32 images per combo (512 total) -- used by everything else,
# including all "large scale" and pilot-scale experiments_visual/ scripts
python prepare_clevr.py --clevr-root ~/path/to/CLEVR_v1.0 --split train \
    --out-dir clevr_crops_large --n-per-combo 32
```

`core_visual/task_env.py` reads from `clevr_crops_large/index.json` by
default for anything using `make_level3_split_large` (i.e. almost every
current script in `experiments_visual/`) — `clevr_crops/` (the n=1 set)
is only still relevant if you're specifically re-running
`experiments_visual/experiment3_multi_seed.py` in its original form, or
want to reproduce the very earliest pilot diagnostics.

**Sanity-check the crops before trusting them**: open a handful of
files in `clevr_crops_large/`, especially high-index ones like
`*_28.png` through `*_32.png` (later in the isolation-ranking, so more
likely to have a quality issue) — confirm each shows exactly one clean,
unobstructed object matching its filename. This isn't optional; the
first version of this pipeline's crop selection method produced
contaminated crops (a second object partially or fully in frame) about
25% of the time before the ranking method was fixed — see #22.

## 3. Running the symbolic-pipeline experiments

**Everything, in order** (~20-30 min at 7B, much longer at 70B):
```bash
python run_all.py
```

**One experiment at a time**:
```bash
python -m experiments.experiment3_iterative_refinement
```

**Multi-seed robustness checks** (each is substantially more expensive
than the single-seed version — 15 min to several hours depending on
model size, see `DEVIATIONS_LOG.md` for measured per-experiment
timings):
```bash
python -m experiments.experiment1_multi_seed
python -m experiments.experiment2_multi_seed
python -m experiments.experiment3_multi_seed
python -m experiments.experiment4_multi_seed
```

Before running any multi-seed script after changing `MODEL_NAME`,
check whether the script has an `ORIGINAL` dict used for a
repeatability sanity-check (e.g. `experiment1_multi_seed.py`) — it
needs to be updated to the CURRENT model's baseline numbers, or the
built-in "did this repeat exactly?" check will misreport a real model
change as noise. See the comment above that dict in the script.

**Diagnostics**:
```bash
python -m experiments.sanity_check_verifier
python -m experiments.diagnose_masked_retrieval
```

## 4. Running the real-image-pipeline experiments

Requires Section 2's CLEVR setup to be done first.

**Recommended order** (roughly cheapest/most-load-bearing first; each
of the 5 main experiments has at least a pilot-scale result, Experiment
3 has the deepest coverage):

```bash
# Sanity check first -- confirms the vision model can actually see
python -m experiments_visual.diagnose_multi_image

# Experiment 3 -- has single-seed, 5-seed (n=16, "original" scale), and
# a separate large-scale (n=160) version; run whichever you need
python -m experiments_visual.experiment3_iterative_refinement   # large scale, n=160
python -m experiments_visual.experiment3_multi_seed              # 5 seeds, n=16 each

python -m experiments_visual.experiment5_memory_quality          # single seed
python -m experiments_visual.experiment5_multi_seed              # 3 seeds, pooled

python -m experiments_visual.experiment2_grounding_ablation      # pilot, n=48
python -m experiments_visual.experiment4_parameter_sensitivity   # pilot, n=48

python -m experiments_visual.experiment1_sota_comparison         # pilot, n=32,
                                                                    # SLOWEST -- HDR
                                                                    # attaches up to 12
                                                                    # images per call
```

**The color-correlation diagnostic chain** (this is what found, isolated,
and fixed the spurious cross-reflection correlation finding — see
`DEVIATIONS_LOG.md` #27-#31 for the full narrative these correspond to):
```bash
# 1. Builds a complete 16-combo reflection library, cached to
#    outputs_visual/reflection_library_all16.json for reuse by the rest
python -m experiments_visual.diagnose_controlled_bias_induction

# 2. Confirms the effect causally via bank-composition ablation
python -m experiments_visual.diagnose_bank_ablation

# 3. A prompt-level mitigation attempt (spoiler: it fails, see #30)
python -m experiments_visual.diagnose_debias_validation

# 4. A structural mitigation attempt (spoiler: it works, see #31)
python -m experiments_visual.diagnose_structural_debias

# 5. Robustness of the structural mitigation under data scarcity
python -m experiments_visual.diagnose_debias_scarcity

# 6. Does the effect generalize to shape/size, not just color?
python -m experiments_visual.diagnose_multidim_bias
```

Steps 2-6 all **reuse** the cached `reflection_library_all16.json` from
step 1 — they do not re-run consolidation, so they're cheap once step 1
is done. If you re-run step 1 after any change to
`core_visual/generator.py` or `core_visual/verifier.py`, delete the
cached file first so it rebuilds rather than silently reusing stale
reflections.

**Other standalone diagnostics** (each was written to answer one
specific question mid-investigation — check `DEVIATIONS_LOG.md` for
which numbered entry a given script's findings correspond to, the
filenames are meant to be self-explanatory):
```bash
python -m experiments_visual.diagnose_hybrid_collapse
python -m experiments_visual.diagnose_text_examples_variant
python -m experiments_visual.diagnose_masked_errors
python -m experiments_visual.diagnose_exp1_baseline_collapse   # reads existing
                                                                  # results, no new
                                                                  # LLM calls
python -m experiments_visual.diagnose_exp1_raw_replies
python -m experiments_visual.fill_sphere_blue_metal_gap        # one-off retry for
                                                                  # a specific combo
```

## 5. What each experiment tests

| # | Question | Symbolic-pipeline answer (7B → 70B) | Real-image-pipeline answer |
|---|----------|--------------------------------------|------------------------------|
| 1 | Does VLEG beat HDR / Reflexion / ICL? | 7B: ICL beats VLEG everywhere. 70B: reverses — VLEG matches/beats ICL at every level (confirmed multi-seed) | VLEG 87.5%; all 3 baselines collapse to a constant prediction, via 3 different mechanisms (#41) |
| 2 | Is Hybrid Grounding better than Abstract-Only / Raw-Only? | 7B: Hybrid worst on Level 3/masked. 70B: mixed — ties for best overall, Abstract-Only keeps a narrower masked-combo edge | Abstract-Only is degenerate by construction here (uninformative query without an image); Raw-Only reproduces the multi-image collapse |
| 3 | Does the K=1→K=3 loop help? | 7B: modest gain via coverage recovery. 70B: gain roughly doubles | Coverage-recovery mechanism replicates; downstream accuracy does NOT track it — net negative at both pilot and large (n=160) scale |
| 4 | Does accuracy plateau at K=3? | 7B: inconsistent across seeds. 70B: clean plateau, but at K=2, one iteration earlier than hypothesized | K=1 alone scored 100% in one pilot run; K=2/K=3 measurably worse — independently confirms Exp3's coverage-vs-accuracy tension |
| 5 | Does verification reduce hallucination? | Drives label-correctness to 100%, no effect on over-specification (~80-87% either way, both scales) | Same label-correctness pattern; NEW finding: visual-attribute hallucination is ~0% regardless of verification — errors are in label reasoning, not perception |

See `DEVIATIONS_LOG.md` for exact numbers, every methodology decision,
and the full causal-ablation chain behind the spurious cross-reflection
correlation finding (search for entries #27 through #31, #34-#36 for
the multi-dimensional and scarcity follow-ups).

## 6. Reproducibility caveats

- **Temperature-0 is usually, but not always, exactly deterministic on
  this stack.** Two documented instances broke it: one under confirmed
  heavy GPU contention from unrelated processes, one with no confirmed
  contention (see `DEVIATIONS_LOG.md`, search "determinism"). Neither
  changed any directional conclusion in this project, but exact-match
  reruns should be treated as a strong signal, not an absolute
  guarantee, especially on a shared GPU.
- **GPU contention can also cause outright failures** (HTTP 500s,
  connection refused). `core/llm_client.py` and `core_visual/llm_client.py`
  both retry transient failures automatically (3 attempts, linear
  backoff) and every experiment script checkpoints incrementally to
  `outputs/` / `outputs_visual/` — a crash mid-run should only cost the
  in-progress item, not the whole run. Check `nvidia-smi` if things seem
  unusually slow or fail repeatedly.
- **Pilot scale vs. large scale**: only Experiment 3 in the real-image
  pipeline has been run at large scale (n=160) and multi-seed.
  Experiments 1, 2, 4, and 5's real-image results are pilot-scale
  (n=32-48), single-seed. Treat these as directionally informative, not
  as precise as the symbolic pipeline's multi-seed numbers.
- Always run `ollama list` and check the relevant `llm_client.py`'s
  `MODEL_NAME` before trusting which model a given result came from —
  it's easy to run a script against the wrong model if you switched
  scales earlier in a session and forgot to switch back.

## 7. If something breaks

**If a run "works" (no crash, exit code 0) but takes dramatically longer than the timings in this README** (e.g. a 7B `run_all.py` taking many hours instead of ~46 minutes, with the slowdown concentrated in the longest-prompt conditions like HDR or multi-attempt Reflexion): this is a strong sign Ollama fell back to CPU instead of using the GPU. This can happen silently if Ollama doesn'''t detect a compatible GPU or judges the available VRAM insufficient. Check:
```bash
nvidia-smi          # confirm GPU utilization actually rises during a run
```
and look in Ollama'''s own server output for lines mentioning how many layers were offloaded to GPU (a low or zero offload count means it'''s running mostly or entirely on CPU). A 7B model needs meaningfully less VRAM than the 70B-class models this repo also uses, but still needs a real GPU to run at the speeds this README assumes — pure CPU inference for a 7B model is commonly 10-20x+ slower, and disproportionately worse for longer prompts (HDR'''s whole-dataset context, Reflexion'''s multi-attempt retries) than for short ones, matching exactly the lopsided slowdown pattern one reproduction of this repo showed (see DEVIATIONS_LOG.md #47).

Paste:
- The full terminal output, especially any `OllamaError` or HTTP
  error message.
- The output of `ollama list` and (if it looks slow/unstable)
  `nvidia-smi`.
- If predictions look degenerate (e.g. always ANOMALY, always NORMAL),
  paste a few of the actual reflections printed during consolidation,
  or re-run the relevant `diagnose_*.py` script if one already exists
  for that symptom — several do (see Section 4).

Expect several round trips — this codebase was built by debugging
against real, sometimes-surprising model output at every step, not by
writing code once and trusting it blindly. `DEVIATIONS_LOG.md` is the
record of every time that approach caught something worth catching.

## Putting this in git

```bash
git add .
git commit -m "..."
git push
```

`outputs/*.json` and `outputs_visual/*.json` are deliberately tracked
(not gitignored) — those files are the actual measured results
`DEVIATIONS_LOG.md` and `final_aligned.tex` cite, so the repo carries
its own evidence rather than just the code that could someday reproduce
it. `clevr_crops/` and `clevr_crops_large/` (the actual cropped images)
are large and derived from a public dataset you can regenerate with
`prepare_clevr.py` — check your `.gitignore` before pushing if you'd
rather not commit ~500 images; `index.json` in each is small and worth
keeping either way, since it records exactly which CLEVR source images
and isolation scores were used.

## Public network-flow extension (Experiment 6)

`traffic/` adds an independent public-flow benchmark pipeline for UNSW-NB15
and CICIDS2017 CSVs. It has **not yet produced real-model/public-flow results**.
See [the protocol and commands](docs/TRAFFIC_BENCHMARK.md) for preparation,
leakage checks, controlled methods, checkpoint/resume, scoring and human audit.

The original `core/task_env.py` uses synthetic **scene attributes**, not measured
network-flow fields. CLEVR is a rendered image benchmark. Neither can establish
operational traffic-detection performance. The JCN draft in `JCN_submission/`
now explicitly distinguishes these evidence sources and the unexecuted flow protocol.

Both verifier pipelines now require a genuine JSON boolean for `valid`; strings
such as `"false"` are rejected. Old results predate this repair and require a
provenance review/rerun; numeric files were not overwritten. Run the regression
and traffic pipeline tests with `python -m unittest discover -s tests -v`.
