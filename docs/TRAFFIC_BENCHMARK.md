# Experiment 6: public flow benchmark validation

Status: implementation tested with fixtures; **no measured public-flow/model result yet**.
Synthetic test fixtures are software tests and must never be reported as experimental results.

## Scope and data

Experiments 1–5 in `core/` are symbolic scene tasks, not network flow experiments.
`core_visual/` uses rendered CLEVR objects, not production photographs or traffic.
Experiment 6 reads recorded flow features directly and tests transfer to an independent
partition. Public IDS datasets are captured/generated in controlled testbeds; this
experiment is not evidence of production deployment or zero-day detection.

Obtain the official CSVs from the dataset owner and follow its attribution terms:

- UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset
  Use `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv` (published split).
- CICIDS2017: https://www.unb.ca/cic/datasets/ids-2017.html
  Use separated periods containing both classes; do not use the benign-only Monday
  partition as the entire memory/test set. Normalize session identity across periods
  before passing `--group-column`. A Flow ID is not automatically a session identifier.

The loader uses an explicit feature allowlist; IPs, timestamps, row IDs, labels,
attack categories and session IDs are excluded from retrieval and test prompts.
Numeric missing/nonfinite values are removed and counted. Selected-feature duplicates
within a partition are removed; conflicting labels for identical selected features
cause an explicit failure and require documented resolution rather than relabeling.
Test duplicates against the **entire** memory source partition are removed before
memory subsampling. With `--group-column`, any overlapping source groups cause failure.
Without it, the supplied split must be documented and independently reviewed: the
code cannot infer session independence from feature-only CSVs.

No dataset is silently downloaded, relabeled, or replaced by synthetic examples.
Raw datasets and prepared files are ignored by git by default.

## Pilot then formal runs

Run from the repository root with Ollama serving the configured chat and embedding
models. Explicit model arguments do not alter the existing experiments' model choice.

```bash
python -m unittest discover -s tests -v
python -m traffic.experiment prepare \
  --dataset unsw \
  --memory-csv data/private/UNSW_NB15_training-set.csv \
  --test-csv data/private/UNSW_NB15_testing-set.csv \
  --split-description 'Official UNSW-NB15 published training/testing split; no additional session metadata available' \
  --memory-per-class 16 --test-limit 100 --seed 20261012 \
  --output data/private/unsw_pilot_20261012.json
python -m traffic.experiment run \
  --prepared data/private/unsw_pilot_20261012.json \
  --model llama3.3:70b --embedding-model nomic-embed-text \
  --K 3 --k 3 --num-ctx 32768 --context-chars 18000 \
  --output outputs_traffic/unsw_pilot_20261012
```

After verifying CSV cleaning, prompt sizes and model outputs, freeze the protocol
before viewing formal test results. Use new memory seeds 20261020–20261024 and
`--test-limit 1000` for formal runs. Keep pilot and formal files separate; do not
select settings based on formal results. Test sampling is fixed across memory seeds
and retains the source test class distribution in expectation (not class-balanced).
Verify actual class counts in each manifest and report them, including per-attack
counts. Report memory-seed variability; do not treat the same 1000 test flows repeated
five times as 5000 independent observations.

At 70B and 32K context, memory requirements can exceed the old short-context setup.
For a resource-constrained pilot use `--num-ctx 8192 --context-chars 6000` and prepare
smaller memory banks, e.g. `--memory-per-class 4`. Keep that budget identical across
conditions and use a new run directory. The runner rejects oversized prompts rather
than silently truncating examples. Generators/verifiers also have conservative input
limits and a response token ceiling. Compare model scales only with the same split,
features, memory seeds and context settings. Never describe CPU-offloaded timing as
GPU-only timing.

## Controlled conditions

| Condition | Construction | Inference |
|---|---|---|
| raw | Labeled historical flow records | Top-k raw examples |
| single_step | First generator proposal, no verification gating | Same top-k raw examples + first reflections |
| vleg | Up to K proposals checked against the known historical label | Verified examples + final reflections among the same top-k |
| raw_on_verified | Same VLEG admission mask | Identical surviving raw examples, no reflections |
| global_summary | One summary of the identical labeled memory set | Summary only |

These are controlled adaptations, **not faithful implementations of the published
Reflexion or HDR systems**. The symbolic repository's current Reflexion uses
multi-attempt environmental feedback, unlike the older manuscript description.

All conditions share one raw-feature embedding retrieval pass and fixed k, model,
context ceiling and target flows. This deliberately differs from reflection-driven
retrieval in the original VLEG formulation, isolating consolidation and admission.
A separate retrieval ablation would be required to evaluate that formulation.
When a VLEG-selected episode fails verification it is dropped, not replaced by a
lower-ranked neighbor; effective example count is saved. `raw_on_verified` isolates
reflection content at the same admission mask. Global-summary inference sees all
memory episodes through one summary rather than top-k examples; it is a comparison
of memory architecture, not an equal-evidence modality ablation.

Historical labels are available to the verifier and raw baselines; query labels are
used only after inference for scoring. This is **supervised historical memory**, not
unsupervised intrusion discovery. Passing a per-flow check does not validate a
universal security rule. For natural traffic, unsupported generalization needs a
manual/cross-episode audit; synthetic keyword checks cannot establish causality.

## Outputs and interpretation

Each condition reports accuracy, Macro-F1, attack recall, benign recall, false positive
rate, invalid-output count, prompt/completion tokens and mean/p95 chat latency.
Invalid answers are errors rather than silently mapped to BENIGN. False positive rate
counts only explicit MALICIOUS predictions on benign flows; pair it with invalid
rate/benign recall so abstentions cannot make a method appear successful.

`checkpoint.json` saves model/config/code/data fingerprints, memory histories,
embeddings, per-query shared retrieval timing and each completed prediction. Run the
same command to resume. Changed inputs, source code or settings require a new directory.
A crash during a single memory episode may repeat that episode; query predictions
are saved one call at a time. `summary.json` and `predictions.json` are final results;
`reflection_audit.csv` is an empty human-review worksheet, not a completed audit.

Construction costs are reported separately; single-step costs include only the
first generator call even though the shared runner also builds VLEG. Retrieval
latency is separate from chat latency. Embedding tokens are not reported by the
existing client and are not included in chat token counts. Capture Ollama version,
actual model digest/quantization, CPU/GPU/offload state and hardware before formal
runs; model names alone do not freeze weights. Model metadata is saved automatically
when the installed Ollama version exposes `/api/tags`.

For manual review, sample reflections without selecting successful cases, hide method
names where practical, use two reviewers, and report agreement plus disagreements.
Separate historical-label consistency, fabricated features, and unsupported
cross-flow generalization. Labels are ground truth for scoring, not proof that a
reflection explains the attack mechanism.

Existing symbolic results were produced before the strict-parser repair. Their
numeric values have not been rerun or silently changed. Re-evaluate affected memory
histories if available; otherwise rerun frozen old conditions before treating their
results as validated under the new parser.

### Aggregate completed formal seeds

```bash
python -m traffic.aggregate \
  outputs_traffic/unsw_formal_20261020 \
  outputs_traffic/unsw_formal_20261021 \
  outputs_traffic/unsw_formal_20261022 \
  outputs_traffic/unsw_formal_20261023 \
  outputs_traffic/unsw_formal_20261024 \
  --output outputs_traffic/unsw_formal_aggregate.json
```

The aggregator checks identical test IDs, sources, truth and model/code settings,
reports seed means/standard deviations and per-attack recall, and computes a paired
bootstrap Macro-F1 difference against raw retrieval. It resamples provided session
groups (or unique feature fingerprints when sessions are unavailable), applying the
same resample to all methods and memory seeds. This is a conditional test-cluster
interval over the fixed memory draws, not an uncertainty claim across future
networks, new memory distributions or model weights. Session-independent splitting
and sampling remain essential. Do not use a positive pilot interval as a reason to
select formal seeds or hide negative comparisons.

## Formal batch launcher and real-data preflight (2026-10-12)

The pinned mirror's CSV filenames are reversed relative to the official row counts.
`traffic.acquire_unsw` checks both file hashes and row counts, explicitly maps the
175,341-row file to historical training and the 82,332-row file to testing, and writes
its full provenance to `unsw_source_manifest.json`. Its hashes also match an
independent research deposit; the original author-hosted link currently requires
login, so this corroboration is not an author-signed authenticity guarantee.

The real CSVs contain conflicting labels for identical **selected** feature vectors.
The loader still defaults to rejecting conflicts. `--conflict-policy drop` is an
explicit alternative that removes **all** observations in conflicting groups;
it never keeps the first label, majority-votes or relabels a flow. The fixed batch
launcher uses this exclusion policy and records its scope. Consequently, its test
population is the deduplicated, unambiguous representation population, not the full
original benchmark. Filtering materially changes class balance; disclose all removal
counts and do not generalize accuracy to the original flow-frequency distribution.

The launcher uses 100 pilot queries and a different set of 1,000 formal queries.
Formal selection uses `test_offset=100` in the same frozen shuffle; the five memory
seeds share those formal queries. This fixes the earlier individual-command recipe,
where an unchanged test offset would reuse pilot queries in the formal sample.
Model inference has **not started** in the editing environment (no GPU, model files
or Ollama service). Preparation is complete; see `docs/REAL_FLOW_PREFLIGHT_20261012.md`.

Run on the actual model machine from the repository root:

```bash
python -m traffic.acquire_unsw --output data/private
# Check/pull models once if they are not installed:
ollama list
ollama pull llama3.3:70b
ollama pull nomic-embed-text
# Ollama must be serving on localhost:11434, or set VLEG_OLLAMA_HOST explicitly.
python -u -m traffic.batch \
  --output outputs_traffic/unsw_70b_formal_20261012 \
  --source-manifest data/private/unsw_source_manifest.json
```

For a detached run, create the parent directory and capture stdout/stderr:

```bash
mkdir -p outputs_traffic
nohup python -u -m traffic.batch \
  --output outputs_traffic/unsw_70b_formal_20261012 \
  --source-manifest data/private/unsw_source_manifest.json \
  > outputs_traffic/unsw_70b_formal_20261012.log 2>&1 &
```

The launcher checks the model service before inference, freezes sources/code and
settings, executes serially to avoid GPU contention, and records environment/status.
It resumes completed predictions and produces `formal_aggregate.json` only after
all five seeds finish. The planned inference count is 25,500 calls (500 pilot,
25,000 formal), plus construction and embeddings. Pilot success is an execution
check, not permission to tune against formal test outcomes. If the model service is
unavailable it exits with `blocked_no_model_service`; it does not report a running
experiment or synthetic results as real measurements.

### Git LFS download fix and Ollama alias compatibility

The CSV mirror stores its large files in Git LFS. A raw.githubusercontent.com
request returns a 133-byte pointer, not the flow data. The downloader now validates
the pointer's pinned SHA256/size, resolves the public media object, then verifies
the actual bytes with MD5, SHA256, size, required headers and row count. A download
mismatch remains an error; the checksum is not disabled or replaced by an unchecked
fallback. HTTP acquisition of both complete files was tested from an empty directory.

Ollama `/api/tags` reports an untagged embedding model as `nomic-embed-text:latest`.
Preflight now recognizes this canonical alias while preserving the actual model
digest in the run metadata. Missing or changed model objects still cause failure.
The added regression tests cover LFS resolution, changed object identifiers,
incorrect downloaded bytes and this model-name alias.

## Completed batch supersedes the preparation-only status

The node107 batch completed and its archive was independently audited on
2026-10-13. See `docs/REAL_FLOW_RESULTS_20261013.md` and
`results/traffic_20261013/audit.json`. Earlier notes describing an unavailable model
service record the editing environment's preflight; they are not the current
server-batch status. The actual experiment did not establish a VLEG advantage over
raw retrieval, and observed timing comparisons are fixed-order/cache-confounded.
