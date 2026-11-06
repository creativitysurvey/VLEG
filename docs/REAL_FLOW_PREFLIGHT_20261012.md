# Public-flow formal experiment — preparation and blocked execution

Status: **real-data preparation completed; model inference not started**.

## Data provenance

- Dataset: UNSW-NB15, published partition sizes 175,341 historical rows and 82,332 test rows.
- Download: public GitHub mirror, pinned commit `161024104f19eea98b5c9aba0499a8fcb867ba1d`.
- Mirror filenames are reversed. Roles explicitly match the official size description.
- Training-role SHA256: `bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa`.
- Test-role SHA256: `734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559`.
- MD5 values agree with the corresponding files in https://zenodo.org/records/21997233.
- Original author-hosted download redirects to a login wall. These are corroborated
  mirror copies; original-author byte verification is unavailable.

## Cleaning audit for the fixed 19-feature representation

| Stage | Historical source | Test source |
|---|---:|---:|
| Original rows | 175,341 | 82,332 |
| Conflicting selected-feature groups | 273 | 105 |
| Rows removed from those groups | 6,896 | 2,912 |
| Remaining same-label duplicate rows removed | 79,746 | 33,161 |
| Retained unique, unambiguous rows | 88,699 | 46,259 |

A further 1,044 retained test fingerprints occur in the cleaned historical source
and are removed before sampling. No label is changed or inferred during cleaning.
These exclusions restrict the estimand to a cleaned representation population;
results cannot be described as performance on the original flow-frequency distribution.

## Prepared conditions

- Historical memory: 16 benign + 16 malicious examples per seed.
- Pilot: seed 20261012, 100 test flows (70 benign, 30 malicious).
- Formal: memory seeds 20261020–20261024, shared 1,000 test flows (663 benign,
  337 malicious). Pilot and formal query fingerprints do not overlap.
- Five conditions: raw, single-step, VLEG, raw-on-verified, global summary.
- Model setting: Llama-3.3-70B, nomic-embed-text, K=3, k=3, context ceiling 32,768.
- Prepared six bundles and froze hashes; 25,500 query inference calls planned.
- No prediction, accuracy, F1 or explanation-audit result is available.

## Environment check and execution attempt

The editing environment exposes about 9.7 GiB RAM, no `nvidia-smi`, no Ollama
executable/models and no listening service on localhost:11434. The batch was
actually invoked and stopped at preflight with `blocked_no_model_service`.
The original 70B/32K-context plan cannot execute here. No remote experiment
machine or accessible model endpoint has been provided.

Nine regression/pipeline tests pass. Run instructions and resumable logs/status
are provided in `docs/TRAFFIC_BENCHMARK.md`. The next required input is an accessible
experiment machine or Ollama endpoint with the frozen models/settings. This report
is a preparation audit, not an experimental-results report.
