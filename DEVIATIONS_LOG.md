# Deviations Log — VLEG Experiment 3 (Iterative Refinement Ablation)

Format: paper says X, code does Y, because Z.

1. **Multi-modal episodes → structured text attribute tuples.**
   Paper says: episodes `e_i` are "high-dimensional multi-modal" (implies
   images/visual scenes with objects on backgrounds).
   Code does: each episode is a plain-text description of a
   `(background, shape, color, texture)` attribute tuple + its ground-truth
   label.
   Because: the paper's core hypothesis under test in Experiment 3 (does
   cyclic verification reduce hallucinated heuristics vs. single-step
   reflection) does not require actual image rendering or a vision-capable
   model — it requires a domain with enough structure that a "reflection"
   can be right or wrong. A symbolic attribute domain preserves that
   structure without needing a multimodal generation/annotation pipeline
   the user doesn't have running locally.

2. **Underlying LLM: GPT-4o → local Ollama model.**
   Paper says: "the same underlying Large Language Model (e.g., GPT-4o)".
   Code does: calls a local Ollama model, `qwen2.5:7b-instruct`, via
   `http://localhost:11434` for Generator/Verifier/Inference, and
   `nomic-embed-text` for retrieval embeddings.
   Because: user is running this on their own machine without a hosted
   API key; these are the models they have pulled in Ollama.

3. ~~Semantic retrieval via dense embedding cosine similarity (Eq. III.11) →
   rule-based attribute-overlap retrieval.~~ **CORRECTED**: user's Ollama
   install does have `nomic-embed-text` pulled. `core/retrieval.py` now
   implements Eq. III.11/III.12 as originally specified — real dense
   embeddings via Ollama's `/api/embeddings` endpoint, cosine similarity,
   top-k selection. No deviation remains here.

4. **Ground-truth environmental rule.**
   Paper does not specify the actual hidden rule governing the
   Experience-Transfer task (it's described only qualitatively: rules are
   context-dependent and "flip" based on background).
   Code does: defines an explicit synthetic oracle rule (see
   `core/task_env.py`) — under `background=white`, label is ANOMALY iff
   `shape=circle AND color=red`; under `background=black`, the condition
   is negated (clean flip). This is necessary to generate labeled episodes
   and to score task accuracy against ground truth.

5. **Task scale.**
   Paper's Level 3 uses "4 of 16" masked combinations out of a presumably
   much larger underlying multi-modal scene space evaluated over 500
   queries. Code uses the full 16-combination space (8 core attribute
   combos × 2 backgrounds), masks 4 for training, tests on all 16. Smaller
   scale, same structure. Test-set size is every combination once (16
   queries), not 500 — because there is no larger scene space to sample
   from in this symbolic environment.

(Log will be appended to as further deviations are made in Phase 3 tuning.)
