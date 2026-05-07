# ml/

Models trained on accumulated data. These are *not* the LLM — they're auxiliary
models that turn `traces/` + `evals/golden/` into knobs for `agent/`.

- `embeddings/` — embedding models + indices.
- `rerankers/` — trained rerankers (replaces black-box reranker once enough data).
- `classifiers/` — intent / failure / safety classifiers.
- `training/` — pipelines that consume `traces/labeled/` + `evals/golden/`.
- `registry/` — versioned model artifacts. Each version becomes a selectable
  variant inside `techniques/*/variants/`.

Mostly dormant on day 1; activates as data accumulates.
