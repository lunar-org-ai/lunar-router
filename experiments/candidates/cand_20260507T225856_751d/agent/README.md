# agent/

The trainable surface. This is to the system what `train.py` is to
karpathy/autoresearch — a single coherent editable artifact whose mutations
have measurable behavioral impact.

- `agent.yaml` — entry point; composes the pipeline.
- `pipeline/*.yaml` — per-stage configs (retrieve, rerank, route, generate, memory).
- `prompts/` — prompt templates (textual "weights").
- `custom/` — Python implementations the agent has synthesized when no existing
  technique variant fit. Last-resort path; prefer composition over custom code.

Mutable. Versioned via `ledger/versions/`.
