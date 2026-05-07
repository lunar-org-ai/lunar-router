# experiments/

The "training run" workspace. Where mutations get tested before they're
allowed to touch live `agent/`.

- `candidates/<id>/` — branched copies of `agent/` with mutations applied.
  Each has a `manifest.json` recording parent, mutations, timestamp.
- `results/<YYYY-MM-DD>.jsonl` — append-only scores per candidate × suite.
  Each line is one comparison: baseline summary + candidate summary + delta.

## CLI

```bash
# Create a candidate (one or more --mutate; format: file:path=value)
uv run python -m experiments create \
  --mutate "pipeline/retrieve.yaml:knobs.k=12" \
  --description "bump rag.k to test recall" \
  --suite evals/suites/smoke_v0.yaml

# Run an existing candidate against a suite
uv run python -m experiments run cand_20260507T205359_7b13 \
  --suite evals/suites/smoke_v0.yaml

# Inspect
uv run python -m experiments list
uv run python -m experiments show cand_20260507T205359_7b13
```

## Mutation spec format

`<file>:<dotted.path>=<value>`

- `<file>` is relative to `agent/` (e.g. `pipeline/retrieve.yaml`, `agent.yaml`).
- `<dotted.path>` is a dotted key path within the YAML root.
- `<value>` is parsed as JSON when possible (`12`, `true`, `"hybrid"`), else string.

Examples:
- `pipeline/retrieve.yaml:knobs.k=12`
- `pipeline/route.yaml:knobs.confidence_threshold=0.5`
- `pipeline/rerank.yaml:variant="cross_encoder"` *(when more variants exist)*

## Promotion path

Candidate scores well → `harness/critics` pass → `harness/approver` greenlights
(auto per `policies/` or via human in `ui/`) → `harness/executor` copies the
candidate's `agent/` into the live `agent/`. Old version snapshotted to
`ledger/versions/`. None of this is wired yet — but `experiments/` is the
input to that pipeline.
