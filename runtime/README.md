# runtime/

Reads `agent/` + `techniques/` and serves requests.

- `compiler/` — turns `agent.yaml` (+ `pipeline/`) into an executable pipeline graph.
  Validates each stage against the corresponding `techniques/*/schema.yaml`.
- `executor/` — runs the compiled pipeline on a single request; emits to `traces/raw/`.
- `guards/` — runtime invariants (schema checks, rate limits, budget caps).

Not mutated by the harness. Code releases only.
