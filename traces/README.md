# traces/

Runtime accumulator. Source of signal for the loop.

- `raw/` — every request/response, append-only.
- `labeled/` — auto + human verdicts (pass / flag / fail).
- `pinned/` — flagged for learning; high-priority candidates for `evals/golden/`.

The eval substrate grows from here.
