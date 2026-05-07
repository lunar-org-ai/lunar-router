# experiments/

The "training run" workspace.

- `candidates/` — branched `agent/` configs the harness is testing.
- `results/` — append-only scores per candidate × suite. Audit trail for the loop.

Promotion path: candidate scores well → `harness/critics` pass → `harness/approver`
greenlights (auto or human) → `harness/executor` copies into `agent/` (live).
