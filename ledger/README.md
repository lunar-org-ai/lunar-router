# ledger/

Append-only audit trail. The history of how the agent evolved.

- `entries/` — one row per decision/outcome.
- `chains/` — causal chains (signal → proposal → eval → approve/reject → outcome).
- `lessons/` — approved lessons (the cards rendered in `ui/`).
- `versions/` — `agent.yaml` snapshots; rollback targets.

Never edited. Append via harness API only.
