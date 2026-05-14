# opentracy_new_mode

A self-improving AI agent harness. Ships as a default, learns from real usage,
and proposes its own improvements — gated by evals and human approval.

## Inspirations (filtered, not copied)
- **AutoHarness** (Lou et al., arxiv 2603.03329) — iterative synthesis with environmental feedback.
- **karpathy/autoresearch** — single high-leverage editable surface + clean eval signal.

## The 13 pillars

| Pillar | Role |
|---|---|
| `agent/` | The trainable surface. YAML + Python. Mutated by Claude Code in the loop. |
| `techniques/` | Catalog of "layer types" (RAG, reranking, routing, …). Read-only. |
| `runtime/` | Compiles `agent/` into an executable pipeline and serves requests. |
| `evals/` | The loss function. Goldens, suites, runners, attribution. |
| `experiments/` | Candidate configs + results. The training workspace. |
| `harness/` | The optimizer: proposer, critics, approver, executor, rollback. |
| `ml/` | Models trained on accumulated data. |
| `ledger/` | Append-only audit trail. |
| `traces/` | Runtime accumulator (conversations, labels, pins). |
| `corpora/` | Knowledge accumulator (RAG content with usage stats). |
| `policies/` | Human-set rules for the harness. |
| `backend/` | Request-serving layer (API, channels). |
| `connectors/` | Outbound integrations. |
| `ui/` | Frontend (React + Vite + TS). |

## The boundary
Claude Code mutates `agent/`, appends to `traces/` and `ledger/` via API, ingests
into `corpora/`. Everything else is framework. See `config/claude_code.yaml` for
the authoritative allowlist.

## The loop
```
traces/  →  evals/  →  harness/proposer/  →  harness/critics/
                              ↓
                     harness/synthesizer/  ↔  experiments/candidates/
                              ↓                     (iterate)
                     harness/approver/   →   agent/ (live)   →   traces/
```

## Languages
- Python: `harness/`, `runtime/`, `evals/`, `ml/`, `techniques/`
- TypeScript: `backend/`, `ui/`

## Distribution modes

| Mode | When | What's different |
|---|---|---|
| **OSS local** (default) | You clone this repo, run it for yourself or a single team. | Everything stays at the project root. No tenants, no admin tokens, no migration. Same as it's always been. |
| **Hosted/infra** (`OPENTRACY_MULTI_TENANT=1`) | A managed deploy serving multiple customer orgs (driven by the private `opentracy-infra` sibling repo). | Per-tenant namespacing under `tenants/<id>/…`, per-tenant Bearer tokens, separate admin gate. See [docs/multi-tenant.md](docs/multi-tenant.md). |

The OSS distribution is the default. Hosted-only features are gated by
the env var and add zero overhead when off.
