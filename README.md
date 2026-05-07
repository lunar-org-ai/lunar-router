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
