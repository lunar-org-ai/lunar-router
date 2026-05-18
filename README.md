# OpenTracy

Self-improving AI agent harness. You ship a default agent; it learns from real
usage, proposes its own improvements, and routes them through evals + human
approval before they go live.

> **Status:** experimental. APIs and the agent DSL move week-to-week. Don't pin
> to a tag yet.

## What it gives you

- A trainable agent surface at `agent/` — one YAML + a handful of Python files.
  Mutated by Claude Code (driven by the harness) in response to evidence from
  real traces.
- An **autonomous engineering loop** modeled on Lin et al.'s AHE algorithm
  (arxiv 2604.25850). The harness proposes candidate edits, critiques them,
  runs evals, and applies the winners as file-level patches with cheap rollback.
- A typed runtime that compiles `agent/` into an executable pipeline and serves
  requests over HTTP, MCP, Slack, WhatsApp, and an embeddable web widget.
- An eval suite with goldens, regression detection, and per-trace attribution
  so you can see *why* a proposed change is better (or worse).

## Quick start

Requirements: Python 3.11+, Node 20+, an Anthropic API key.

```bash
git clone https://github.com/OpenTracy/OpenTracy
cd OpenTracy

# 1. Runtime (port 8001)
uv sync               # or: pip install -e .
cp .env.example .env  # then fill in ANTHROPIC_API_KEY
uv run python -m runtime

# 2. Backend gateway (port 8002) — new terminal
cd backend && npm install && npm run dev

# 3. UI (port 5173) — new terminal
cd ui && npm install && npm run dev
```

Open <http://localhost:5173>. The shell boots straight to Evolution — no login,
no signup. OSS runs single-tenant on localhost by design.

## Architecture

| Directory | Role |
|---|---|
| `agent/` | The trainable surface. YAML + Python. Mutated by the harness. |
| `techniques/` | Catalog of layer types (RAG, reranking, routing). Read-only. |
| `runtime/` | Compiles `agent/` into a pipeline and serves requests. |
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

The loop:

```
traces/  →  evals/  →  harness/proposer/  →  harness/critics/
                              ↓
                     harness/synthesizer/  ↔  experiments/candidates/
                              ↓                     (iterate)
                     harness/approver/   →   agent/ (live)   →   traces/
```

The harness mutates `agent/`, appends to `traces/` and `ledger/` via API, and
ingests into `corpora/`. Everything else is framework. See
`config/claude_code.yaml` for the authoritative allowlist.

Languages: Python (`harness/`, `runtime/`, `evals/`, `ml/`, `techniques/`),
TypeScript (`backend/`, `ui/`).

## Distribution modes

| Mode | When | What's different |
|---|---|---|
| **OSS local** *(default)* | Clone, run for yourself or a single team. | Single-tenant. No login. Everything at the project root. |
| **Hosted/multi-tenant** | A managed deploy serving multiple orgs. Enable via `OPENTRACY_MULTI_TENANT=1`. | Per-tenant namespacing under `tenants/<id>/…`, Firebase-backed login, per-tenant Bearer tokens, KMS-encrypted BYOK keys. Requires the private `opentracy-infra` sibling repo. |

Hosted-only code is gated behind the env flag and adds zero overhead when off.

## Configuration

See [`.env.example`](.env.example) for the full list. The minimum to get
running is `ANTHROPIC_API_KEY`.

## Contributing

PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).
