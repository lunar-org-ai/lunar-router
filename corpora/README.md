# corpora/

Knowledge base. RAG content + usage telemetry.

- `ingested/` — raw uploaded docs.
- `indexed/` — embedded + indexed for retrieval.
- `usage_stats/` — per-doc retrieval / citation rates. Docs that never get used
  decay in ranking; docs heavily used get prioritized.

Mutable: harness can add to `ingested/`. Indexing is automatic.
