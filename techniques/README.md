# techniques/

Catalog of "layer types" available to `agent/`. Each technique exposes:

- `schema.yaml` — knobs the technique accepts (validates against `agent/`).
- `variants/` — alternative implementations (intercambiáveis via `variant:` em agent.yaml).
- `impl.py` — fixed framework implementation; not mutated by the loop.

To add a technique: write `schema.yaml`, drop variant impls in `variants/`, register
it as importable in `runtime/compiler/`. Implementation changes are framework
releases, not auto-improvements.

Read-only from the loop's perspective. Humans add techniques; the loop selects them.
