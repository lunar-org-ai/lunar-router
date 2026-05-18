# Contributing to OpenTracy

Thanks for taking the time. This is an experimental project — the agent DSL,
runtime APIs, and harness internals move week-to-week, so the most useful
contributions right now are:

- **Bug reports** with a minimal repro (a failing eval case is ideal).
- **New techniques** under `techniques/` — RAG, reranking, routing, and
  similar layer types are first-class plug-ins.
- **Eval suites** for new domains under `evals/suites/`.
- **Docs**, especially for the parts you found hardest to figure out.

## Workflow

1. Fork and clone.
2. Create a branch: `git checkout -b <topic>/<short-name>`.
3. Make your change, add tests, and run them:
   ```bash
   uv run pytest                         # Python
   cd backend && npm run typecheck       # TS gateway
   cd ui && npm run build                # UI build + typecheck
   ```
4. Commit with a clear subject line (we follow Conventional Commits where it
   helps, but it's not enforced).
5. Open a PR against `main`. Link any related issue.

## What stays out of the OSS repo

Deployment, infra, and tenant onboarding live in a separate private repo
(`opentracy-infra`). If your change touches Dockerfiles, Cloud Run config,
Firebase auth, or KMS wiring, it likely belongs there instead — ping us in
the issue first and we'll route it.

## Code style

- Python: ruff (config in `pyproject.toml`).
- TypeScript: `tsc --noEmit` must pass.
- No new files outside `runtime/`, `backend/`, `ui/`, `harness/`, `evals/`,
  `techniques/`, `agent/` without discussion — the directory layout has
  meaning (see the README).

## License

By contributing, you agree your work is licensed under the project's MIT
license.
