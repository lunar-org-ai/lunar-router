# P15.4 — Dataset backend, autonomous curation

Datasets become first-class AHE components. Every named dataset has its
own version chain, mutations, and Lessons (`kind="dataset"`). Auto-
curation runs through the same `proposer → critic → approver →
executor → ledger` pipeline that powers `router_config` and prompt
edits — single editable surface, single decision substrate.

Inspired by AHE (`arxiv 2604.25850`) for the three-pillar component-
experience-decision shape, and reuses the KMeans clusters fitted in
P15.3 for autonomy-driving coverage analysis.

> The PLANs (`PLAN_P15.4.<n>.md`) carry implementation detail. Where
> a PLAN supersedes this roadmap, the PLAN wins; this doc is patched
> on land.

## Locked decisions (operator confirmed)

- **Mapping dataset → eval suite:** lives in suite YAMLs (existing `goldens:` list). Critic reverse-lookups "which suites use this dataset?" — zero migration on the suites side.
- **Sample storage:** the backend carries the full set per sample — `{trace_id, prompt, ground_truth?, tag, embedding, added_at, source}`. UI fetches the subset it needs (`{id, preview, tag}`).
- **Migration:** `evals/golden/*.yaml` becomes the seed dataset `goldens` (single editable surface). Suite YAMLs stop listing golden IDs and start naming a dataset.
- **Distillation:** `use: ['Distill']` is accepted on Datasets but the critic only validates `'Eval'` for now. Distill validation is explicitly deferred until a distillation pipeline exists.
- **UI:** port verbatim. The existing `Datasets` screen drives the data shape; backend matches it.
- **AHE alignment:** every change (auto + manual) goes through the existing pipeline. `kind="dataset"` is a single bucket; per-dataset overrides via `Policy.overrides["dataset"]` keep policy clean.
- **Autonomy trigger:** reuses `harness/wakeup/scheduler.maybe_fire()` from P15.3.9. Wakeup runner gets a sibling for datasets that wakes Claude Code with `dataset_health_check()` data.
- **Cluster reuse:** samples store `embedding`; `cluster_id` is recomputed on-demand against `router_config_current` when running coverage reports. No hard pin to a specific K.

## File layout (target)

```
datasets/
  <name>/
    v0.json                      [P15.4.1 — schema seed; manual placeholder]
    v<n>.json                    [P15.4.4 — fitted snapshots from auto-curation]
    current                      [symlink with .txt fallback — same pattern as router_config]
  _registry.json                 [list + metadata of named datasets]

router/data/dataset.py            [extended — Dataset + DatasetSample stay here]

harness/proposer/dataset_proposer.py   [P15.4.4 — orchestrates source mining + Proposal]
harness/proposer/dataset/
  mining/
    flagged_traces.py            [P15.4.3 — yields samples from pinned/labeled traces]
    language_router.py           [P15.4.3 — filters non-en traces]
    failed_lookups.py            [P15.4.3 — retrieve.docs_out=0 traces]
    feedback_signals.py          [P15.4.3 — stub; not yet wired]
  coverage.py                    [P15.4.4 — cluster gap analysis]

harness/critics/dataset_critic.py [P15.4.4 — reverse-lookup suites; run them]
harness/executor/promote.py      [P15.4.4 — apply_dataset_candidate + promote_dataset]

runtime/server.py                [P15.4.2 — 6 new endpoints]
backend/channels/dataset/handler.ts  [P15.4.2 — proxy]

ui/src/screens/Technical.tsx     [P15.4.2 — wire fetch into Datasets section; no layout change]

harness/introspection/lib.py     [P15.4.5 — 2 new MCP tools]
harness/introspection/agent.py   [P15.4.5 — register tools]
harness/introspection/mcp_server.py
harness/wakeup/runner.py         [P15.4.5 — extend for datasets]

tests/                           [each phase ships its tests]
```

## Dependency graph

```
P15.4.1 ──┬── P15.4.2 ──── P15.4.5
          ├── P15.4.3 ──┐
          └── ────────  P15.4.4 ── P15.4.5
```

Hard prereqs: P15.3.7+ (the AHE pipeline machinery is reused; PRs #10–#14 must be merged or stacked under).

---

## P15.4.1 — Core + storage + goldens migration

**Goal.** Land the data structures, the on-disk format, and migrate the
existing `evals/golden/*.yaml` files into the new `datasets/goldens/`
artifact. Back-compat shim in the suite loader so existing YAMLs keep
working through the transition.

**Deliverable.**
- `router/data/dataset.py` extended with `DatasetMetadata`, `DatasetSample` (full storage shape), and serialization helpers.
- `router/data/dataset_io.py` — `load_current(name)`, `save_dataset(name, payload)`, atomic write + pointer flip (mirrors `router/config_io.py`).
- `router/data/dataset_registry.py` — `_registry.json` index + CRUD on dataset names.
- `router/errors.py` extended with `DatasetNotFoundError`, `DatasetInvalidError`.
- `tools/migrate_goldens_to_dataset.py` — one-shot script that reads all `evals/golden/*.yaml` and writes `datasets/goldens/v1.json`. Suite YAMLs gain optional `dataset:` field (back-compat: `goldens: [...]` still loads via the legacy path with a deprecation warning).
- `evals/loader.py` extended to accept `dataset: <name>` in suites; reverse-lookup helper `find_suites_for_dataset(name)` for the future critic.
- Tests: storage round-trip, migration produces a valid dataset, suite loader accepts both old + new formats.

**Deps.** P15.3.1 (PromptDataset already exists; this extends it), P15.3.4 (embedder).

**DoD.** `python -m tools.migrate_goldens_to_dataset` produces `datasets/goldens/v1.json` with embedding-bearing samples and the existing `smoke_v0.yaml` suite still loads + runs after pointing at the dataset.

---

## P15.4.2 — Backend endpoints + UI wiring (verbatim)

**Goal.** Expose datasets via HTTP and wire the existing UI Datasets
screen to real data. **No UI layout changes** — same constraint as
P15.3.10.

**Deliverable.**
- `runtime/server.py`:
  - `GET  /v1/datasets`                — list, filterable by `use/owner/sourceType`. Feeds the grid.
  - `GET  /v1/datasets/{name}`         — full payload + samples + history. Feeds the drawer.
  - `GET  /v1/datasets/{name}/health`  — coverage report (cluster gap distribution).
  - `POST /v1/datasets`                — manual create via `record_manual_change(kind="dataset")`. Used by the existing UI modal.
  - `PUT  /v1/datasets/{name}`         — manual meta edit (rename/desc/use) via `record_manual_change`.
  - `DELETE /v1/datasets/{name}`       — soft delete (mark + keep versions for rollback).
- `backend/channels/dataset/handler.ts` — Hono proxies for all six.
- `ui/src/api.ts` — `getDatasets`, `getDataset`, `getDatasetHealth`, `createDataset`, `updateDataset`, `deleteDataset`.
- `ui/src/screens/Technical.tsx` (Datasets section) — `useEffect` mounts → `getDatasets()` → replaces `DATASETS_INITIAL` mock. Mock falls back if backend errors (preserve panel rule).
- Drawer Overview tab shows the dataset's `health` (coverage gaps) inside the existing `meta-grid` chrome — **no new components**.

**Deps.** P15.4.1.

**DoD.** Open `/technical/datasets` → grid populates from real backend. Click a dataset → drawer shows real samples + history. Create via modal → Lesson(kind="dataset", proposal_source="human") in `/v1/lessons`.

---

## P15.4.3 — Source-mining adapters

**Goal.** One thin adapter per `source` string the UI surfaces, each
yielding candidate samples that match the source's semantic.

**Deliverable.**
- `harness/proposer/dataset/mining/flagged_traces.py` — reads `traces/pinned/` (P16.1 already populates it). Yields samples tagged with the user's flag reason.
- `harness/proposer/dataset/mining/language_router.py` — reads `traces/raw/` and filters where `metadata.language != "en"` (best-effort; if language detection isn't wired, falls back to a heuristic).
- `harness/proposer/dataset/mining/failed_lookups.py` — filters traces where the retrieve stage produced `docs_out == 0`.
- `harness/proposer/dataset/mining/feedback_signals.py` — **stub** (deferred — UI shows the source but no production feedback channel exists yet). Honest `NotImplementedError` so callers can decide.
- Common interface: `iter_candidates(*, since_iso, embedder) -> Iterator[DatasetSample]`. Returns deduplicated samples (by `prompt_hash`) ready to add.

**Deps.** P15.3.4 (`store_adapter`), P16.1 (pinned traces).

**DoD.** Each non-stub adapter, run against current `traces/raw/` data, yields >0 candidate samples that pass the dedup hash check.

---

## P15.4.4 — Proposer + Critic + Executor

**Goal.** Wire the AHE pipeline for `kind="dataset"`. Auto-curation
walks: read dataset → pick mining adapter by `source` → cluster gap
analysis → propose additions → critic re-runs affected suites → policy
decides → executor writes new version.

**Deliverable.**
- `harness/proposer/dataset_proposer.py` — `DatasetProposer.propose(name) -> Proposal` with `kind="dataset"`. Invokes the right mining adapter; runs `coverage.cluster_gaps()` to scope the additions; emits inline candidate payload on `Mutation.value`.
- `harness/proposer/dataset/coverage.py` — `cluster_gaps(dataset, embedder, assigner) -> dict[int, int]` returns expected vs actual count per cluster.
- `harness/critics/dataset_critic.py` — `DatasetCritic` (`CriticStage.POST`). Reverse-lookups suites that name this dataset; runs each via the existing eval runner; passes when no rubric regresses below a floor + coverage gap shrinks. Distill validation is acknowledged in the verdict reason but not gated.
- `harness/executor/promote.py:apply_dataset_candidate(payload, name) + promote_dataset(outcome)` — atomic write to `datasets/<name>/v<n>.json` + pointer flip + Lesson + ledger entry. Mirrors `apply_router_candidate` from P15.3.7.
- `harness/types.py:kind_from_mutations` learns `datasets/` paths → `"dataset"`.
- `harness/executor/voices.yaml` — `dataset` voice templates.

**Deps.** P15.4.1, P15.4.3.

**DoD.** End-to-end smoke: synthetic traces in `tmp_path`, run `DatasetProposer.propose("smoke")` → `RouterCritic` ✓ → `promote_dataset()` → `versions` updated, Lesson written, history visible via `/v1/datasets/smoke`.

---

## P15.4.5 — MCP tools + autonomy + closeout

**Goal.** Close the loop. Claude Code reads dataset health, decides
whether a curation cycle is worth proposing, and the wakeup scheduler
fires it automatically every N traces.

**Deliverable.**
- `harness/introspection/lib.py`:
  - `dataset_health_check(name=None)` — single dataset or all datasets. Returns name, size, last_curation_at, coverage_gap_score, growing flag, source, use.
  - `propose_dataset_curation(name, source=None, rationale="")` — gated by Policy; runs the full pipeline (proposer → critic → policy → executor); returns typed dict same shape as `propose_router_retrain`.
- Two new tools registered in `harness/introspection/agent.py:TOOLS` and `harness/introspection/mcp_server.py`.
- `harness/wakeup/runner.py` extended (or sibling `dataset_runner.py`) — wakeup composes a prompt that exposes BOTH router health AND dataset health; brain decides which (if any) to act on.
- `harness/wakeup/prompt.py` — extended template that frames the multi-target decision.
- `ledger/decisions/dataset_<iso>.json` — same artifact pattern as `router_wakeup_*.json`.
- End-to-end live verification documented (manual; needs brain).

**Deps.** P15.4.4.

**DoD.** With `HARNESS_ROUTER_WAKEUP_N=5` and `growing=true` on the migrated `goldens` dataset, sending 5 `/run` requests fires a wakeup → Claude Code sees dataset health → either proposes (Lesson written) or skips (decision artifact written). Both paths exit cleanly.

---

## Out of scope for P15.4

- **Distillation pipeline** — `use: ['Distill']` is accepted in dataset schema and surfaced in UI, but the critic doesn't validate distill quality. Lands when a distillation runner is wired.
- **`feedback_signals` mining** — UI shows it as a source string; backend stub raises `NotImplementedError`. Lands when an explicit user-feedback channel exists in production.
- **Per-sample retention policy** — datasets grow forever in v1. Pruning / GC is a follow-up phase.
- **Multi-tenant datasets** — single global registry per project, like the router config.
- **Active learning loops** — the harness doesn't choose *which trace to label*; it ingests already-labeled signals (P16.1's flag-to-golden, language router, etc.).

## AHE alignment recap (paper 2604.25850)

| Pillar | How P15.4 maps |
|---|---|
| **Component** (mutations + versions) | `datasets/<name>/v<n>.json` versioned chain; `Mutation.file = "datasets/<name>/v<n>.json"` for dispatch |
| **Experience** (traces + delta) | Critic re-runs affected suites; the `delta` is per-rubric eval lift; Lesson carries it |
| **Decision** (voice + proposal_source + Prediction → VerificationOutcome) | `Lesson.proposal_source = "claude_code"\|"human"`; the proposer's rationale becomes the `Prediction.rationale`; dataset eval result becomes the `VerificationOutcome` |

Single editable surface, single pipeline, single ledger.
