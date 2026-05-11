# PLAN — P15.4.2 · Backend endpoints + UI wiring (verbatim)

| Field | Value |
|---|---|
| Phase | P15.4.2 |
| Parent | P15.4 (Dataset backend, autonomous curation) |
| Status | Not started |
| Depends on | P15.4.1 (storage + registry exist, `goldens` migrated) |
| Unblocks | P15.4.5 (MCP tools surface health), end of stack |
| Reference | P15.3.10 (Router UI wiring — same shape) |

## Goal

Expose datasets via HTTP and connect the existing UI **without changing
its layout**. The Datasets screen drives the data shape; the backend
matches it. Manual create / edit / delete flow through the same AHE
pipeline (`record_manual_change(kind="dataset")`).

## Scope

### In scope
- 6 endpoints on `runtime/server.py`:
  - `GET    /datasets` — list, optional `use=`/`owner=`/`sourceType=` filters.
  - `GET    /datasets/{name}` — full detail (samples + history) for the drawer.
  - `GET    /datasets/{name}/health` — coverage report (cluster gap distribution).
  - `POST   /datasets` — manual create → `record_manual_change(kind="dataset")`.
  - `PUT    /datasets/{name}` — manual meta edit (desc / use / growing) → same pipeline.
  - `DELETE /datasets/{name}` — soft delete (registry flag, files retained).
- `backend/channels/dataset/handler.ts` — Hono proxy mirroring `router/handler.ts` (`getProxy` helper, body-passing POST/PUT/DELETE).
- Wire `app.route('/v1/datasets', datasetRouter)` in `backend/api/server.ts`.
- `ui/src/api.ts` — types (`DatasetView`, `DatasetDetail`, `DatasetSampleView`, `DatasetHistoryEntry`, `DatasetHealth`) + 6 client functions.
- `ui/src/screens/Technical.tsx` — `Datasets` component switches `DATASETS_INITIAL` from a mock seed to a `useEffect` fetch. **Mock falls back if backend errors** (preserve the chrome — same rule as P15.3.10). No layout / visual changes.
- Sample shape on the wire: `{id, preview, tag}` — strip embeddings (UI doesn't need 384-dim arrays).
- `fresh` is computed server-side as a relative-time string ("3d", "6h", "just now") from `updated_at`.
- Tests:
  - `runtime/tests/test_datasets_endpoints.py` — GET list, GET detail, GET health, POST creates, PUT updates, DELETE soft-deletes. Verify cold-start (empty registry) returns `[]` cleanly.
  - Manual edits emit `Lesson(kind="dataset", proposal_source="human")` — assert via `read_lessons()`.

### Out of scope (deferred)
- Coverage report **content** beyond `cluster_distribution + sample_count` (P15.4.4 adds gap analysis).
- Sample mutation endpoints (add / remove individual samples) — comes with P15.4.4's proposer.
- Auto-curation, MCP tools (P15.4.4 / P15.4.5).
- UI changes beyond the wire-up.

## Surfaces

### Endpoint shapes

```python
# GET /datasets
class DatasetView(BaseModel):
    id: str               # equals `name` for backend storage
    name: str
    desc: str
    size: int
    source: str           # registry's `desc`-derived source label
    sourceType: str       # "auto" | "manual"
    fresh: str            # relative-time string ("3d", "6h", "just now")
    use: list[str]        # ["Eval"] | ["Eval","Distill"] | ["Distill"]
    owner: str            # "agent" | "human"
    growing: bool

# GET /datasets/{name}
class DatasetSampleView(BaseModel):
    id: str
    preview: str          # prompt truncated to 200 chars (UI shows preview, not full prompt)
    tag: Optional[str]

class DatasetHistoryEntry(BaseModel):
    when: str             # relative-time
    what: str

class DatasetDetail(DatasetView):
    samples: list[DatasetSampleView]      # up to N (first 50; full list deferred until paginate)
    history: list[DatasetHistoryEntry]

# GET /datasets/{name}/health
class DatasetHealth(BaseModel):
    name: str
    size: int
    cluster_distribution: dict[str, int]    # cluster_id (str) → count; empty when router cold-start
    coverage_gap_score: Optional[float]     # 0..1; null until P15.4.4 computes it
    last_curation_at: Optional[str]         # ISO; null until first auto-curation
```

### Create / edit / delete payloads

```python
class DatasetCreateRequest(BaseModel):
    name: str
    desc: str = ""
    source: str = "manual"
    sourceType: str = "manual"
    use: list[str] = ["Eval"]
    owner: str = "human"
    growing: bool = False

class DatasetUpdateRequest(BaseModel):
    desc: Optional[str] = None
    use: Optional[list[str]] = None
    growing: Optional[bool] = None
```

`POST` calls `record_manual_change(_writer, kind="dataset", summary=...)` where `_writer` invokes `dataset_io.save_dataset()` with a v1 payload (zero samples). `PUT` reads current, applies partial updates, bumps version, calls `record_manual_change` the same way. `DELETE` calls `dataset_registry.delete_dataset()` directly (soft-delete is not a versioned mutation; files stay on disk).

## Tasks (atomic, ordered)

### T1 — `runtime/server.py`: Pydantic models + 6 endpoints
- Add `DatasetView`, `DatasetDetail`, `DatasetSampleView`, `DatasetHistoryEntry`, `DatasetHealth`, `DatasetCreateRequest`, `DatasetUpdateRequest`.
- Add `_dataset_to_view()`, `_dataset_to_detail()` builders. Compute `fresh` via small helper `_relative_time(iso) -> str`.
- Wire the 6 routes. Use `record_manual_change` for create/update, `dataset_registry.delete_dataset()` for delete.
- Source-label derivation: registry stores `sourceType`; the `source` text for the UI is the dataset's `desc`-derived label OR the registry's `source` field if present. For `goldens`: source="manual" (registered as such).

### T2 — `backend/channels/dataset/handler.ts`
- Copy the `router/handler.ts` structure: `getProxy()` helper for the GETs, body-passing handlers for POST/PUT/DELETE. Pass through 400/404/409/422/500 status codes so the UI sees real reasons.

### T3 — Register channel in `backend/api/server.ts`
- `import { datasetRouter } from '../channels/dataset/handler'` + `app.route('/v1/datasets', datasetRouter)`.

### T4 — `ui/src/api.ts`
- Add types + 6 functions following the `getRouterConfig` / `updateRouterConfig` pattern.

### T5 — Wire `ui/src/screens/Technical.tsx` Datasets
- Inside `Datasets`, add `useEffect(() => { getDatasets().then(setDatasets).catch(() => {/* keep mock */}); }, [])`.
- For drawer open: when `openId` changes, fire `getDataset(name)` to fetch full samples + history. Cache on the row.
- Wire `handleCreate` → `createDataset()`; on success replace the optimistic add with the server response; on failure show toast.
- Wire `deleteDataset` → `deleteDataset(name)`; remove on 200.
- **Don't** change any rendering / JSX / styles. Only swap state sources.

### T6 — Tests
- `runtime/tests/test_datasets_endpoints.py`:
  - Cold-start: `GET /datasets` → `[]`.
  - Migrate goldens (test-scoped tmp dir), `GET /datasets` → 1 entry with `size=8, owner=human, sourceType=manual`.
  - `GET /datasets/goldens` → samples projected with `preview` (truncated prompt) + no embedding leak.
  - `GET /datasets/goldens/health` → returns cluster_distribution dict (empty when router cold-start).
  - `POST /datasets` with valid body → returns 201 + Lesson exists in ledger with `kind="dataset"`, `proposal_source="human"`.
  - `POST` duplicate name → 409.
  - `PUT /datasets/goldens` with `desc` → succeeds; second `GET` shows new desc; ledger has another Lesson.
  - `DELETE /datasets/goldens` → soft-deletes; next `GET /datasets` no longer lists it; v1.json still on disk.
- Re-run full suite — no regressions.

### T7 — Manual smoke
- Restart runtime + backend + vite.
- Open `http://127.0.0.1:5174` → Technical → Datasets.
- Confirm: `goldens` row appears with `size=8, owner=you, sourceType=manual`.
- Click → drawer shows 5 sample previews + 1 history entry (migration line).
- Modal "Create dataset" → fill in `test-from-ui` → submit → row appears + Lesson visible in Evolution.

## Acceptance criteria (DoD)

1. `GET /v1/datasets` returns the migrated `goldens` entry with the UI's exact shape; no embeddings in the payload.
2. `GET /v1/datasets/goldens` returns 8 samples with `preview` (prompt-truncated) + 1 history row.
3. `GET /v1/datasets/goldens/health` returns `cluster_distribution` (empty dict when router cold-start) without 5xx.
4. `POST /v1/datasets` creates a new dataset + writes a `kind="dataset"` Lesson with `proposal_source="human"`; visible in `/v1/lessons`.
5. `PUT /v1/datasets/{name}` bumps version + writes another `kind="dataset"` Lesson.
6. `DELETE /v1/datasets/{name}` soft-deletes via the registry; `v<n>.json` files stay on disk.
7. UI Datasets tab populates from real backend; mock falls back on error; no layout changes.
8. `python -m pytest` green; tests for the 6 endpoints added.

## Risks / open questions

- **`source` label decoupling.** UI's `source` field is freeform string (`"flagged traces"`, `"feedback signals"`, etc.). For the manual `goldens` dataset, we set `source="manual"` (registered that way). Auto-curated datasets will set source to the mining adapter that produced them (P15.4.4). UI doesn't care — it's just a chip.
- **Soft-delete UI surface.** Current UI fully removes a deleted dataset from state. Backend soft-delete keeps files but excludes from list. UI behavior unchanged — the list endpoint already filters; the next `GET /v1/datasets` won't return it.
- **`growing` semantic.** Backend stores it in registry. For the migrated `goldens`: `growing=false` (manual). UI reflects what backend says.
- **Lesson voice.** Use first-person voice for manual edits ("I created dataset X for Eval/Distill use cases."). Mirrors existing `record_manual_change` voices.
- **AHE alignment.** Every create/edit goes through `record_manual_change(kind="dataset")`. The dataset's own version bump (via `save_dataset`) is the mutation; the Lesson + ledger entry tie it back to the harness. Delete is *not* versioned because it's a registry flag, not a mutation of the dataset payload — same as the policy's `enabled=false` toggle.
- **No `/datasets/{name}/samples` endpoint yet.** The drawer renders the first batch baked into `GET /datasets/{name}`; pagination lands when P15.4.4's proposer starts adding samples in bulk.

## Stack-of-PRs

- Branch off `feat/p15.4.1-dataset-storage-migration` once #15 lands or remains stacked.
- New branch: `feat/p15.4.2-datasets-endpoints-ui`.
- PR title: `P15.4.2: datasets endpoints + UI wiring`.
