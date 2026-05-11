# PLAN — P15.4.1 · Core + storage + goldens migration

| Field | Value |
|---|---|
| Phase | P15.4.1 |
| Parent | P15.4 (Dataset backend, autonomous curation) |
| Status | Not started |
| Depends on | P15.3.1 (PromptDataset exists), P15.3.4 (PromptEmbedder for sample embeddings) |
| Unblocks | P15.4.2 (HTTP endpoints), P15.4.3 (mining adapters), P15.4.4 (proposer/critic) |
| Reference | none — net-new for this project, modeled after P15.3.1's scaffolding shape |

## Goal

Land the data structures and on-disk format for named datasets, and
migrate the existing `evals/golden/*.yaml` files into a single
versioned `datasets/goldens/` artifact. Suite YAMLs gain an optional
`dataset:` field; the legacy `goldens: [...]` list keeps working with
a deprecation warning so the transition is non-breaking.

This phase ships **storage + migration only**. No HTTP endpoints
(P15.4.2), no proposer (P15.4.4), no auto-curation (P15.4.5).

## Scope

### In scope
- `router/data/dataset.py` (extend) — add `DatasetMetadata`, `DatasetSample` (full storage shape), `Dataset` typed bag. The existing `PromptDataset`/`PromptSample` from P15.3.4 stay (used by the trainer); the new types are richer (carry `embedding`, `tag`, `trace_id`, etc.) and can convert down to `PromptDataset` for the trainer.
- `router/data/dataset_io.py` — `load_current(name)`, `save_dataset(name, payload, embedder=None)`, `get_current_version(name)`. Atomic JSON write via `tempfile + os.replace`; pointer is symlink with `.txt` fallback (mirrors `router/config_io.py`).
- `router/data/dataset_registry.py` — `list_datasets()`, `get_dataset_meta(name)`, `register_dataset(meta)`. Backed by `datasets/_registry.json`.
- `router/errors.py` (extend) — `DatasetNotFoundError`, `DatasetInvalidError`.
- `tools/migrate_goldens_to_dataset.py` — one-shot CLI script. Reads `evals/golden/*.yaml` → emits `datasets/goldens/v1.json` with each sample carrying its `prompt`, `ground_truth`, optional `tag` from `expected.category`, and a freshly computed `embedding`.
- `evals/loader.py` (extend) — `load_suite()` accepts both legacy `goldens: [id, ...]` and new `dataset: <name>` forms; `find_suites_for_dataset(name) -> list[str]` reverse-lookup helper for P15.4.4's critic.
- `evals/types.py` (extend) — `Suite.dataset: Optional[str] = None`. Mutually-exclusive with the legacy `goldens` field via a Pydantic validator.
- `versions/router_config_v0.json`-equivalent: `datasets/goldens/v0.json` documents the on-disk shape (emitted by the migration script for v0 only).
- Tests: storage round-trip, migration produces valid + reproducible output, suite loader accepts both formats, registry CRUD.

### Out of scope (deferred)
- HTTP endpoints (P15.4.2).
- Source-mining adapters (P15.4.3).
- DatasetProposer + DatasetCritic + executor extension (P15.4.4).
- MCP tools + autonomy (P15.4.5).
- UI changes (P15.4.2).
- Distillation pipeline integration.
- Per-sample retention / GC.

## Schema

```json
// datasets/<name>/v<n>.json
{
  "version": 3,
  "name": "goldens",
  "desc": "Eval suite goldens. Migrated from evals/golden/*.yaml.",
  "source": "manual",
  "sourceType": "manual",
  "use": ["Eval"],
  "owner": "human",
  "growing": false,
  "created_at": "2026-05-09T18:30:00Z",
  "embedder_model": "sentence-transformers/all-MiniLM-L6-v2",
  "embedding_dim": 384,
  "samples": [
    {
      "id": "smp_<short_hash>",
      "prompt": "What is your refund policy?",
      "ground_truth": "30-day refund policy.",
      "tag": "policy_lookup",
      "trace_id": null,
      "added_at": "2026-05-09T18:30:00Z",
      "source": "manual",
      "embedding": [0.012, -0.043, ...]    // length = embedding_dim
    }
  ],
  "history": [
    {"when": "2026-05-09T18:30:00Z", "what": "Migrated from evals/golden/*.yaml (8 entries)."}
  ],
  "metadata": {
    "phase": "P15.4.1",
    "migration_source": "evals/golden/"
  }
}
```

```json
// datasets/_registry.json
{
  "datasets": {
    "goldens": {
      "current_version": 1,
      "use": ["Eval"],
      "owner": "human",
      "sourceType": "manual",
      "growing": false,
      "size": 8
    }
  }
}
```

## Pre-work

- P15.3 stack merged or cherry-picked. PR #14 includes the embedder pool we'll use.
- Verify before starting: `python -c "from router.core.embeddings import PromptEmbedder, SentenceTransformerProvider; from router.data.dataset import PromptDataset"` — both work.

## Tasks (atomic, ordered)

### T1 — Add error types to `router/errors.py`
```python
class DatasetNotFoundError(RouterError):
    """No dataset with that name (or no current pointer)."""

class DatasetInvalidError(RouterError):
    """A dataset_<n>.json fails schema validation."""
```

### T2 — Extend `router/data/dataset.py` with new types
Add (don't replace) alongside existing `PromptSample`/`PromptDataset`:

```python
@dataclass
class DatasetSample:
    """Storage-shape sample for the new dataset backend."""
    id: str                       # stable: hash(prompt + tag)
    prompt: str
    ground_truth: str             # "" when not labeled
    tag: Optional[str]
    trace_id: Optional[str]       # provenance back to traces/raw/
    added_at: str                 # ISO 8601 UTC
    source: str                   # source string the UI surfaces
    embedding: list[float]        # MiniLM 384-dim by default

    def to_prompt_sample(self) -> PromptSample:
        """Down-convert for the trainer / evaluator that already speaks PromptSample."""
        return PromptSample(
            prompt=self.prompt,
            ground_truth=self.ground_truth,
            category=self.tag,
            metadata={"trace_id": self.trace_id, "source": self.source},
        )


@dataclass
class DatasetMetadata:
    """Top-level fields stored alongside samples."""
    name: str
    desc: str
    source: str
    sourceType: str               # "auto" | "manual"
    use: list[str]                # subset of {"Eval", "Distill"}
    owner: str                    # "agent" | "human"
    growing: bool
    embedder_model: str
    embedding_dim: int


@dataclass
class Dataset:
    """In-memory representation of a versioned dataset."""
    metadata: DatasetMetadata
    version: int
    samples: list[DatasetSample]
    history: list[dict]           # [{"when": iso, "what": str}, ...]
    created_at: str
    extra: dict                   # passthrough for "metadata" key

    def to_prompt_dataset(self) -> PromptDataset:
        return PromptDataset(
            [s.to_prompt_sample() for s in self.samples],
            name=self.metadata.name,
        )

    def size(self) -> int:
        return len(self.samples)
```

### T3 — `router/data/dataset_io.py`
Surface mirrors `config_io.py`:
```python
DEFAULT_DATASETS_DIR = Path("datasets")

def get_current_version(name: str, *, datasets_dir: Optional[Path] = None) -> Optional[int]: ...
def load_current(name: str, *, datasets_dir: Optional[Path] = None) -> Dataset: ...
def load_dataset_payload(name: str, version: int, *, datasets_dir: Optional[Path] = None) -> dict: ...
def save_dataset(payload: dict, *, datasets_dir: Optional[Path] = None, update_pointer: bool = True) -> Path: ...
```

Implementation notes:
- Lazy `_vd()` helper so monkeypatching `DEFAULT_DATASETS_DIR` in tests works.
- Atomic write via `tempfile.mkstemp + os.replace` in `<datasets_dir>/.staging/`.
- Pointer flip: try symlink first, `.txt` fallback when `os.symlink` raises.
- Validate the payload has the required keys (`version`, `name`, `samples`); raise `DatasetInvalidError` otherwise.

### T4 — `router/data/dataset_registry.py`
```python
def list_datasets(*, datasets_dir: Optional[Path] = None) -> list[DatasetMetadata]: ...
def get_dataset_meta(name: str, *, datasets_dir: Optional[Path] = None) -> Optional[DatasetMetadata]: ...
def register_dataset(meta: DatasetMetadata, *, datasets_dir: Optional[Path] = None) -> None: ...
def update_dataset_meta(name: str, *, size: Optional[int] = None, growing: Optional[bool] = None, ...) -> None: ...
```

`_registry.json` is the single source of truth for "what datasets exist". `save_dataset` from T3 calls into `register_dataset` to keep them in sync.

### T5 — Migration script `tools/migrate_goldens_to_dataset.py`
```
python -m tools.migrate_goldens_to_dataset
python -m tools.migrate_goldens_to_dataset --name goldens --dry-run
```

Reads every `*.yaml` in `evals/golden/`, builds a `DatasetSample` per file (prompt = `input.request`, ground_truth = `expected.exact or ""`, tag = `expected.category`, embedding via the warm `PromptEmbedder` from `runtime/embedder_pool`). Writes `datasets/goldens/v1.json` and updates the registry.

Idempotent — re-running it a second time produces identical output (same embeddings = same hash IDs).

The script also creates `datasets/goldens/v0.json` as the schema-doc placeholder (matches `router_config_v0.json` convention).

### T6 — `evals/loader.py` extension + `evals/types.py` validator
```python
class Suite(BaseModel):
    suite: str
    description: str = ""
    dataset: Optional[str] = None     # NEW — preferred form
    goldens: Optional[list[str]] = None  # legacy (deprecation warning on use)
    rubrics: list[Rubric] = []
    aggregation: ...

    @model_validator(mode="after")
    def _exactly_one_source(self):
        if self.dataset is None and not self.goldens:
            raise ValueError("Suite must specify either 'dataset' or 'goldens'")
        if self.dataset and self.goldens:
            raise ValueError("Suite cannot specify both 'dataset' and 'goldens'")
        return self
```

`load_suite(path)` resolves either form:
- `dataset: goldens` → `load_current("goldens")` → samples
- `goldens: [golden_001, ...]` → legacy path with `warnings.warn(DeprecationWarning, ...)`.

Add `find_suites_for_dataset(name) -> list[str]` that scans `evals/suites/` and returns suite names that reference the dataset. Used by P15.4.4's critic.

### T7 — Suite loader cache invalidation
Existing `evals/loader.py` likely caches goldens; verify nothing pins to the legacy path. If it does, swap the cache key to be either dataset-name or golden-id-list.

### T8 — Tests
Create `router/tests/test_dataset_io.py`:
- `test_save_load_round_trip` — full payload survives.
- `test_save_creates_versioned_files` — v1, v2, v3 each persist; current pointer flips.
- `test_load_current_cold_start_raises` — `load_current("ghost")` → `DatasetNotFoundError`.
- `test_invalid_schema_raises` — payload missing `samples` → `DatasetInvalidError`.
- `test_pointer_uses_symlink_or_txt_fallback` — works in both modes.

Create `router/tests/test_dataset_registry.py`:
- `test_register_creates_entry` — meta lands in registry.
- `test_list_datasets_excludes_deleted` — soft-deleted ones don't surface.
- `test_update_meta_persists` — partial updates merge.

Create `tools/tests/test_migrate_goldens_to_dataset.py`:
- `test_migration_produces_dataset_v1` — runs against `evals/golden/`, checks 8 samples, embeddings shape (384,).
- `test_migration_is_idempotent` — second run produces identical bytes.
- `test_migration_dry_run_prints_only` — `--dry-run` doesn't write.

Extend `evals/tests/test_loader.py` (or add):
- `test_suite_with_dataset_field` — new form loads.
- `test_suite_with_legacy_goldens_field_warns` — old form loads with warning.
- `test_suite_with_both_fields_raises` — Pydantic validator rejects.
- `test_find_suites_for_dataset` — reverse lookup works on a tmp_path suite dir.

### T9 — Validate
```
cd /Users/diogovieira/Developer/opentracy_new_mode

# tests
uv run python -m pytest router/tests/test_dataset_io.py \
    router/tests/test_dataset_registry.py \
    tools/tests/test_migrate_goldens_to_dataset.py \
    -v

# migration smoke (real data)
python -m tools.migrate_goldens_to_dataset --name goldens
ls datasets/goldens/
cat datasets/_registry.json | python -m json.tool

# back-compat — existing suite YAMLs still load + run
uv run python -m pytest evals/tests/  # if tests exist
```

## Acceptance criteria (DoD)

1. `python -m tools.migrate_goldens_to_dataset` produces `datasets/goldens/v1.json` with **8 samples** carrying real MiniLM embeddings (`shape=(384,)`).
2. `datasets/_registry.json` has a `goldens` entry with `current_version=1, size=8, owner=human, sourceType=manual`.
3. `datasets/goldens/current` (symlink or `.txt`) points at `v1.json`.
4. Re-running the migration produces **byte-identical** output (embeddings deterministic for same prompt + same model).
5. `Suite.dataset = "goldens"` form loads + runs through the existing suite runner. The legacy `Suite.goldens = [...]` form still works with a `DeprecationWarning`.
6. `find_suites_for_dataset("goldens")` returns `["smoke_v0"]` after pointing the smoke suite at the dataset.
7. `load_current("ghost")` raises `DatasetNotFoundError` cleanly.
8. `python -m pytest router/tests/test_dataset_io.py router/tests/test_dataset_registry.py tools/tests/test_migrate_goldens_to_dataset.py` is green.
9. Full project: `python -m pytest` is green; no regressions.

## Risks / open questions

- **Embedding storage size.** 8 samples × 384 floats × ~7 bytes/float (JSON) ≈ 22KB per dataset. Linear in size — a 5000-sample dataset is ~14MB JSON. Acceptable for now; if datasets grow large we move embeddings to a sidecar `.npz` like router_config did. Document the threshold.
- **Embedder model coupling.** Stored embeddings are tied to `embedder_model` + `embedding_dim`. If the embedder changes, all samples need re-embedding. The dataset metadata records both fields so a future migration can detect mismatch and re-embed lazily. Out of scope for P15.4.1.
- **Suite YAML compat.** Migrating one of the existing `evals/suites/*.yaml` to the new `dataset:` form is needed to verify the loader path. Plan: migrate `smoke_v0.yaml` as part of T6 and keep the others on the legacy path.
- **Idempotent migration claim.** SentenceTransformer outputs are deterministic for the same input + same model + same device, but tokenizer version changes can shift bits. The "byte-identical" DoD is a strong claim — if a future tokenizer bump perturbs values, replace this assertion with "shape + first-3 decimals match".
- **Atomic flip on existing platforms.** macOS + Linux dev boxes do symlinks fine. CI in containers may not. The `.txt` fallback path from `config_io.py` works; copy-paste that helper.
- **`goldens` is reserved as the migration target name.** If an operator already has `datasets/goldens/v0.json` somewhere, the script aborts with a clear error — no silent overwrite.
