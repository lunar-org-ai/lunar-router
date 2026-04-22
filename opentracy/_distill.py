"""One-call distillation: `ot.distill(dataset=..., teacher=..., student=...) -> Student`.

This module wraps the full 4-phase distillation pipeline
(data-generation → curation → training → export) so callers don't need to
stand up a FastAPI service, a ClickHouse DB, or a tenant. It's the entry
point for users who just want to train a student from a dataset file and
get back a callable :class:`opentracy.Student`.

Example
-------
::

    import opentracy as ot

    student = ot.distill(
        dataset="tickets.jsonl",
        teacher="openai/gpt-4o",
        student="llama-3.2-1b",
        steps=60,
    )

    print(student("Classify: 'Please refund me'"))
    student.save("./my-classifier")

Design notes
~~~~~~~~~~~~
* Heavy deps (torch, transformers, peft, trl, unsloth) are only imported by
  the subprocess training phase — importing this module on the base wheel
  is safe.
* The distillation pipeline expects a ``repository`` module backed by
  ClickHouse. We swap that at runtime for an in-memory dict via
  :func:`_patch_repo` so nothing touches the DB when ``distill()`` runs
  standalone.
* Teacher calls go through the bundled Go engine. If one isn't already
  running at ``OPENTRACY_ENGINE_URL`` we spawn :class:`opentracy.engine.GoEngine`
  for the duration of the job and tear it down at the end.
* ``on_progress`` is called with ``{job_id, phase, status, progress, log}``
  dicts every time the pipeline transitions — roughly once per phase plus
  any log lines.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Union

from ._env import env, env_in_environ
from .student import Student, StudentError

logger = logging.getLogger(__name__)

__all__ = ["distill", "DistillError"]


class DistillError(RuntimeError):
    """Raised when the distillation pipeline fails."""


ProgressFn = Callable[[dict], None]
Dataset = Union[str, "os.PathLike[str]", list[Mapping[str, Any]], Callable[[], Iterable[Mapping[str, Any]]]]


# ---------------------------------------------------------------------- #
# Public API
# ---------------------------------------------------------------------- #


def distill(
    dataset: Dataset,
    *,
    teacher: str = "openai/gpt-4o",
    student: str = "llama-3.2-1b",
    num_prompts: Optional[int] = None,
    steps: int = 500,
    n_samples: int = 4,
    bond_beta: float = 0.5,
    bond_gamma: float = 0.1,
    temperature: float = 0.8,
    output_dir: Optional[Union[str, "os.PathLike[str]"]] = None,
    quantize: Optional[Union[str, list[str]]] = "q4_k_m",
    engine_url: Optional[str] = None,
    on_progress: Optional[ProgressFn] = None,
) -> Student:
    """Run the 4-phase distillation pipeline in-process and return a :class:`Student`.

    Args:
        dataset: One of —
            * A path to a ``.jsonl`` / ``.json`` file with rows containing
              ``prompt`` (or ``input`` / ``text``) and optional ``response``.
            * A list of dicts in the same shape.
            * A zero-arg callable that yields such dicts (for streaming from
              traces or a generator).
        teacher: Provider-prefixed teacher model (``"openai/gpt-4o"``).
        student: Student model key — accepts either a short alias
            (``"llama-3.2-1b"``) or a full HF repo id. Aliases are mapped via
            :data:`opentracy.distillation.schemas.STUDENT_MODEL_MAP`.
        num_prompts: Optional cap on the number of prompts used. Defaults to
            all of them.
        steps: Fine-tune training steps. Small datasets need fewer.
        n_samples: Best-of-N candidates per prompt in the BOND phase.
        bond_beta, bond_gamma, temperature: BOND hyperparameters. Defaults
            are good for classification-style tasks.
        output_dir: Where artifacts land. Defaults to a fresh temp dir.
        quantize: GGUF quantization(s) to export. ``None`` skips the GGUF
            export entirely and returns a PEFT-backed Student. A string is
            shorthand for ``[string]``.
        engine_url: Override the Go engine URL. If unset, an engine is
            spawned for the duration of this call.
        on_progress: Callback invoked on every pipeline transition with a
            dict ``{"job_id", "phase", "status", "progress", "log"}``.

    Returns:
        A :class:`Student` wrapping the freshest artifact — GGUF if one was
        exported, else the PEFT adapter directory.

    Raises:
        DistillError: If any phase fails. The pipeline's error message is
            attached.
    """
    # Late imports — these only exist when the `[distill]` extra is installed.
    try:
        from .distillation import pipeline as _pipeline_mod
        from .distillation.schemas import DistillationConfig, resolve_student_model
    except ImportError as e:  # pragma: no cover - import error path
        raise DistillError(
            "Distillation requires the `[distill]` extra. "
            "Install with: pip install opentracy[distill]"
        ) from e

    # Normalize + validate dataset BEFORE preflight so "empty dataset" surfaces
    # as a tidy DistillError even in test envs that don't have torch installed.
    prompts = _normalize_dataset(dataset)
    if num_prompts is not None:
        prompts = prompts[: max(1, int(num_prompts))]
    if not prompts:
        raise DistillError("Dataset is empty — distill() needs at least one prompt.")

    # Preflight: the training subprocess imports torch. If it's missing we'd
    # blow through the teacher + judge phases (real OpenAI spend) and fail
    # at phase 3 anyway. Fail fast before spending any money.
    # Skip-hook: tests that monkey-patch the pipeline don't actually touch
    # torch, so they can set `OPENTRACY_SKIP_DISTILL_PREFLIGHT=1` to bypass.
    if not env("SKIP_DISTILL_PREFLIGHT"):
        _preflight_training_env()

    quant_list = _normalize_quant(quantize)
    export_gguf = bool(quant_list)

    base_model = resolve_student_model(student)
    config = DistillationConfig(
        base_model=base_model,
        teacher_model=teacher,
        n_samples=n_samples,
        num_prompts=len(prompts),
        training_steps=steps,
        bond_beta=bond_beta,
        bond_gamma=bond_gamma,
        temperature=temperature,
        export_gguf=export_gguf,
        quantization_types=quant_list,
        prompts=prompts,
        student_model=student,
    )

    out_dir = Path(output_dir).expanduser().resolve() if output_dir else Path(tempfile.mkdtemp(prefix="opentracy-distill-"))
    out_dir.mkdir(parents=True, exist_ok=True)

    job_id = f"local-{uuid.uuid4().hex[:8]}"
    tenant_id = "local"

    logger.info("opentracy.distill: job=%s prompts=%d steps=%d", job_id, len(prompts), steps)

    repo_state = _InMemoryRepo(on_progress=on_progress)
    with _patch_repo(_pipeline_mod, repo_state), \
         _ensure_engine(engine_url), \
         _scoped_env("DATA_DIR", str(out_dir)), \
         _scoped_env("OPENTRACY_DATA_DIR", str(out_dir)):

        # Seed the repo with the job entry the sub-phases expect.
        repo_state.seed_job(job_id, tenant_id, config)

        try:
            _run_coro_sync(
                lambda: _pipeline_mod._run_pipeline(job_id, tenant_id, config)
            )
        except Exception as e:  # pragma: no cover - pipeline wraps its own errors
            raise DistillError(f"distillation failed: {e}") from e

        job = repo_state.jobs[(tenant_id, job_id)]
        artifacts = job.get("artifacts") or {}

        if job["status"] != "completed":
            # Export can fail (missing llama.cpp, OOM, etc.) after the
            # adapter is already trained and saved to disk. That's still a
            # usable result — return a PEFT-backed Student. The caller
            # didn't get GGUF, but they got a working model.
            adapter_path = artifacts.get("adapter_path") if isinstance(artifacts, dict) else None
            if adapter_path and Path(adapter_path).exists():
                logger.warning(
                    "opentracy.distill: pipeline status=%s but adapter exists at %s — "
                    "returning PEFT Student (GGUF export skipped: %s)",
                    job["status"], adapter_path, job.get("error", "<no error>"),
                )
                return Student(
                    backend="peft",
                    model_path=str(Path(adapter_path).resolve()),
                    base_model=base_model,
                )
            err = job.get("error") or f"pipeline finished with status={job['status']}"
            raise DistillError(err)

    return _student_from_artifacts(
        artifacts, base_model=base_model, prefer_gguf=export_gguf, quant=quant_list,
    )


# ---------------------------------------------------------------------- #
# Preflight
# ---------------------------------------------------------------------- #


def _preflight_training_env() -> None:
    """Fail fast if the training subprocess will crash on torch / GPU.

    The pipeline spends real money on teacher + judge API calls BEFORE the
    training subprocess starts. Catching missing torch / no-CUDA up front
    avoids dead-in-the-water runs that already cost the user $$$.
    """
    try:
        import torch  # noqa: F401
    except ImportError as e:
        raise DistillError(
            "Training needs PyTorch, but `import torch` failed. "
            "Reinstall the distill extras with a CUDA-enabled environment: "
            "`pip install --upgrade opentracy[distill]`. "
            "For quick prototyping on Colab: Runtime \u2192 Change runtime type \u2192 T4 GPU."
        ) from e

    try:
        cuda_ok = bool(torch.cuda.is_available())
    except Exception:
        cuda_ok = False
    if not cuda_ok:
        raise DistillError(
            "No CUDA GPU is visible to PyTorch. The training phase uses "
            "unsloth which is CUDA-only, so this pipeline cannot complete "
            "on a CPU-only host. On Colab: Runtime \u2192 Change runtime type \u2192 T4 GPU. "
            "On your own infra: run the notebook on a machine with an NVIDIA "
            "GPU + matching PyTorch build."
        )


# ---------------------------------------------------------------------- #
# Async pipeline dispatch
# ---------------------------------------------------------------------- #


def _run_coro_sync(coro_factory: Callable[[], Any]) -> Any:
    """Run an async coroutine synchronously, robust to Jupyter/IPython.

    ``asyncio.run()`` raises ``RuntimeError`` when invoked from a thread
    that already has a running event loop — which is exactly what the
    IPython/Jupyter kernel gives us. When that happens, we spin a
    dedicated worker thread with a fresh loop and block on it. Outside
    a running loop we use ``asyncio.run()`` directly so behaviour is
    unchanged for regular scripts.

    Args:
        coro_factory: Zero-arg callable returning a fresh coroutine.
            Taking a factory (rather than a pre-built coroutine) lets
            us defer creation until we're inside the target loop's
            thread, keeping the coroutine's binding tidy.
    """
    try:
        asyncio.get_running_loop()
        inside_loop = True
    except RuntimeError:
        inside_loop = False

    if not inside_loop:
        return asyncio.run(coro_factory())

    holder: dict[str, Any] = {}

    def _worker() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            holder["value"] = loop.run_until_complete(coro_factory())
        except BaseException as exc:  # propagate back to the caller
            holder["error"] = exc
        finally:
            loop.close()

    t = threading.Thread(target=_worker, name="opentracy-distill", daemon=False)
    t.start()
    t.join()
    if "error" in holder:
        raise holder["error"]
    return holder.get("value")


# ---------------------------------------------------------------------- #
# Dataset normalization
# ---------------------------------------------------------------------- #


def _normalize_dataset(dataset: Dataset) -> list[dict[str, Any]]:
    """Turn any of the accepted dataset forms into a list of {id, text} dicts."""
    if callable(dataset):
        dataset = list(dataset())

    if isinstance(dataset, (str, os.PathLike)):
        path = Path(dataset).expanduser()
        if not path.exists():
            raise DistillError(f"Dataset file not found: {path}")
        return _load_dataset_file(path)

    if isinstance(dataset, list):
        return [_normalize_row(row, i) for i, row in enumerate(dataset)]

    raise DistillError(
        f"Unsupported dataset type: {type(dataset).__name__}. "
        "Pass a path, a list of dicts, or a callable."
    )


def _load_dataset_file(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    elif suffix == ".json":
        data = json.loads(path.read_text())
        rows = data if isinstance(data, list) else [data]
    else:
        raise DistillError(
            f"Unsupported dataset file format: {suffix}. Use .jsonl or .json."
        )
    return [_normalize_row(r, i) for i, r in enumerate(rows)]


def _normalize_row(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Accept row shapes:  {prompt, response}, {input, expected_output}, {text}."""
    if not isinstance(row, Mapping):
        raise DistillError(f"Row {index} is not a dict: {row!r}")
    text = row.get("prompt") or row.get("input") or row.get("text")
    if not text:
        raise DistillError(
            f"Row {index} has no 'prompt' / 'input' / 'text' field: {row!r}"
        )
    out = {
        "id": str(row.get("id") or index),
        "text": str(text),
        "system": str(row.get("system") or ""),
    }
    # Preserve the gold response (used by evaluation / curation downstream).
    response = row.get("response") or row.get("expected_output")
    if response is not None:
        out["response"] = str(response)
    return out


def _normalize_quant(quantize: Optional[Union[str, list[str]]]) -> list[str]:
    if quantize is None or quantize == "":
        return []
    if isinstance(quantize, str):
        return [quantize]
    return list(quantize)


# ---------------------------------------------------------------------- #
# In-memory repo (replaces the ClickHouse-backed `repository` module for
# the duration of a standalone distill() call)
# ---------------------------------------------------------------------- #


class _InMemoryRepo:
    """Minimal in-memory stand-in for ``opentracy.distillation.repository``.

    Implements the subset of functions the pipeline touches:
      - Jobs: ``update_job_status``, ``append_log``, ``update_job``,
        ``get_job``.
      - Candidates: ``insert_candidates`` (batch from data_gen),
        ``insert_candidate`` (single upsert from curation),
        ``get_candidates`` (read-back for the judge phase).
      - Metrics: ``record_training_metric`` + ``get_latest_metric`` +
        ``list_metrics``. The pipeline tolerates missing rows, so
        these are cheap — metrics land in a list keyed by ``job_id``.
    """

    def __init__(self, *, on_progress: Optional[ProgressFn]) -> None:
        self.jobs: dict[tuple[str, str], dict[str, Any]] = {}
        # Candidates are keyed by (tenant_id, job_id). Curation reads them
        # back via get_candidates(job_id, limit=N), so we also index by
        # candidate_id to let insert_candidate() upsert scores.
        self.candidates: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self._candidate_idx: dict[str, tuple[tuple[str, str], int]] = {}
        self.metrics: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.RLock()
        self._on_progress = on_progress

    # ---- Public API consumed by the pipeline ---------------------------- #

    def seed_job(self, job_id: str, tenant_id: str, config: Any) -> None:
        cfg_dict = config.model_dump() if hasattr(config, "model_dump") else dict(config)
        with self._lock:
            self.jobs[(tenant_id, job_id)] = {
                "job_id": job_id,
                "tenant_id": tenant_id,
                "status": "pending",
                "phase": "initializing",
                "config": cfg_dict,
                "progress": {
                    "data_generation": {"status": "pending", "progress": 0},
                    "curation": {"status": "pending", "progress": 0},
                    "training": {"status": "pending", "progress": 0},
                    "export": {"status": "pending", "progress": 0},
                },
                "results": {},
                "artifacts": {},
                "pipeline_logs": [],
                "error": "",
            }

    def update_job_status(
        self,
        tenant_id: str,
        job_id: str,
        *,
        status: Optional[str] = None,
        phase: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        with self._lock:
            job = self.jobs.get((tenant_id, job_id))
            if job is None:
                return
            if status is not None:
                job["status"] = status
            if phase is not None:
                job["phase"] = phase
            if error is not None:
                job["error"] = error
        self._emit(tenant_id, job_id, log=None)

    def append_log(self, tenant_id: str, job_id: str, line: str) -> None:
        with self._lock:
            job = self.jobs.get((tenant_id, job_id))
            if job is None:
                return
            job["pipeline_logs"].append(line)
        logger.info("[distill %s] %s", job_id, line)
        self._emit(tenant_id, job_id, log=line)

    def update_job(
        self,
        tenant_id: str,
        job_id: str,
        patch: dict[str, Any],
    ) -> None:
        with self._lock:
            job = self.jobs.get((tenant_id, job_id))
            if job is None:
                return
            for k, v in patch.items():
                job[k] = v
        self._emit(tenant_id, job_id, log=None)

    def get_job(self, tenant_id: str, job_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            job = self.jobs.get((tenant_id, job_id))
            return dict(job) if job else None

    # ---- Candidates ---------------------------------------------------- #
    # Shape matches ``opentracy.distillation.repository`` — each candidate is
    # a dict with at least candidate_id / job_id / tenant_id / prompt_id /
    # prompt / response / score / selected.

    def insert_candidates(self, candidates: list[dict[str, Any]]) -> None:
        """Batch insert. Called once per prompt-batch by data_gen."""
        if not candidates:
            return
        with self._lock:
            for c in candidates:
                key = (c.get("tenant_id", "local"), c.get("job_id", ""))
                bucket = self.candidates.setdefault(key, [])
                bucket.append(dict(c))
                cid = c.get("candidate_id")
                if cid:
                    self._candidate_idx[cid] = (key, len(bucket) - 1)

    def insert_candidate(self, candidate: dict[str, Any]) -> None:
        """Upsert a single candidate. Curation uses this to write the score
        back onto a row that data_gen already inserted.
        """
        cid = candidate.get("candidate_id")
        with self._lock:
            if cid and cid in self._candidate_idx:
                key, pos = self._candidate_idx[cid]
                self.candidates[key][pos].update(candidate)
                return
            # Fall through: treat as a new row.
            key = (candidate.get("tenant_id", "local"), candidate.get("job_id", ""))
            bucket = self.candidates.setdefault(key, [])
            bucket.append(dict(candidate))
            if cid:
                self._candidate_idx[cid] = (key, len(bucket) - 1)

    def get_candidates(
        self,
        job_id: str,
        limit: int = 100_000,
        tenant_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Return candidates for a job. ``tenant_id`` defaults to the only
        tenant we seeded when ``get_candidates(job_id)`` is called without it.
        """
        with self._lock:
            if tenant_id is not None:
                return list(self.candidates.get((tenant_id, job_id), []))[:limit]
            # No tenant filter: concat across whichever buckets match job_id.
            out: list[dict[str, Any]] = []
            for (t, j), bucket in self.candidates.items():
                if j == job_id:
                    out.extend(bucket)
                    if len(out) >= limit:
                        return out[:limit]
            return out

    # Kept as a lenient alias for older call-sites. Accepts the empty/no-arg
    # form so legacy no-op tests (which predate the plural insert_candidates
    # rename) keep passing.
    def append_candidates(
        self,
        candidates: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        if candidates:
            self.insert_candidates(candidates)

    def list_candidates(
        self,
        job_id: Optional[str] = None,
        limit: int = 100,
        tenant_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        if job_id is None:
            return []
        return self.get_candidates(job_id, limit=limit, tenant_id=tenant_id)

    # ---- Metrics ------------------------------------------------------- #

    def record_training_metric(
        self,
        job_id: Optional[str] = None,
        metric: Optional[dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
    ) -> None:
        """Append a training metric row. Pipeline writes one per optimizer
        step; we keep them in order of insertion which is also step order.

        Args are nominally required, but defaults are permitted so the legacy
        no-op test shape ``r.record_training_metric()`` still works.
        """
        if job_id is None or metric is None:
            return
        row = dict(metric)
        row.setdefault("job_id", job_id)
        if tenant_id is not None:
            row.setdefault("tenant_id", tenant_id)
        with self._lock:
            self.metrics.setdefault(job_id, []).append(row)

    # Alias for the REST-style name used by some callers.
    def record_metrics(
        self,
        job_id: Optional[str] = None,
        metric: Optional[dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
    ) -> None:
        if job_id is None or metric is None:
            return
        self.record_training_metric(job_id, metric, tenant_id=tenant_id)

    def list_metrics(
        self,
        job_id: Optional[str] = None,
        limit: int = 5000,
    ) -> list[dict[str, Any]]:
        if job_id is None:
            return []
        with self._lock:
            return list(self.metrics.get(job_id, []))[-limit:]

    def get_latest_metric(self, job_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            rows = self.metrics.get(job_id)
            return dict(rows[-1]) if rows else None

    # ---- Internal --------------------------------------------------- #

    def _emit(self, tenant_id: str, job_id: str, *, log: Optional[str]) -> None:
        if self._on_progress is None:
            return
        job = self.jobs.get((tenant_id, job_id))
        if not job:
            return
        self._on_progress({
            "job_id": job_id,
            "phase": job.get("phase"),
            "status": job.get("status"),
            "progress": job.get("progress", {}),
            "log": log,
        })


@contextlib.contextmanager
def _patch_repo(pipeline_mod: Any, repo_state: _InMemoryRepo):
    """Swap ``distillation.repository`` for the in-memory version everywhere
    the pipeline imports it from. The patch is limited to the duration of
    the ``with`` block.

    Submodules (``data_gen``, ``curation``, ``trainer``, ``export``) all
    imported ``from . import repository as repo`` at import time, so we
    patch those module-level names too.
    """
    from opentracy import distillation

    patched: list[tuple[Any, str, Any]] = []

    def _swap(module: Any, attr: str) -> None:
        original = getattr(module, attr, _SENTINEL)
        patched.append((module, attr, original))
        setattr(module, attr, repo_state)

    _swap(pipeline_mod, "repo")
    for sub in ("data_gen", "curation", "trainer", "export"):
        mod = getattr(distillation, sub, None)
        if mod is None:  # submodule not imported yet — force-load it
            mod = __import__(f"opentracy.distillation.{sub}", fromlist=["repo"])
        if hasattr(mod, "repo"):
            _swap(mod, "repo")

    try:
        yield
    finally:
        for module, attr, original in patched:
            if original is _SENTINEL:
                delattr(module, attr)
            else:
                setattr(module, attr, original)


_SENTINEL = object()


# ---------------------------------------------------------------------- #
# Engine lifecycle
# ---------------------------------------------------------------------- #


@contextlib.contextmanager
def _ensure_engine(engine_url: Optional[str]):
    """Ensure a Go engine is reachable while the pipeline runs.

    Priority:
      1. Caller-provided ``engine_url`` — we only set the env var.
      2. Environment already has ``OPENTRACY_ENGINE_URL`` / ``LUNAR_ENGINE_URL``
         and a ``/health`` probe succeeds → use it.
      3. Spawn :class:`opentracy.engine.GoEngine` locally for the duration.
    """
    if engine_url:
        with _scoped_env("OPENTRACY_ENGINE_URL", engine_url):
            yield engine_url
        return

    existing = env("ENGINE_URL") if env_in_environ("ENGINE_URL") else None
    if existing and _engine_healthy(existing):
        yield existing
        return

    from .engine import GoEngine
    logger.info("opentracy.distill: spawning local Go engine (no OPENTRACY_ENGINE_URL set)")
    engine = GoEngine()
    try:
        engine.start()
        with _scoped_env("OPENTRACY_ENGINE_URL", engine.base_url):
            yield engine.base_url
    finally:
        engine.stop()


def _engine_healthy(url: str, timeout: float = 2.0) -> bool:
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(f"{url.rstrip('/')}/health")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return 200 <= r.status < 300
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


@contextlib.contextmanager
def _scoped_env(key: str, value: str):
    """Set ``os.environ[key] = value`` for the duration of the ``with`` block."""
    prev = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev


# ---------------------------------------------------------------------- #
# Artifacts → Student
# ---------------------------------------------------------------------- #


def _student_from_artifacts(
    artifacts: dict[str, Any],
    *,
    base_model: str,
    prefer_gguf: bool,
    quant: list[str],
) -> Student:
    """Build a :class:`Student` from the pipeline's artifacts dict."""
    gguf_paths = artifacts.get("gguf_paths") or {}
    adapter_path = artifacts.get("adapter_path")

    if prefer_gguf and gguf_paths:
        # Prefer the user's first-requested quant, else the smallest file.
        chosen = None
        for q in quant:
            if q in gguf_paths and Path(gguf_paths[q]).exists():
                chosen = gguf_paths[q]
                break
        if chosen is None:
            existing = [p for p in gguf_paths.values() if Path(p).exists()]
            if existing:
                chosen = min(existing, key=lambda p: Path(p).stat().st_size)
        if chosen:
            return Student(
                backend="gguf",
                model_path=str(Path(chosen).resolve()),
                base_model=base_model,
            )

    if adapter_path and Path(adapter_path).exists():
        return Student(
            backend="peft",
            model_path=str(Path(adapter_path).resolve()),
            base_model=base_model,
        )

    raise DistillError(
        f"Pipeline reported success but no usable artifacts were found. "
        f"adapter_path={adapter_path!r} gguf_paths={gguf_paths!r}"
    )
