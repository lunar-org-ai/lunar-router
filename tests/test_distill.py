"""Tests for opentracy.distill (Phase 2 of the SDK-DX plan).

The real 4-phase pipeline needs a GPU, a teacher API key, and minutes of
training. We don't run it end-to-end in unit tests. Instead we verify the
harness around it:

  1. Dataset normalization — jsonl / json / list[dict] / callable, and row
     shape coercion across the prompt/input/text aliases.
  2. The in-memory repo satisfies the pipeline's `repository` interface
     and surfaces progress to ``on_progress``.
  3. ``distill()`` builds a correct :class:`DistillationConfig`, calls the
     pipeline exactly once, reads artifacts, and returns a :class:`Student`
     of the right backend.
  4. Error paths: empty dataset, failed pipeline, bad dataset type.
  5. Engine-ensure logic — if ``OPENTRACY_ENGINE_URL`` is already healthy
     we don't spawn a new engine.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import opentracy as ot
import opentracy._distill as distill_mod
import opentracy.distillation.pipeline as pipeline_mod
from opentracy._distill import (
    DistillError,
    _InMemoryRepo,
    _normalize_dataset,
    _normalize_quant,
    _normalize_row,
    _student_from_artifacts,
    distill,
)
from opentracy.student import Student


# Stand-in for _ensure_engine — returns a no-op context manager yielding a URL.
import contextlib


@contextlib.contextmanager
def _fake_engine_ctx(engine_url=None):
    yield engine_url or "http://fake-engine:8080"


@pytest.fixture(autouse=True)
def _skip_distill_preflight(monkeypatch):
    """CI runs these tests without torch installed. The production preflight
    check in ``distill()`` would raise DistillError before the mocked pipeline
    ever runs, hiding the real assertions each test wants to make. The escape
    hatch is an env var the SDK honours."""
    monkeypatch.setenv("OPENTRACY_SKIP_DISTILL_PREFLIGHT", "1")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def jsonl_dataset(tmp_path: Path) -> Path:
    """A JSONL dataset with mixed row shapes and enough rows to satisfy
    DistillationConfig's `num_prompts >= 10` constraint."""
    path = tmp_path / "data.jsonl"
    rows = [
        {"prompt": "Classify: A", "response": "billing"},
        {"input": "Classify: B", "expected_output": "technical"},
        {"text": "Classify: C"},
    ]
    # Pad with synthetic rows so the pydantic validator is happy.
    rows.extend(
        {"prompt": f"Classify synthetic {i}", "response": "other"}
        for i in range(10)
    )
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


@pytest.fixture
def fake_artifacts(tmp_path: Path) -> dict:
    """Simulate what the real pipeline writes to disk + records in the repo."""
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "unsloth/Llama-3.2-1B-Instruct"})
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"fake")

    gguf = tmp_path / "model.q4_k_m.gguf"
    gguf.write_bytes(b"GGUF" + b"\x00" * 32)

    return {
        "adapter_path": str(adapter),
        "gguf_paths": {"q4_k_m": str(gguf)},
    }


# ---------------------------------------------------------------------------
# Dataset normalization
# ---------------------------------------------------------------------------


class TestDatasetNormalization:
    def test_jsonl_file(self, jsonl_dataset: Path) -> None:
        rows = _normalize_dataset(jsonl_dataset)
        assert len(rows) >= 3
        assert rows[0]["text"] == "Classify: A"
        assert rows[0]["response"] == "billing"
        assert rows[1]["text"] == "Classify: B"
        assert rows[1]["response"] == "technical"
        assert rows[2]["text"] == "Classify: C"
        assert "response" not in rows[2]  # no response given

    def test_list_of_dicts(self) -> None:
        rows = _normalize_dataset([
            {"prompt": "P1"},
            {"prompt": "P2", "response": "R2"},
        ])
        assert rows[0] == {"id": "0", "text": "P1", "system": ""}
        assert rows[1] == {"id": "1", "text": "P2", "system": "", "response": "R2"}

    def test_callable(self) -> None:
        def gen():
            yield {"prompt": "X"}
            yield {"prompt": "Y"}
        rows = _normalize_dataset(gen)
        assert [r["text"] for r in rows] == ["X", "Y"]

    def test_json_array_file(self, tmp_path: Path) -> None:
        path = tmp_path / "data.json"
        path.write_text(json.dumps([{"prompt": "A"}, {"prompt": "B"}]))
        rows = _normalize_dataset(path)
        assert [r["text"] for r in rows] == ["A", "B"]

    def test_json_object_file(self, tmp_path: Path) -> None:
        path = tmp_path / "data.json"
        path.write_text(json.dumps({"prompt": "solo"}))
        rows = _normalize_dataset(path)
        assert rows == [{"id": "0", "text": "solo", "system": ""}]

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DistillError, match="not found"):
            _normalize_dataset(tmp_path / "nope.jsonl")

    def test_unknown_format_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "data.csv"
        bad.write_text("a,b,c")
        with pytest.raises(DistillError, match="Unsupported dataset file format"):
            _normalize_dataset(bad)

    def test_unsupported_dataset_type(self) -> None:
        with pytest.raises(DistillError, match="Unsupported dataset type"):
            _normalize_dataset(42)  # type: ignore[arg-type]

    def test_row_missing_prompt(self) -> None:
        with pytest.raises(DistillError, match="no 'prompt'"):
            _normalize_row({"not_a_prompt": "x"}, 3)

    def test_row_not_a_dict(self) -> None:
        with pytest.raises(DistillError, match="not a dict"):
            _normalize_row("bad", 0)  # type: ignore[arg-type]


class TestQuantNormalization:
    def test_string(self) -> None:
        assert _normalize_quant("q4_k_m") == ["q4_k_m"]

    def test_list(self) -> None:
        assert _normalize_quant(["q4_k_m", "q8_0"]) == ["q4_k_m", "q8_0"]

    def test_none_means_skip(self) -> None:
        assert _normalize_quant(None) == []
        assert _normalize_quant("") == []


# ---------------------------------------------------------------------------
# In-memory repo
# ---------------------------------------------------------------------------


class TestInMemoryRepo:
    def _seed(self, repo: _InMemoryRepo) -> None:
        cfg = MagicMock()
        cfg.model_dump.return_value = {"base_model": "x"}
        repo.seed_job("j1", "t1", cfg)

    def test_status_update_updates_job(self) -> None:
        r = _InMemoryRepo(on_progress=None)
        self._seed(r)
        r.update_job_status("t1", "j1", status="running", phase="data_generation")
        job = r.get_job("t1", "j1")
        assert job["status"] == "running"
        assert job["phase"] == "data_generation"

    def test_append_log_stores_and_calls_progress(self) -> None:
        events: list[dict] = []
        r = _InMemoryRepo(on_progress=events.append)
        self._seed(r)
        r.append_log("t1", "j1", "Phase 1/4: Data Generation")
        assert events[-1]["log"] == "Phase 1/4: Data Generation"
        assert r.get_job("t1", "j1")["pipeline_logs"][-1] == "Phase 1/4: Data Generation"

    def test_update_job_patches(self) -> None:
        r = _InMemoryRepo(on_progress=None)
        self._seed(r)
        r.update_job("t1", "j1", {"artifacts": {"adapter_path": "/x"}})
        assert r.get_job("t1", "j1")["artifacts"] == {"adapter_path": "/x"}

    def test_missing_job_is_silent(self) -> None:
        r = _InMemoryRepo(on_progress=None)
        # Must not raise — pipeline's failure path sometimes calls these
        # before seed_job completes in pathological scenarios.
        r.update_job_status("t1", "missing", status="failed")
        r.append_log("t1", "missing", "whoops")
        assert r.get_job("t1", "missing") is None

    def test_noop_methods_return_sensibly(self) -> None:
        r = _InMemoryRepo(on_progress=None)
        r.append_candidates()
        r.record_metrics()
        r.record_training_metric()
        assert r.list_metrics() == []
        assert r.list_candidates() == []


# ---------------------------------------------------------------------------
# Artifact → Student
# ---------------------------------------------------------------------------


class TestStudentFromArtifacts:
    def test_prefers_gguf_when_available(self, fake_artifacts: dict) -> None:
        s = _student_from_artifacts(
            fake_artifacts,
            base_model="unsloth/Llama-3.2-1B-Instruct",
            prefer_gguf=True,
            quant=["q4_k_m"],
        )
        assert s.backend == "gguf"
        assert s.model_path.endswith("model.q4_k_m.gguf")
        assert s.base_model == "unsloth/Llama-3.2-1B-Instruct"

    def test_falls_back_to_adapter_when_no_gguf(self, fake_artifacts: dict) -> None:
        only_adapter = {"adapter_path": fake_artifacts["adapter_path"]}
        s = _student_from_artifacts(
            only_adapter,
            base_model="unsloth/Llama-3.2-1B-Instruct",
            prefer_gguf=True,
            quant=["q4_k_m"],
        )
        assert s.backend == "peft"
        assert s.model_path == str(Path(fake_artifacts["adapter_path"]).resolve())

    def test_returns_peft_when_prefer_gguf_false(self, fake_artifacts: dict) -> None:
        s = _student_from_artifacts(
            fake_artifacts,
            base_model="unsloth/Llama-3.2-1B-Instruct",
            prefer_gguf=False,
            quant=[],
        )
        assert s.backend == "peft"

    def test_raises_when_nothing_usable(self, tmp_path: Path) -> None:
        with pytest.raises(DistillError, match="no usable artifacts"):
            _student_from_artifacts(
                {"adapter_path": str(tmp_path / "missing")},
                base_model="x", prefer_gguf=False, quant=[],
            )


# ---------------------------------------------------------------------------
# distill() end-to-end with mocked pipeline
# ---------------------------------------------------------------------------


class TestDistillEndToEnd:
    def _make_async_pipeline(self, fake_artifacts: dict):
        """Returns an `async def` that mimics a successful pipeline run —
        it captures the tenant/job/config, updates the repo to "completed"
        with the artifacts, and returns."""
        captured: dict = {}

        async def fake_run(job_id, tenant_id, config):
            from opentracy.distillation import pipeline as pipeline_mod

            captured["job_id"] = job_id
            captured["tenant_id"] = tenant_id
            captured["config"] = config

            repo = pipeline_mod.repo
            repo.update_job_status(tenant_id, job_id, status="running", phase="data_generation")
            repo.append_log(tenant_id, job_id, "Phase 1/4: Data Generation")
            repo.update_job_status(tenant_id, job_id, phase="training")
            repo.append_log(tenant_id, job_id, "Phase 3/4: Training")
            repo.update_job(tenant_id, job_id, {"artifacts": fake_artifacts})
            repo.update_job_status(tenant_id, job_id, status="completed", phase="completed")

        return fake_run, captured

    def test_happy_path_returns_gguf_student(
        self, jsonl_dataset: Path, fake_artifacts: dict, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_pipeline, captured = self._make_async_pipeline(fake_artifacts)
        events: list[dict] = []

        monkeypatch.setattr(pipeline_mod, "_run_pipeline", fake_pipeline)
        monkeypatch.setattr(distill_mod, "_ensure_engine", _fake_engine_ctx)

        student = distill(
            jsonl_dataset,
            teacher="openai/gpt-4o",
            student="llama-3.2-1b",
            steps=10,
            output_dir=tmp_path / "out",
            on_progress=events.append,
        )

        assert isinstance(student, Student)
        assert student.backend == "gguf"
        assert student.model_path.endswith("model.q4_k_m.gguf")

        # Config should have been built with our args + inline prompts
        cfg = captured["config"]
        assert cfg.teacher_model == "openai/gpt-4o"
        assert cfg.training_steps == 10
        assert cfg.export_gguf is True
        assert cfg.quantization_types == ["q4_k_m"]
        assert cfg.prompts is not None and len(cfg.prompts) >= 10
        assert cfg.prompts[0]["text"] == "Classify: A"

        # Progress callback saw the transitions
        logs = [e.get("log") for e in events if e.get("log")]
        assert any("Phase 1/4" in l for l in logs)
        assert any("Phase 3/4" in l for l in logs)

    def test_num_prompts_caps_dataset(
        self, jsonl_dataset: Path, fake_artifacts: dict, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_pipeline, captured = self._make_async_pipeline(fake_artifacts)

        monkeypatch.setattr(pipeline_mod, "_run_pipeline", fake_pipeline)
        monkeypatch.setattr(distill_mod, "_ensure_engine", _fake_engine_ctx)

        distill(
            jsonl_dataset,
            student="llama-3.2-1b",
            num_prompts=10,
            output_dir=tmp_path / "out",
        )

        assert len(captured["config"].prompts) == 10

    def test_quantize_none_returns_peft_student(
        self, jsonl_dataset: Path, fake_artifacts: dict, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_pipeline, _ = self._make_async_pipeline(fake_artifacts)

        monkeypatch.setattr(pipeline_mod, "_run_pipeline", fake_pipeline)
        monkeypatch.setattr(distill_mod, "_ensure_engine", _fake_engine_ctx)

        student = distill(
            jsonl_dataset,
            student="llama-3.2-1b",
            quantize=None,
            output_dir=tmp_path / "out",
        )

        assert student.backend == "peft"

    def test_failed_pipeline_raises_distillerror(
        self, jsonl_dataset: Path, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def bad_run(job_id, tenant_id, config):
            pipeline_mod.repo.update_job_status(
                tenant_id, job_id, status="failed",
                error="teacher model returned 401",
            )

        monkeypatch.setattr(pipeline_mod, "_run_pipeline", bad_run)
        monkeypatch.setattr(distill_mod, "_ensure_engine", _fake_engine_ctx)

        with pytest.raises(DistillError, match="teacher model returned 401"):
            distill(
                jsonl_dataset,
                student="llama-3.2-1b",
                output_dir=tmp_path / "out",
            )

    def test_empty_dataset_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "e.jsonl"
        empty.write_text("")
        with pytest.raises(DistillError, match="empty"):
            distill(empty, student="llama-3.2-1b")


# ---------------------------------------------------------------------------
# Engine spawn logic
# ---------------------------------------------------------------------------


class TestEngineEnsure:
    def test_uses_caller_url_without_spawn(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        spawn = MagicMock()
        # Stub GoEngine only if the attribute exists on the module (it's
        # imported lazily inside _ensure_engine, so we patch at import time
        # by pre-populating the module).
        import opentracy.engine as engine_mod
        monkeypatch.setattr(engine_mod, "GoEngine", spawn)

        with distill_mod._ensure_engine("http://override:8080") as url:
            assert url == "http://override:8080"
        spawn.assert_not_called()

    def test_uses_existing_healthy_engine(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OPENTRACY_ENGINE_URL", "http://prebooted:8080")
        monkeypatch.setattr(distill_mod, "_engine_healthy", lambda url, timeout=2.0: True)

        spawn = MagicMock()
        import opentracy.engine as engine_mod
        monkeypatch.setattr(engine_mod, "GoEngine", spawn)

        with distill_mod._ensure_engine(None) as url:
            assert url == "http://prebooted:8080"
        spawn.assert_not_called()


# ---------------------------------------------------------------------------
# Public export surface
# ---------------------------------------------------------------------------


class TestPublicExports:
    def test_distill_top_level(self) -> None:
        assert ot.distill is distill  # type: ignore[attr-defined]
        assert ot.DistillError is DistillError  # type: ignore[attr-defined]
