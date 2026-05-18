"""Smoke test for the AHE evolution orchestrator.

Wires the loop end-to-end with mocked components:
  - per-agent executor → stub that returns canned ExecutionRecords
  - SandboxRun → stub that simulates the Evolve Agent writing a
    pending manifest into the workspace

Then asserts the IterationResult shape + verifies the pending manifest
landed and the per-agent cache was invalidated.
"""

from __future__ import annotations

import io
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from runtime.evolution import run_one_iteration
from runtime.evolution.types import Evidence, RolloutResult, TaskOutcome


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


@dataclass
class _StubRecord:
    response: str
    duration_ms: float = 12.0
    success: bool = True
    error: Optional[str] = None


class _StubExecutor:
    """Returns the same canned response for any task. Lets the rollout
    assert tasks were replayed without depending on the real pipeline."""

    def __init__(self, response: str = "stub-response", succeed: bool = True):
        self.calls: list[str] = []
        self._response = response
        self._succeed = succeed

    def run(self, request, history=None, session_id=None):
        self.calls.append(request)
        return None, _StubRecord(
            response=self._response,
            success=self._succeed,
            error=None if self._succeed else "stub-error",
        )


def _make_fake_sandbox_factory(pending_manifest: dict[str, Any] | None):
    """Return a fake SandboxRun class that simulates the Evolve Agent.

    Preserves the uploaded workspace tar in the snapshot back so
    history dirs etc. aren't wiped — mimics what a real sandbox
    does (it untars on entry, runs claude, re-tars EVERYTHING on
    exit). On top of the uploaded contents the fake injects a fresh
    pending manifest + a new skill file to represent the Evolve
    Agent's edits."""

    class _Sandbox:
        def __init__(self, *, anthropic_key, template=None, timeout_s=300):
            self.anthropic_key = anthropic_key
            self._uploaded_tar: bytes = b""

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def upload_workspace_tar(self, data: bytes) -> None:
            self._uploaded_tar = data

        def run_claude(self, _prompt, *, system=None, model=None):
            yield {"type": "stdout", "data": "Evolved! Edited skills/plan_first.md."}
            yield {"type": "done", "exit_code": 0}

        def snapshot_workspace_tar(self) -> bytes:
            # Start from the uploaded tar so all existing files survive
            # (history archives, memory, prompts, etc.). Then overlay the
            # Evolve Agent's "edits" on top.
            buf_in = io.BytesIO(self._uploaded_tar) if self._uploaded_tar else None
            buf_out = io.BytesIO()
            seen: set[str] = set()
            with tarfile.open(fileobj=buf_out, mode="w:gz") as tar_out:
                if buf_in is not None:
                    with tarfile.open(fileobj=buf_in, mode="r:gz") as tar_in:
                        for member in tar_in.getmembers():
                            if member.name in (
                                ".opentracy/manifest/pending.json",
                                ".opentracy/skills/plan_first.md",
                            ):
                                # We'll re-add updated versions below.
                                continue
                            extracted = tar_in.extractfile(member)
                            tar_out.addfile(
                                member,
                                io.BytesIO(extracted.read()) if extracted else None,
                            )
                            seen.add(member.name)
                if pending_manifest is not None:
                    payload = json.dumps(pending_manifest, indent=2).encode("utf-8")
                    info = tarfile.TarInfo(".opentracy/manifest/pending.json")
                    info.size = len(payload)
                    tar_out.addfile(info, io.BytesIO(payload))
                skill_payload = b"# Plan first\nAlways plan first.\n"
                info = tarfile.TarInfo(".opentracy/skills/plan_first.md")
                info.size = len(skill_payload)
                tar_out.addfile(info, io.BytesIO(skill_payload))
            return buf_out.getvalue()

    return _Sandbox


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_workspace(tmp_path, monkeypatch):
    """Per-test workspace store rooted in tmp_path; agent_id = 'demo'."""
    from runtime.workspaces import store as ws_store

    monkeypatch.setattr(
        ws_store, "_agents_root", lambda root=None: root if root else tmp_path,
    )
    (tmp_path / "demo").mkdir(exist_ok=True)
    yield ws_store.get_workspace("demo", root=tmp_path)


@pytest.fixture
def stub_executor():
    return _StubExecutor()


@pytest.fixture(autouse=True)
def _stubs(monkeypatch, stub_executor):
    # BYOK resolver → fake key so the loop doesn't fall over.
    monkeypatch.setattr(
        "runtime.agents.secrets.get_secret",
        lambda provider, agent_id=None: "sk-ant-fake",
    )
    # Multi-tenant gate ON so the per-agent executor path is taken.
    monkeypatch.setattr(
        "runtime.tenants.feature.is_multi_tenant_enabled",
        lambda: True,
    )
    # Pin the active tenant.
    from runtime import tenant_context as tctx
    tctx.set_active("acme")

    # Resolver returns the stub executor, no matter what dir is on disk.
    from runtime.executor import per_agent as pae
    monkeypatch.setattr(
        pae, "get_executor_for_agent",
        lambda tenant, agent, *, fallback_executor=None, agents_root=None: stub_executor,
    )
    # v1: cluster_failures makes a real Anthropic call by default —
    # short-circuit to identity so tests don't hit the network.
    from runtime.evolution import loop as _loop
    monkeypatch.setattr(
        _loop, "cluster_failures",
        lambda evidence, *, anthropic_key, model=None: evidence,
    )
    yield
    tctx.set_active(None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_iteration_replays_tasks_and_returns_rollout(isolated_workspace, stub_executor):
    factory = _make_fake_sandbox_factory(pending_manifest={
        "changed_files": [".opentracy/skills/plan_first.md"],
        "claimed_fixes": ["agent skips planning"],
        "rationale": "fail rate suggests no upfront plan",
    })
    result = run_one_iteration(
        agent_id="demo",
        tasks=["how do I reset my password?", "where's my order?"],
        sandbox_factory=factory,
        k=1,
    )

    assert stub_executor.calls == [
        "how do I reset my password?",
        "where's my order?",
    ]
    assert result.rollout.passed == 2
    assert result.rollout.failed == 0
    assert all(o.response == "stub-response" for o in result.rollout.outcomes)


def test_iteration_writes_evidence_summary(isolated_workspace):
    factory = _make_fake_sandbox_factory(pending_manifest=None)
    result = run_one_iteration(
        agent_id="demo",
        tasks=["t1"],
        sandbox_factory=factory,
        k=1,
    )
    assert "Rollout:" in result.evidence.summary
    assert "PASS" in result.evidence.summary


def test_iteration_evolve_writes_pending_manifest(isolated_workspace):
    manifest = {
        "changed_files": [".opentracy/skills/plan_first.md"],
        "claimed_fixes": ["plan more"],
        "at_risk_regressions": ["slower on cold tasks"],
        "rationale": "evidence X",
    }
    factory = _make_fake_sandbox_factory(pending_manifest=manifest)
    result = run_one_iteration(
        agent_id="demo",
        tasks=["t1"],
        sandbox_factory=factory,
        k=1,
    )
    assert result.evolve.pending_manifest is not None
    assert result.evolve.pending_manifest["claimed_fixes"] == ["plan more"]
    # files_edited picks up the new file the sandbox dropped.
    assert ".opentracy/skills/plan_first.md" in result.evolve.files_edited


def test_iteration_with_no_prior_pending_records_no_signal(isolated_workspace):
    factory = _make_fake_sandbox_factory(pending_manifest=None)
    result = run_one_iteration(
        agent_id="demo",
        tasks=["t1"],
        sandbox_factory=factory,
        k=1,
    )
    assert result.verification.verdict == "no_signal"
    assert result.verification.pending_archived_to is None


def test_iteration_verifies_prior_pending_and_archives(isolated_workspace):
    # Seed a prior pending manifest BEFORE running the loop.
    isolated_workspace.write_pending_manifest({
        "claimed_fixes": ["agent A"],
        "at_risk_regressions": ["maybe B"],
    })
    factory = _make_fake_sandbox_factory(pending_manifest={
        "claimed_fixes": ["new fix"],
        "rationale": "follow-up",
    })
    result = run_one_iteration(
        agent_id="demo",
        tasks=["t1", "t2"],
        sandbox_factory=factory,
        k=1,
    )
    # All rollouts passed → confirmed verdict on prior pending.
    assert result.verification.verdict == "confirmed"
    assert result.verification.pending_archived_to is not None
    history = isolated_workspace.list_manifest_history()
    assert len(history) == 1
    assert history[0]["outcome"]["verdict"] == "confirmed"
    # The newly-written pending manifest replaces the old one.
    new_pending = isolated_workspace.read_pending_manifest()
    assert new_pending["claimed_fixes"] == ["new fix"]


def test_iteration_verdict_regressed_when_failures_with_at_risk(isolated_workspace):
    isolated_workspace.write_pending_manifest({
        "claimed_fixes": [],
        "at_risk_regressions": ["might fail Y"],
    })
    factory = _make_fake_sandbox_factory(pending_manifest=None)
    # Use a failing executor.
    from runtime.executor import per_agent as pae
    failing = _StubExecutor(succeed=False)
    pae.get_executor_for_agent = lambda *a, **k: failing  # type: ignore[assignment]

    result = run_one_iteration(
        agent_id="demo", tasks=["t1"], sandbox_factory=factory, k=1,
    )
    assert result.rollout.failed == 1
    assert result.verification.verdict == "regressed"


def test_iteration_verdict_mixed_when_failures_without_predictions(isolated_workspace):
    isolated_workspace.write_pending_manifest({
        "claimed_fixes": ["something else"],
        "at_risk_regressions": [],
    })
    factory = _make_fake_sandbox_factory(pending_manifest=None)
    from runtime.executor import per_agent as pae
    failing = _StubExecutor(succeed=False)
    pae.get_executor_for_agent = lambda *a, **k: failing  # type: ignore[assignment]

    result = run_one_iteration(
        agent_id="demo", tasks=["t1"], sandbox_factory=factory, k=1,
    )
    assert result.verification.verdict == "mixed"


def test_iteration_raises_without_anthropic_key(isolated_workspace, monkeypatch):
    monkeypatch.setattr(
        "runtime.agents.secrets.get_secret",
        lambda provider, agent_id=None: None,
    )
    factory = _make_fake_sandbox_factory(pending_manifest=None)
    with pytest.raises(RuntimeError, match="no Anthropic key"):
        run_one_iteration(
            agent_id="demo", tasks=["t1"], sandbox_factory=factory,
        )


def test_iteration_invalidates_per_agent_cache(isolated_workspace, monkeypatch):
    invalidated: list = []
    from runtime.executor import per_agent as pae
    monkeypatch.setattr(
        pae, "invalidate",
        lambda tenant, agent: invalidated.append((tenant, agent)) or True,
    )
    factory = _make_fake_sandbox_factory(pending_manifest=None)
    run_one_iteration(
        agent_id="demo", tasks=["t1"], sandbox_factory=factory, k=1,
    )
    assert invalidated == [("acme", "demo")]


# ---------------------------------------------------------------------------
# v1 — k>=2, clustering, file-level rollback
# ---------------------------------------------------------------------------


def test_iteration_with_k_2_replays_every_task_twice(isolated_workspace, stub_executor):
    """k=2 means each task is run twice. Outcomes count goes up but
    per-task aggregates (passed/failed) stay anchored to the unique
    task list — majority-pass per task."""
    factory = _make_fake_sandbox_factory(pending_manifest=None)
    run_one_iteration(
        agent_id="demo",
        tasks=["t1", "t2"],
        sandbox_factory=factory,
        k=2,
    )
    # 2 tasks × 2 replays = 4 executor.run calls.
    assert stub_executor.calls == ["t1", "t2", "t1", "t2"]


def test_iteration_with_k_2_majority_pass_per_task(isolated_workspace, monkeypatch):
    """Mix of pass+fail per task. With k=3, 2/3 majority counts as
    PASS, 1/3 majority counts as FAIL — flaky flag set when split."""
    factory = _make_fake_sandbox_factory(pending_manifest=None)

    # Per-task result schedule: t1 always passes, t2 fails on run 0 only.
    state = {"run": -1}
    def _run(task, history=None, session_id=None):
        state["run"] += 1
        idx_in_round = state["run"] % 2  # alternates t1/t2 within a round
        round_num = state["run"] // 2
        if task == "t2" and round_num == 0:
            return None, _StubRecord(response="oops", success=False, error="boom")
        return None, _StubRecord(response="ok", success=True)

    from runtime.executor import per_agent as pae
    monkeypatch.setattr(
        pae, "get_executor_for_agent",
        lambda *a, **kw: type("E", (), {"run": staticmethod(_run)})(),
    )

    result = run_one_iteration(
        agent_id="demo",
        tasks=["t1", "t2"],
        sandbox_factory=factory,
        k=3,
    )
    aggs = result.rollout.task_aggregates
    assert aggs["t1"]["passed_runs"] == 3
    assert aggs["t1"]["majority_pass"] is True
    assert aggs["t2"]["passed_runs"] == 2  # 2 of 3 passed (failed only run 0)
    assert aggs["t2"]["majority_pass"] is True
    assert aggs["t2"]["flaky"] is True
    assert result.rollout.flaky_tasks == ["t2"]


def test_iteration_calls_cluster_failures_with_evidence(isolated_workspace, monkeypatch):
    """Distill phase composes summarize + cluster. Verify the stub is
    invoked with the post-summarize Evidence + the BYOK key."""
    calls = []
    factory = _make_fake_sandbox_factory(pending_manifest=None)

    from runtime.evolution import loop as _loop

    def _spy(evidence, *, anthropic_key, model=None):
        calls.append({"evidence": evidence, "anthropic_key": anthropic_key})
        return evidence

    monkeypatch.setattr(_loop, "cluster_failures", _spy)

    run_one_iteration(
        agent_id="demo", tasks=["t1"], sandbox_factory=factory, k=1,
    )
    assert len(calls) == 1
    assert calls[0]["anthropic_key"] == "sk-ant-fake"
    assert "Rollout:" in calls[0]["evidence"].summary


def test_iteration_passes_clusters_to_evolve_agent(isolated_workspace, monkeypatch):
    """When cluster_failures returns clusters, they show up in the
    evidence_summary that run_evolve is invoked with."""
    from runtime.evolution import loop as _loop
    from runtime.evolution.types import EvidenceCluster

    def _add_clusters(evidence, *, anthropic_key, model=None):
        evidence.clusters = [
            EvidenceCluster(
                root_cause="agent-skipped-planning",
                tasks=["t1"],
                severity=4,
                notes="model jumped to action without scoping",
            ),
        ]
        return evidence
    monkeypatch.setattr(_loop, "cluster_failures", _add_clusters)

    captured = {}
    real_run_evolve = _loop.run_evolve
    def _spy_evolve(**kwargs):
        captured["evidence_summary"] = kwargs.get("evidence_summary")
        return real_run_evolve(**kwargs)
    monkeypatch.setattr(_loop, "run_evolve", _spy_evolve)

    factory = _make_fake_sandbox_factory(pending_manifest=None)
    run_one_iteration(
        agent_id="demo", tasks=["t1"], sandbox_factory=factory, k=1,
    )
    summary = captured["evidence_summary"]
    assert "Agent Debugger clusters" in summary
    assert "agent-skipped-planning" in summary
    assert "[severity 4]" in summary


def test_rollback_snapshot_persists_post_evolve(isolated_workspace):
    """When the Evolve Agent declares changed_files, the loop writes
    a rollback snapshot capturing the pre-edit content of those
    files. Files that existed pre-edit are captured as content;
    new files are captured as None (rollback = unlink)."""
    # Seed system_prompt.md with known content BEFORE evolve runs.
    sp = isolated_workspace.path / ".opentracy" / "system_prompt.md"
    sp.write_text("ORIGINAL PROMPT", encoding="utf-8")

    factory = _make_fake_sandbox_factory(pending_manifest={
        "changed_files": [
            ".opentracy/system_prompt.md",      # edited (existed before)
            ".opentracy/skills/plan_first.md",  # newly created
        ],
        "claimed_fixes": ["agent will plan now"],
        "at_risk_regressions": [],
    })
    run_one_iteration(
        agent_id="demo", tasks=["t1"], sandbox_factory=factory, k=1,
    )
    snapshot = isolated_workspace.read_rollback_snapshot()
    assert snapshot is not None
    assert snapshot["files"][".opentracy/system_prompt.md"] == "ORIGINAL PROMPT"
    # The skill file is NEW post-edit → snapshot value None means
    # rollback = unlink.
    assert snapshot["files"][".opentracy/skills/plan_first.md"] is None


def test_rollback_applied_on_regressed_verdict(isolated_workspace):
    """Two-iteration scenario:
      iter 1: evolve edits system_prompt.md → rollback snapshot saved.
      iter 2: rollout fails AND prior pending had at_risk_regressions
              → verdict=regressed → rollback restores original prompt.
    """
    isolated_workspace.path.joinpath(".opentracy", "system_prompt.md").write_text(
        "PROMPT BEFORE", encoding="utf-8",
    )

    # iter 1: edit + predict regression risk.
    factory_iter1 = _make_fake_sandbox_factory(pending_manifest={
        "changed_files": [".opentracy/system_prompt.md"],
        "claimed_fixes": ["x"],
        "at_risk_regressions": ["y might break"],
    })
    run_one_iteration(
        agent_id="demo", tasks=["t1"], sandbox_factory=factory_iter1, k=1,
    )
    # After iter 1, the fake sandbox replaced the prompt content
    # via the workspace tar roundtrip... but our fake sandbox doesn't
    # actually mutate the prompt — it only injects skills/plan_first.md
    # and the pending manifest. So the rollback snapshot captures the
    # current (still "PROMPT BEFORE") content as if it had been edited.
    snapshot = isolated_workspace.read_rollback_snapshot()
    assert snapshot["files"][".opentracy/system_prompt.md"] == "PROMPT BEFORE"

    # Manually flip the prompt to simulate an "edit" that broke things.
    isolated_workspace.path.joinpath(".opentracy", "system_prompt.md").write_text(
        "PROMPT AFTER (broken)", encoding="utf-8",
    )

    # iter 2: rollout fails → regressed verdict → rollback should restore.
    from runtime.executor import per_agent as pae
    failing = _StubExecutor(succeed=False)
    pae.get_executor_for_agent = lambda *a, **k: failing  # type: ignore[assignment]

    factory_iter2 = _make_fake_sandbox_factory(pending_manifest=None)
    result2 = run_one_iteration(
        agent_id="demo", tasks=["t1"], sandbox_factory=factory_iter2, k=1,
    )
    assert result2.verification.verdict == "regressed"
    assert ".opentracy/system_prompt.md" in result2.verification.delta["rollback_applied"]
    # Prompt content reverted.
    assert isolated_workspace.path.joinpath(".opentracy", "system_prompt.md").read_text() == "PROMPT BEFORE"
    # Snapshot was consumed.
    assert isolated_workspace.read_rollback_snapshot() is None


def test_confirmed_verdict_clears_stale_rollback_snapshot(isolated_workspace):
    """When the rollout passes (verdict=confirmed) the snapshot is
    obsolete — the edits are now the new baseline. Snapshot dropped."""
    # Seed pending + rollback snapshot from a "prior iteration".
    isolated_workspace.write_pending_manifest({
        "claimed_fixes": ["fixed"],
        "at_risk_regressions": [],
    })
    isolated_workspace.write_rollback_snapshot(
        iteration_id="prior",
        files={".opentracy/system_prompt.md": "STALE BACKUP"},
    )
    factory = _make_fake_sandbox_factory(pending_manifest=None)
    result = run_one_iteration(
        agent_id="demo", tasks=["t1"], sandbox_factory=factory, k=1,
    )
    assert result.verification.verdict == "confirmed"
    # The OLD rollback snapshot must be gone (confirmed → no rollback).
    # A NEW one is written only if the new pending manifest declares
    # changed_files; here pending_manifest=None so no new snapshot.
    assert isolated_workspace.read_rollback_snapshot() is None


def test_iteration_result_has_expected_dict_shape(isolated_workspace):
    factory = _make_fake_sandbox_factory(pending_manifest=None)
    result = run_one_iteration(
        agent_id="demo", tasks=["t1"], sandbox_factory=factory, k=1,
    )
    data = result.to_dict()
    assert set(data.keys()) >= {
        "iteration_id", "agent_id", "tenant_id",
        "verification", "rollout", "evidence", "evolve",
    }
    assert data["iteration_id"].startswith("evo-")
    assert data["agent_id"] == "demo"
    assert data["tenant_id"] == "acme"
