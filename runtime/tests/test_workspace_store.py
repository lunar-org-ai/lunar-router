"""Tests for runtime.workspaces.store — NexAU layout + Change Manifest."""

from __future__ import annotations

import io
import json
import tarfile

import pytest

from runtime.workspaces.store import (
    MANIFEST_HISTORY_DIR,
    MANIFEST_PENDING,
    MEMORY_DIR,
    MIDDLEWARE_DIR,
    OPENTRACY_DIR,
    PLAN_FILE,
    SKILLS_DIR,
    STATE_FILE,
    SUBAGENTS_DIR,
    SYSTEM_PROMPT_FILE,
    TOOLS_DIR,
    WorkspaceStore,
    get_workspace,
)


@pytest.fixture
def root(tmp_path):
    """A temporary agents root with one fake agent dir."""
    (tmp_path / "demo").mkdir()
    return tmp_path


# ---------------------------------------------------------------------------
# NexAU layout (AHE §3.1 — Component Observability)
# ---------------------------------------------------------------------------


def test_ensure_creates_nexau_layout_with_seeds(root):
    ws = WorkspaceStore("demo", root=root)
    ws.ensure()

    # Mandatory seeds: system_prompt + memory/{plan,state}.
    assert (ws.path / SYSTEM_PROMPT_FILE).is_file()
    assert (ws.path / PLAN_FILE).is_file()
    assert (ws.path / STATE_FILE).is_file()

    state = json.loads((ws.path / STATE_FILE).read_text(encoding="utf-8"))
    assert state["next_step"] is None
    assert state["facts"] == []
    assert state["blockers"] == []


def test_ensure_creates_empty_component_dirs_per_minimal_seed(root):
    """AHE invariant §3.2: tools/middleware/skills/subagents start empty.

    Only the *directories* exist so the evolution agent has discoverable
    mount points; no seed content is written so we don't bias toward
    our own editorial choices.
    """
    ws = WorkspaceStore("demo", root=root)
    ws.ensure()

    assert (ws.path / TOOLS_DIR).is_dir()
    assert list((ws.path / TOOLS_DIR).iterdir()) == []
    assert (ws.path / MIDDLEWARE_DIR).is_dir()
    assert list((ws.path / MIDDLEWARE_DIR).iterdir()) == []
    assert (ws.path / SKILLS_DIR).is_dir()
    assert list((ws.path / SKILLS_DIR).iterdir()) == []
    assert (ws.path / SUBAGENTS_DIR).is_dir()
    assert list((ws.path / SUBAGENTS_DIR).iterdir()) == []
    assert (ws.path / MANIFEST_HISTORY_DIR).is_dir()


def test_ensure_does_not_overwrite_existing_plan(root):
    ws = WorkspaceStore("demo", root=root)
    ws.ensure()
    plan_path = ws.path / PLAN_FILE
    plan_path.write_text("# Custom plan\n\nstep one done\n", encoding="utf-8")

    ws.ensure()  # second call must be idempotent
    assert "step one done" in plan_path.read_text(encoding="utf-8")


def test_ensure_does_not_overwrite_existing_system_prompt(root):
    ws = WorkspaceStore("demo", root=root)
    ws.ensure()
    sp = ws.path / SYSTEM_PROMPT_FILE
    sp.write_text("You are evolved.\n", encoding="utf-8")

    ws.ensure()
    assert sp.read_text(encoding="utf-8") == "You are evolved.\n"


def test_read_system_prompt_returns_default_when_missing(root):
    ws = WorkspaceStore("demo", root=root)
    # No ensure call.
    text = ws.read_system_prompt()
    assert "autonomous engineer" in text.lower()
    assert ".opentracy/memory/plan.md" in text


def test_read_plan_returns_default_when_missing(root):
    ws = WorkspaceStore("demo", root=root)
    assert "No plan yet" in ws.read_plan()


def test_read_state_recovers_from_corrupt_json(root):
    ws = WorkspaceStore("demo", root=root)
    ws.ensure()
    (ws.path / STATE_FILE).write_text("{not json", encoding="utf-8")

    state = ws.read_state()
    assert state["next_step"] is None


def test_list_files_returns_relative_paths_sorted(root):
    ws = WorkspaceStore("demo", root=root)
    ws.ensure()
    (ws.path / "src").mkdir()
    (ws.path / "src" / "main.py").write_text("print('hi')\n", encoding="utf-8")
    (ws.path / "README.md").write_text("# demo\n", encoding="utf-8")

    files = ws.list_files()
    assert "README.md" in files
    assert "src/main.py" in files
    assert PLAN_FILE in files
    assert STATE_FILE in files
    assert SYSTEM_PROMPT_FILE in files


def test_list_nexau_components_shows_zero_seed_baseline(root):
    ws = WorkspaceStore("demo", root=root)
    ws.ensure()

    snapshot = ws.list_nexau_components()
    assert snapshot["system_prompt"] == ["system_prompt.md"]
    assert snapshot["tools"] == []
    assert snapshot["middleware"] == []
    assert snapshot["skills"] == []
    assert snapshot["subagents"] == []
    # memory always has the two seeds.
    assert sorted(snapshot["memory"]) == ["plan.md", "state.json"]


def test_list_nexau_components_picks_up_added_files(root):
    ws = WorkspaceStore("demo", root=root)
    ws.ensure()
    (ws.path / TOOLS_DIR / "rg.json").write_text("{}", encoding="utf-8")
    (ws.path / TOOLS_DIR / "rg.sh").write_text("#!/bin/sh\nrg \"$@\"\n", encoding="utf-8")
    (ws.path / SKILLS_DIR / "plan_first.md").write_text("Always plan first.\n", encoding="utf-8")

    snapshot = ws.list_nexau_components()
    assert sorted(snapshot["tools"]) == ["rg.json", "rg.sh"]
    assert snapshot["skills"] == ["plan_first.md"]


# ---------------------------------------------------------------------------
# Change Manifest (AHE §3.3 — Decision Observability)
# ---------------------------------------------------------------------------


def test_pending_manifest_roundtrip(root):
    ws = WorkspaceStore("demo", root=root)
    ws.ensure()
    ws.write_pending_manifest({
        "changed_files": [".opentracy/skills/plan_first.md"],
        "claimed_fixes": ["agent skipped planning on task #42"],
        "at_risk_regressions": ["could slow down simple replies"],
    })

    data = ws.read_pending_manifest()
    assert data is not None
    assert data["changed_files"] == [".opentracy/skills/plan_first.md"]
    assert "created_at" in data


def test_pending_manifest_none_when_unset(root):
    ws = WorkspaceStore("demo", root=root)
    ws.ensure()
    assert ws.read_pending_manifest() is None


def test_roll_pending_to_history_archives_and_clears(root):
    ws = WorkspaceStore("demo", root=root)
    ws.ensure()
    ws.write_pending_manifest({"changed_files": ["x"], "claimed_fixes": ["y"]})

    archive = ws.roll_pending_to_history(
        outcome={"verdict": "confirmed", "evidence": "trace-001"}
    )
    assert archive is not None
    assert archive.exists()
    assert (ws.path / MANIFEST_PENDING).exists() is False

    history = ws.list_manifest_history()
    assert len(history) == 1
    assert history[0]["outcome"]["verdict"] == "confirmed"
    assert history[0]["outcome"]["evidence"] == "trace-001"
    assert "verified_at" in history[0]["outcome"]


def test_roll_pending_to_history_no_op_when_nothing_pending(root):
    ws = WorkspaceStore("demo", root=root)
    ws.ensure()
    assert ws.roll_pending_to_history(outcome={"verdict": "confirmed"}) is None


def test_list_manifest_history_newest_first(root):
    ws = WorkspaceStore("demo", root=root)
    ws.ensure()
    import time

    ws.write_pending_manifest({"claimed_fixes": ["first"]})
    ws.roll_pending_to_history(outcome={"verdict": "confirmed"})
    time.sleep(1.1)  # iso filename has second resolution
    ws.write_pending_manifest({"claimed_fixes": ["second"]})
    ws.roll_pending_to_history(outcome={"verdict": "regressed"})

    history = ws.list_manifest_history()
    assert history[0]["claimed_fixes"] == ["second"]
    assert history[1]["claimed_fixes"] == ["first"]


# ---------------------------------------------------------------------------
# Tar round trip — workspace transfer to/from E2B sandbox
# ---------------------------------------------------------------------------


def test_tar_roundtrip_preserves_files(root):
    ws = WorkspaceStore("demo", root=root)
    ws.ensure()
    (ws.path / "hello.txt").write_text("world\n", encoding="utf-8")
    (ws.path / TOOLS_DIR / "rg.sh").write_text("#!/bin/sh\n", encoding="utf-8")

    data = ws.to_tar_bytes()
    assert len(data) > 0

    import shutil
    shutil.rmtree(ws.path)
    ws.from_tar_bytes(data)

    assert (ws.path / "hello.txt").read_text(encoding="utf-8") == "world\n"
    assert (ws.path / TOOLS_DIR / "rg.sh").is_file()
    assert (ws.path / PLAN_FILE).is_file()
    assert (ws.path / SYSTEM_PROMPT_FILE).is_file()


def test_from_tar_drops_files_not_in_archive(root):
    ws = WorkspaceStore("demo", root=root)
    ws.ensure()
    (ws.path / "keep.txt").write_text("ok", encoding="utf-8")

    data = ws.to_tar_bytes()

    (ws.path / "gone.txt").write_text("bye", encoding="utf-8")

    ws.from_tar_bytes(data)
    assert (ws.path / "keep.txt").exists()
    assert not (ws.path / "gone.txt").exists()


def test_from_tar_reseeds_nexau_layout_if_missing(root):
    ws = WorkspaceStore("demo", root=root)
    ws.ensure()

    # Hand-build a tar without .opentracy/ — simulates a sandbox that
    # nuked the harness layer entirely.
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo("hello.txt")
        payload = b"world"
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))

    ws.from_tar_bytes(buf.getvalue())
    assert (ws.path / "hello.txt").exists()
    # All seeded mount points back in place.
    assert (ws.path / SYSTEM_PROMPT_FILE).is_file()
    assert (ws.path / PLAN_FILE).is_file()
    assert (ws.path / STATE_FILE).is_file()
    assert (ws.path / TOOLS_DIR).is_dir()
    assert (ws.path / MIDDLEWARE_DIR).is_dir()
    assert (ws.path / SKILLS_DIR).is_dir()
    assert (ws.path / SUBAGENTS_DIR).is_dir()


def test_from_tar_rejects_path_traversal(root):
    ws = WorkspaceStore("demo", root=root)
    ws.ensure()

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo("../escape.txt")
        payload = b"nope"
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
        info2 = tarfile.TarInfo("safe.txt")
        payload2 = b"ok"
        info2.size = len(payload2)
        tar.addfile(info2, io.BytesIO(payload2))

    ws.from_tar_bytes(buf.getvalue())
    assert not (ws.path.parent / "escape.txt").exists()
    assert (ws.path / "safe.txt").exists()


def test_get_workspace_factory_ensures(root):
    ws = get_workspace("demo", root=root)
    assert ws.path.is_dir()
    assert (ws.path / SYSTEM_PROMPT_FILE).is_file()
    assert (ws.path / PLAN_FILE).is_file()


def test_opentracy_dir_constant_matches_layout(root):
    """Sanity: every well-known path is under .opentracy/ — keeps the
    NexAU mount root invariant explicit so future refactors notice it."""
    ws = WorkspaceStore("demo", root=root)
    ws.ensure()
    for relpath in (
        SYSTEM_PROMPT_FILE,
        PLAN_FILE,
        STATE_FILE,
        TOOLS_DIR,
        MIDDLEWARE_DIR,
        SKILLS_DIR,
        SUBAGENTS_DIR,
        MEMORY_DIR,
        MANIFEST_PENDING,
        MANIFEST_HISTORY_DIR,
    ):
        assert relpath.startswith(OPENTRACY_DIR + "/")
