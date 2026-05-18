"""Tests for ledger.writer.write_decision."""

from __future__ import annotations

import json
from pathlib import Path

from ledger.writer import write_decision


def test_write_decision_creates_file(tmp_path: Path):
    path = write_decision(
        "router_wakeup",
        {"action": "skipped", "rationale": "drift low"},
        decisions_dir=tmp_path,
    )
    assert path.exists()
    body = json.loads(path.read_text())
    assert body["kind"] == "router_wakeup"
    assert "timestamp" in body
    assert body["payload"]["action"] == "skipped"


def test_write_decision_filename_includes_iso(tmp_path: Path):
    path = write_decision(
        "router_wakeup", {"action": "blocked"}, decisions_dir=tmp_path
    )
    assert path.name.startswith("router_wakeup_")
    assert path.name.endswith(".json")
    # ISO compact has format YYYYMMDDTHHMMSSZ.
    middle = path.name[len("router_wakeup_") : -len(".json")]
    assert len(middle) == 16
    assert middle.endswith("Z")


def test_write_decision_creates_directory(tmp_path: Path):
    target = tmp_path / "nested" / "decisions"
    assert not target.exists()
    write_decision("router_wakeup", {"x": 1}, decisions_dir=target)
    assert target.exists()
