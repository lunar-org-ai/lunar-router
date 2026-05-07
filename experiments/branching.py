"""Create candidate directories by branching agent/ and applying mutations."""

from __future__ import annotations

import json
import secrets
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

from experiments.types import CandidateManifest, Mutation
from runtime.compiler.loader import load_agent

CANDIDATES_DIR = Path("experiments/candidates")
BASELINE_AGENT_DIR = Path("agent")


def _new_candidate_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    rand = secrets.token_hex(2)
    return f"cand_{ts}_{rand}"


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _set_dotted(d: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur: Any = d
    for p in parts[:-1]:
        if not isinstance(cur, dict) or p not in cur:
            raise KeyError(f"path {path!r}: intermediate key {p!r} missing")
        cur = cur[p]
    if not isinstance(cur, dict):
        raise KeyError(f"path {path!r}: target is not a dict")
    if parts[-1] not in cur:
        raise KeyError(f"path {path!r}: leaf key {parts[-1]!r} missing")
    cur[parts[-1]] = value


def _apply_mutation(candidate_dir: Path, m: Mutation) -> None:
    target = candidate_dir / "agent" / m.file
    if not target.exists():
        raise FileNotFoundError(f"mutation target {target} does not exist in candidate")
    with target.open() as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, dict):
        raise ValueError(f"{target} root is not a mapping; cannot apply path {m.path}")
    _set_dotted(doc, m.path, m.value)
    with target.open("w") as f:
        yaml.safe_dump(doc, f, sort_keys=False)


def create_candidate(
    mutations: list[Mutation],
    description: Optional[str] = None,
    baseline_dir: Path | str = BASELINE_AGENT_DIR,
    candidates_dir: Path | str = CANDIDATES_DIR,
) -> CandidateManifest:
    """Branch agent/ into a new candidate dir and apply mutations.

    Returns the candidate manifest. The candidate's agent.yaml is fully
    runnable on its own; point the runner at
    experiments/candidates/<id>/agent/agent.yaml.
    """
    if not mutations:
        raise ValueError("at least one mutation required")

    baseline_dir = Path(baseline_dir)
    candidates_dir = Path(candidates_dir)
    candidates_dir.mkdir(parents=True, exist_ok=True)

    cid = _new_candidate_id()
    cand_root = candidates_dir / cid
    cand_root.mkdir(parents=True, exist_ok=False)

    # Copy the entire agent/ tree (yaml + prompts + custom)
    shutil.copytree(baseline_dir, cand_root / "agent")

    # Read baseline version (before any mutation applies to the candidate copy)
    baseline_cfg = load_agent(baseline_dir / "agent.yaml")

    # Apply each mutation in order
    for m in mutations:
        _apply_mutation(cand_root, m)

    manifest = CandidateManifest(
        id=cid,
        parent_version=baseline_cfg.version,
        parent_path=str((baseline_dir / "agent.yaml").resolve()),
        created_at=_now_iso(),
        description=description,
        mutations=list(mutations),
    )
    with (cand_root / "manifest.json").open("w") as f:
        json.dump(asdict(manifest), f, indent=2)

    return manifest


def candidate_agent_path(candidate_id: str, candidates_dir: Path | str = CANDIDATES_DIR) -> Path:
    """Path to a candidate's agent.yaml — feed this to run_suite."""
    return Path(candidates_dir) / candidate_id / "agent" / "agent.yaml"


def list_candidates(candidates_dir: Path | str = CANDIDATES_DIR) -> list[CandidateManifest]:
    """Return all candidate manifests, oldest first."""
    root = Path(candidates_dir)
    if not root.exists():
        return []
    out: list[CandidateManifest] = []
    for d in sorted(root.iterdir()):
        manifest_path = d / "manifest.json"
        if not manifest_path.exists():
            continue
        with manifest_path.open() as f:
            data = json.load(f)
        out.append(
            CandidateManifest(
                id=data["id"],
                parent_version=data["parent_version"],
                parent_path=data["parent_path"],
                created_at=data["created_at"],
                description=data.get("description"),
                mutations=[Mutation(**m) for m in data["mutations"]],
            )
        )
    return out
