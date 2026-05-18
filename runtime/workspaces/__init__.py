"""Per-agent persistent workspace store.

A workspace is the agent's filesystem between turns — code, scratch
files, and the ``.opentracy/`` memory directory the autonomous loop
reads and writes each invocation. In multi-tenant mode the workspace
lives under the tenant's gcsfuse-mounted agents tree, so persistence
is automatic without an extra service.
"""

from runtime.workspaces.store import (
    MANIFEST_DIR,
    MANIFEST_HISTORY_DIR,
    MANIFEST_PENDING,
    MEMORY_DIR,
    MIDDLEWARE_DIR,
    OPENTRACY_DIR,
    PLAN_FILE,
    ROLLBACK_SNAPSHOT,
    SKILLS_DIR,
    STATE_FILE,
    SUBAGENTS_DIR,
    SYSTEM_PROMPT_FILE,
    TOOLS_DIR,
    WorkspaceStore,
    get_workspace,
)

__all__ = [
    "MANIFEST_DIR",
    "MANIFEST_HISTORY_DIR",
    "MANIFEST_PENDING",
    "MEMORY_DIR",
    "MIDDLEWARE_DIR",
    "OPENTRACY_DIR",
    "PLAN_FILE",
    "ROLLBACK_SNAPSHOT",
    "SKILLS_DIR",
    "STATE_FILE",
    "SUBAGENTS_DIR",
    "SYSTEM_PROMPT_FILE",
    "TOOLS_DIR",
    "WorkspaceStore",
    "get_workspace",
]
