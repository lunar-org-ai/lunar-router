"""Sandbox runtime for the autonomous engineering loop.

The autonomous agent operates inside an E2B sandbox per turn: workspace
files are uploaded, ``claude`` is invoked, output streams back, and the
mutated workspace is snapshotted before the sandbox is killed.
"""

from runtime.sandbox.e2b import (
    DEFAULT_TEMPLATE,
    SandboxRun,
    SandboxUnavailable,
    is_sandbox_available,
)

__all__ = [
    "DEFAULT_TEMPLATE",
    "SandboxRun",
    "SandboxUnavailable",
    "is_sandbox_available",
]
