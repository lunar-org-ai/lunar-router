"""E2B sandbox wrapper — the autonomous engineering loop's compute layer.

A ``SandboxRun`` owns the lifecycle of one E2B sandbox: spawn, upload
workspace tar, run ``claude`` headless with the tenant's BYOK Anthropic
key injected, stream stdout/stderr back, snapshot ``/workspace`` to a
tar, and kill. One sandbox per agent turn — short-lived by design.

Activation
----------

The E2B SDK is an optional dep (``opentracy-new-mode[sandbox]``).
Without ``OPENTRACY_E2B_API_KEY`` set the wrapper raises
:class:`SandboxUnavailable` on first call, so OSS installs that never
take the autonomous path don't need the dep installed.

The custom template (``opentracy-sandbox``) bakes in node + the
``@anthropic-ai/claude-code`` CLI and is built from
``opentracy-infra/e2b/`` — see that repo for the Dockerfile + e2b.toml.
"""

from __future__ import annotations

import logging
import os
import queue
import shlex
import threading
from typing import Any, Iterator, Optional


logger = logging.getLogger("runtime.sandbox.e2b")


DEFAULT_TEMPLATE = "opentracy-sandbox"
DEFAULT_TIMEOUT_S = 300
DEFAULT_WORKSPACE_PATH = "/workspace"

_API_KEY_ENV = "OPENTRACY_E2B_API_KEY"
_TEMPLATE_ENV = "OPENTRACY_E2B_TEMPLATE"

# Marker types streamed back from run_claude() — keep them lightweight
# dicts so the executor can re-emit them in OpenAI-compat chunks
# without an adapter layer.
_EVT_STDOUT = "stdout"
_EVT_STDERR = "stderr"
_EVT_DONE = "done"
_EVT_ERROR = "error"


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


class SandboxUnavailable(RuntimeError):
    """Raised when the sandbox can't be started — missing SDK or API key."""


def is_sandbox_available() -> bool:
    """Cheap precheck — true iff both SDK and API key are present.

    Doesn't actually try to spawn a sandbox; the first ``SandboxRun``
    call surfaces real connection errors.
    """
    if not os.environ.get(_API_KEY_ENV):
        return False
    try:
        import e2b  # noqa: F401
    except Exception:
        return False
    return True


def _require_sdk():
    """Lazy import + uniform error when the dep is missing."""
    try:
        from e2b import Sandbox
    except Exception as exc:
        raise SandboxUnavailable(
            "e2b SDK not installed — pip install 'opentracy-new-mode[sandbox]'"
        ) from exc
    return Sandbox


def _require_api_key() -> str:
    key = os.environ.get(_API_KEY_ENV)
    if not key:
        raise SandboxUnavailable(
            f"{_API_KEY_ENV} not set — sandbox runtime disabled"
        )
    return key


# ---------------------------------------------------------------------------
# SandboxRun
# ---------------------------------------------------------------------------


class SandboxRun:
    """One sandbox lifetime: spawn → upload → exec → snapshot → kill.

    Use as a context manager so cleanup always runs::

        with SandboxRun(anthropic_key=...) as sb:
            sb.upload_workspace_tar(tar_bytes)
            for evt in sb.run_claude(prompt, system=system_prompt):
                yield evt
            new_tar = sb.snapshot_workspace_tar()
    """

    def __init__(
        self,
        *,
        anthropic_key: str,
        template: Optional[str] = None,
        timeout_s: int = DEFAULT_TIMEOUT_S,
        workspace_path: str = DEFAULT_WORKSPACE_PATH,
    ) -> None:
        if not anthropic_key:
            raise ValueError("anthropic_key is required (tenant BYOK)")
        self._anthropic_key = anthropic_key
        self._template = template or os.environ.get(_TEMPLATE_ENV) or DEFAULT_TEMPLATE
        self._timeout_s = timeout_s
        self._workspace_path = workspace_path
        self._sandbox: Any = None

    # -- lifecycle -----------------------------------------------------

    def __enter__(self) -> "SandboxRun":
        Sandbox = _require_sdk()
        api_key = _require_api_key()
        logger.info("sandbox: spawning template=%s timeout=%ds", self._template, self._timeout_s)
        self._sandbox = Sandbox.create(
            template=self._template,
            timeout=self._timeout_s,
            api_key=api_key,
            envs={"ANTHROPIC_API_KEY": self._anthropic_key},
        )
        # Ensure /workspace exists even when the template doesn't seed it.
        try:
            self._sandbox.commands.run(f"mkdir -p {self._workspace_path}")
        except Exception:
            logger.debug("sandbox: workspace mkdir failed (template may already create it)")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._sandbox is None:
            return
        try:
            self._sandbox.kill()
        except Exception:
            logger.warning("sandbox: kill failed", exc_info=True)
        finally:
            self._sandbox = None

    # -- workspace transfer -------------------------------------------

    def upload_workspace_tar(self, tar_bytes: bytes) -> None:
        """Write the workspace tar into the sandbox and extract it.

        Uses a tmp file inside the sandbox so the whole archive doesn't
        live in command argv (avoids ARG_MAX limits for big workspaces).

        ``--no-same-owner --no-same-permissions`` are mandatory: the
        tar was built on our side (root via Cloud Run gcsfuse) but
        extracts in the sandbox as a non-root user. Without these
        flags GNU tar tries to restore the original uid/perms on
        existing dirs (including ``.``) and fails with EPERM.
        """
        self._require_active()
        tar_path = "/tmp/opentracy-in.tar.gz"
        self._sandbox.files.write(tar_path, tar_bytes)
        # ``-m`` (--touch) disables mtime restoration; combined with
        # --no-same-owner / --no-same-permissions, the extract never
        # touches metadata it can't set as a non-root user. The
        # workspace store also avoids emitting a ``.`` root entry, so
        # belt + suspenders here.
        result = self._sandbox.commands.run(
            f"mkdir -p {self._workspace_path} && "
            f"tar xzmf {tar_path} -C {self._workspace_path} "
            f"--no-same-owner --no-same-permissions && "
            f"rm -f {tar_path}",
            timeout=self._timeout_s,
        )
        if getattr(result, "exit_code", 0) != 0:
            stderr = getattr(result, "stderr", "") or ""
            raise RuntimeError(f"sandbox: workspace upload failed: {stderr}")

    def snapshot_workspace_tar(self) -> bytes:
        """Tar ``/workspace`` and pull the bytes out.

        ``files.read`` defaults to text mode (UTF-8), which corrupts a
        gzip stream (replacement chars on invalid sequences). We force
        ``format='bytes'`` so the tarball arrives byte-perfect. Older
        SDK versions don't accept ``format`` — fall back to text and
        re-encode as latin-1 (preserves the byte layout in the cases
        where the SDK happens to round-trip cleanly).
        """
        self._require_active()
        tar_path = "/tmp/opentracy-out.tar.gz"
        result = self._sandbox.commands.run(
            f"tar czf {tar_path} -C {self._workspace_path} .",
            timeout=self._timeout_s,
        )
        if getattr(result, "exit_code", 0) != 0:
            stderr = getattr(result, "stderr", "") or ""
            raise RuntimeError(f"sandbox: workspace snapshot failed: {stderr}")
        try:
            data = self._sandbox.files.read(tar_path, format="bytes")
        except TypeError:
            # Older e2b SDKs don't accept format=; best-effort fallback.
            data = self._sandbox.files.read(tar_path)
        if isinstance(data, str):
            # Last-ditch: try latin-1 round trip; if the SDK already
            # mangled the bytes via utf-8 errors='replace' this will
            # still produce a corrupt tar, but the error surfaces in
            # from_tar_bytes rather than silently here.
            data = data.encode("latin-1", errors="replace")
        return data

    # -- claude execution ---------------------------------------------

    def run_claude(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Iterator[dict]:
        """Run ``claude --print`` headless inside the sandbox.

        Yields a stream of dicts::

            {"type": "stdout", "data": "..."}
            {"type": "stderr", "data": "..."}
            {"type": "done", "exit_code": 0}
            {"type": "error", "detail": "..."}

        The CLI's ``--print`` mode emits the assistant's text to stdout
        and tool/diagnostic output to stderr. Callers normally surface
        stdout to the user and log stderr separately.
        """
        self._require_active()

        argv = self._build_claude_argv(prompt, system=system, model=model)
        # E2B's ``commands.run`` takes a single shell string, not a list.
        # ``shlex.join`` quotes each argv element so prompts containing
        # quotes/backticks/$ don't break out into shell evaluation.
        cmd = shlex.join(argv)
        q: "queue.Queue[dict]" = queue.Queue()

        def _on_stdout(data: Any) -> None:
            q.put({"type": _EVT_STDOUT, "data": _coerce_text(data)})

        def _on_stderr(data: Any) -> None:
            q.put({"type": _EVT_STDERR, "data": _coerce_text(data)})

        def _runner() -> None:
            try:
                # cwd is set via the SDK parameter instead of a CLI flag
                # because claude-code doesn't expose --cwd; the binary
                # implicitly uses the process working directory for
                # file ops.
                result = self._sandbox.commands.run(
                    cmd,
                    cwd=self._workspace_path,
                    on_stdout=_on_stdout,
                    on_stderr=_on_stderr,
                    timeout=self._timeout_s,
                )
                q.put({"type": _EVT_DONE, "exit_code": getattr(result, "exit_code", 0)})
            except Exception as exc:
                logger.warning("sandbox: claude run failed", exc_info=True)
                q.put({"type": _EVT_ERROR, "detail": str(exc)})

        thread = threading.Thread(target=_runner, daemon=True, name="sandbox-claude")
        thread.start()

        while True:
            evt = q.get()
            yield evt
            if evt["type"] in (_EVT_DONE, _EVT_ERROR):
                break

        thread.join(timeout=1.0)

    # -- internals -----------------------------------------------------

    def _require_active(self) -> None:
        if self._sandbox is None:
            raise RuntimeError("SandboxRun used outside of its context manager")

    def _build_claude_argv(
        self,
        prompt: str,
        *,
        system: Optional[str],
        model: Optional[str],
    ) -> list[str]:
        """Compose the ``claude --print`` invocation.

        Stays close to the CLI's headless contract — ``--print`` (single
        turn), ``--output-format text`` (plain stdout), ``--cwd`` pinned
        to /workspace so file ops resolve relative to the agent's tree.
        ``--dangerously-skip-permissions`` is necessary because the
        sandbox doesn't have a tty for the standard approval prompt and
        the whole point is autonomy — the isolation is at the sandbox
        layer, not the CLI's permission gate.
        """
        argv = [
            "claude",
            "--print",
            "--output-format", "text",
            "--dangerously-skip-permissions",
        ]
        if system:
            argv.extend(["--append-system-prompt", system])
        if model:
            argv.extend(["--model", model])
        argv.append(prompt)
        return argv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_text(data: Any) -> str:
    if isinstance(data, str):
        return data
    if isinstance(data, bytes):
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("utf-8", errors="replace")
    return str(data)
