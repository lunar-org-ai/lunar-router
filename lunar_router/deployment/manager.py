"""vLLM deployment manager: launches, monitors, and stops vLLM subprocesses."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from typing import Optional

import httpx

from . import storage

logger = logging.getLogger(__name__)

# Track running vLLM processes
_processes: dict[str, asyncio.subprocess.Process] = {}

# Port allocation
_BASE_PORT = 8090
_allocated_ports: set[int] = set()


def _allocate_port() -> int:
    """Find the next available port starting from _BASE_PORT."""
    port = _BASE_PORT
    while port in _allocated_ports:
        port += 1
    _allocated_ports.add(port)
    return port


def _release_port(port: int) -> None:
    _allocated_ports.discard(port)


async def deploy_model(
    model_id: str,
    model_path: str,
    config: dict | None = None,
) -> dict:
    """Deploy a model using vLLM. Returns deployment info for the API response.

    This is the main entry point called by distillation/routes.py deploy endpoint
    and by the deployment routes directly.
    """
    config = config or {}

    # Check if already deployed
    existing = storage.list_deployments()
    for dep in existing:
        if dep.get("model_id") == model_id and dep.get("status") in ("in_service", "starting", "creating"):
            return {
                "deployment_id": dep["id"],
                "model_id": model_id,
                "status": dep["status"],
                "endpoint_url": dep.get("endpoint_url", ""),
                "already_deployed": True,
            }

    import uuid
    deployment_id = f"dep-{uuid.uuid4().hex[:12]}"
    port = _allocate_port()

    storage.insert_deployment(
        deployment_id=deployment_id,
        model_id=model_id,
        model_path=model_path,
        port=port,
        instance_type=config.get("instance_type", "local-gpu"),
        config=config,
    )

    # Launch vLLM in background
    asyncio.create_task(_launch_and_monitor(deployment_id, model_path, port, config))

    return {
        "deployment_id": deployment_id,
        "model_id": model_id,
        "status": "creating",
        "endpoint_url": "",
        "already_deployed": False,
    }


async def _launch_and_monitor(
    deployment_id: str,
    model_path: str,
    port: int,
    config: dict,
) -> None:
    """Launch vLLM subprocess, wait for health, update status."""
    try:
        storage.update_deployment(deployment_id, status="starting")

        # Build vLLM command
        vllm_args = _build_vllm_args(model_path, port, config)
        logger.info(f"Launching vLLM for {deployment_id}: {' '.join(vllm_args)}")

        proc = await asyncio.create_subprocess_exec(
            *vllm_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _processes[deployment_id] = proc

        storage.update_deployment(deployment_id, pid=proc.pid)
        logger.info(f"vLLM started (PID {proc.pid}) on port {port}")

        # Wait for health check
        healthy = await _wait_for_health(port, timeout=300)

        if healthy:
            endpoint_url = f"http://127.0.0.1:{port}"
            storage.update_deployment(
                deployment_id,
                status="in_service",
                endpoint_url=endpoint_url,
            )
            logger.info(f"Deployment {deployment_id} is in_service at {endpoint_url}")

            # Register in Go engine
            await _register_in_engine(deployment_id, model_path, endpoint_url)

            # Monitor the process — update status if it dies
            await proc.wait()
            storage.update_deployment(
                deployment_id,
                status="stopped",
                error_message="vLLM process exited",
            )
        else:
            # Health check timed out
            stderr = ""
            if proc.returncode is None:
                proc.kill()
                _, stderr_bytes = await proc.communicate()
                stderr = stderr_bytes.decode(errors="replace")[-2000:]
            else:
                if proc.stderr:
                    stderr = (await proc.stderr.read()).decode(errors="replace")[-2000:]

            storage.update_deployment(
                deployment_id,
                status="failed",
                error_message=f"user_message=vLLM failed to start within timeout, error_code=startup_crash, details={stderr[-500:]}",
                error_code="startup_crash",
            )
            logger.error(f"Deployment {deployment_id} failed to start: {stderr[-500:]}")

    except Exception as e:
        logger.exception(f"Deployment {deployment_id} error: {e}")
        storage.update_deployment(
            deployment_id,
            status="failed",
            error_message=f"user_message={e}, error_code=unknown",
            error_code="unknown",
        )
    finally:
        _processes.pop(deployment_id, None)
        _release_port(port)


def _build_vllm_args(model_path: str, port: int, config: dict) -> list[str]:
    """Build the vLLM command-line arguments."""
    args = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--host", "127.0.0.1",
    ]

    # Parse vllm_args string from config if provided
    vllm_args_str = config.get("vllm_args", "")
    if vllm_args_str:
        args.extend(vllm_args_str.split())
    else:
        # Sensible defaults
        args.extend([
            "--max-model-len", str(config.get("max_model_len", 4096)),
            "--dtype", config.get("dtype", "auto"),
            "--gpu-memory-utilization", str(config.get("gpu_memory_utilization", 0.9)),
        ])

    return args


async def _wait_for_health(port: int, timeout: int = 300) -> bool:
    """Poll vLLM health endpoint until ready or timeout."""
    url = f"http://127.0.0.1:{port}/health"
    deadline = asyncio.get_event_loop().time() + timeout

    async with httpx.AsyncClient(timeout=5.0) as http:
        while asyncio.get_event_loop().time() < deadline:
            try:
                resp = await http.get(url)
                if resp.status_code == 200:
                    return True
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout):
                pass
            await asyncio.sleep(3)

    return False


async def _register_in_engine(deployment_id: str, model_path: str, endpoint_url: str) -> None:
    """Register the deployed model in the Go engine so it can be routed to."""
    engine_url = os.environ.get("LUNAR_ENGINE_URL", "http://localhost:8080")
    try:
        async with httpx.AsyncClient(timeout=10.0) as http:
            await http.post(
                f"{engine_url}/v1/models",
                json={
                    "model_id": deployment_id,
                    "provider": "vllm",
                    "endpoint": endpoint_url,
                },
            )
            logger.info(f"Registered {deployment_id} in Go engine")
    except Exception as e:
        logger.warning(f"Could not register deployment in engine: {e}")


async def stop_deployment(deployment_id: str) -> None:
    """Stop a running vLLM deployment."""
    proc = _processes.pop(deployment_id, None)
    if proc and proc.returncode is None:
        try:
            proc.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(proc.wait(), timeout=15)
            except asyncio.TimeoutError:
                proc.kill()
        except ProcessLookupError:
            pass

    dep = storage.get_deployment(deployment_id)
    if dep and dep.get("port"):
        _release_port(dep["port"])


async def pause_deployment(deployment_id: str) -> None:
    """Pause (SIGSTOP) a running vLLM deployment."""
    proc = _processes.get(deployment_id)
    if proc and proc.returncode is None:
        try:
            proc.send_signal(signal.SIGSTOP)
            storage.update_deployment(deployment_id, status="paused")
        except (ProcessLookupError, OSError) as e:
            logger.warning(f"Could not pause {deployment_id}: {e}")


async def resume_deployment(deployment_id: str) -> None:
    """Resume (SIGCONT) a paused vLLM deployment."""
    proc = _processes.get(deployment_id)
    if proc and proc.returncode is None:
        try:
            proc.send_signal(signal.SIGCONT)
            storage.update_deployment(deployment_id, status="in_service")
        except (ProcessLookupError, OSError) as e:
            logger.warning(f"Could not resume {deployment_id}: {e}")


async def cleanup_stale_deployments() -> None:
    """Check for deployments with PIDs that are no longer alive. Run on startup."""
    deployments = storage.list_deployments(
        statuses=["in_service", "starting", "creating", "paused"]
    )
    for dep in deployments:
        pid = dep.get("pid", 0)
        if pid > 0:
            try:
                os.kill(pid, 0)  # Check if alive
            except (OSError, ProcessLookupError):
                logger.info(f"Stale deployment {dep['id']} (PID {pid}) — marking stopped")
                storage.update_deployment(
                    dep["id"],
                    status="stopped",
                    error_message="Server restarted — process no longer running",
                )
