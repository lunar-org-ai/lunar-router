"""
Distillation & Training SDK Client

High-level Python client wrapping the `/v1/distillation` and `/v1/training`
REST endpoints. Mirrors the style of `lunar_router.sdk` (completion/Router).

Quick start:
    >>> from lunar_router import Distiller
    >>> d = Distiller(base_url="http://localhost:8000")
    >>> job = d.create(
    ...     name="gpt-4o → Llama-3.2-1B",
    ...     student_model="llama-3.2-1b",
    ...     teacher_model="openai/gpt-4o",
    ...     num_prompts=200,
    ...     training_steps=100,
    ... )
    >>> job = d.wait(job["id"])          # block until completed
    >>> artifacts = d.artifacts(job["id"])
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, Iterable, List, Optional

import httpx


DEFAULT_BASE_URL = os.environ.get("LUNAR_API_URL", "http://localhost:8000")
DEFAULT_TENANT = "default"


class DistillerError(RuntimeError):
    """Raised when the distillation API returns an error."""


class Distiller:
    """Client for the BOND distillation pipeline (`/v1/distillation`)."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        tenant_id: str = DEFAULT_TENANT,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.tenant_id = tenant_id
        headers = {"Content-Type": "application/json"}
        key = api_key or os.environ.get("LUNAR_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        self._http = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "Distiller":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def _request(self, method: str, path: str, **kwargs) -> Any:
        params = kwargs.pop("params", {}) or {}
        params.setdefault("tenant_id", self.tenant_id)
        r = self._http.request(method, path, params=params, **kwargs)
        if r.status_code >= 400:
            raise DistillerError(f"{method} {path} → {r.status_code}: {r.text}")
        if not r.content:
            return None
        return r.json()

    def student_models(self) -> List[Dict[str, Any]]:
        """List available student models (short-key → HF path)."""
        data = self._request("GET", "/v1/distillation/student-models")
        if isinstance(data, dict):
            return data.get("models") or data.get("students") or []
        return data or []

    def teacher_models(self) -> List[Dict[str, Any]]:
        """List available teacher models."""
        data = self._request("GET", "/v1/distillation/teacher-models")
        if isinstance(data, dict):
            return data.get("models") or data.get("teachers") or []
        return data or []

    def estimate(
        self,
        student_model: str = "llama-3.2-1b",
        num_prompts: int = 1000,
        n_samples: int = 4,
    ) -> Dict[str, Any]:
        """Estimate cost/time for a distillation job before creating it."""
        body = {
            "student_model": student_model,
            "num_prompts": num_prompts,
            "n_samples": n_samples,
        }
        return self._request("POST", "/v1/distillation/estimate", json=body)

    def create(
        self,
        name: str,
        *,
        student_model: str = "llama-3.2-1b",
        teacher_model: str = "openai/gpt-4o",
        num_prompts: int = 1000,
        n_samples: int = 4,
        training_steps: int = 500,
        bond_beta: float = 0.5,
        bond_gamma: float = 0.1,
        temperature: float = 0.8,
        export_gguf: bool = True,
        quantization_types: Optional[List[str]] = None,
        dataset_id: Optional[str] = None,
        description: str = "",
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create & launch a distillation job. Returns the serialized job dict."""
        config: Dict[str, Any] = {
            "student_model": student_model,
            "teacher_model": teacher_model,
            "num_prompts": num_prompts,
            "n_samples": n_samples,
            "training_steps": training_steps,
            "bond_beta": bond_beta,
            "bond_gamma": bond_gamma,
            "temperature": temperature,
            "export_gguf": export_gguf,
            "quantization_types": quantization_types or ["q4_k_m", "q8_0"],
        }
        if dataset_id:
            config["dataset_id"] = dataset_id
        if extra_config:
            config.update(extra_config)

        body = {
            "name": name,
            "description": description,
            "tenant_id": self.tenant_id,
            "config": config,
        }
        return self._request("POST", "/v1/distillation", json=body)

    def list(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        params = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        return self._request("GET", "/v1/distillation", params=params)

    def get(self, job_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/v1/distillation/{job_id}")

    def cancel(self, job_id: str) -> Dict[str, Any]:
        return self._request("POST", f"/v1/distillation/{job_id}/cancel")

    def delete(self, job_id: str) -> Dict[str, Any]:
        return self._request("DELETE", f"/v1/distillation/{job_id}")

    def logs(self, job_id: str) -> Any:
        return self._request("GET", f"/v1/distillation/{job_id}/logs")

    def candidates(self, job_id: str, limit: int = 100) -> Any:
        return self._request(
            "GET", f"/v1/distillation/{job_id}/candidates", params={"limit": limit}
        )

    def artifacts(self, job_id: str) -> Any:
        return self._request("GET", f"/v1/distillation/{job_id}/artifacts")

    def metrics(self, job_id: str, limit: int = 5000) -> Any:
        return self._request(
            "GET", f"/v1/distillation/{job_id}/metrics", params={"limit": limit}
        )

    def wait(
        self,
        job_id: str,
        timeout: float = 3600.0,
        poll_interval: float = 5.0,
        on_update: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """Poll a job until it reaches a terminal state (completed/failed/cancelled)."""
        deadline = time.time() + timeout
        terminal = {"completed", "failed", "cancelled"}
        while True:
            job = self.get(job_id)
            if on_update:
                on_update(job)
            if job.get("status") in terminal:
                return job
            if time.time() > deadline:
                raise TimeoutError(f"Job {job_id} did not finish within {timeout}s")
            time.sleep(poll_interval)

    def stream_progress(
        self, job_id: str, poll_interval: float = 5.0
    ) -> Iterable[Dict[str, Any]]:
        """Yield job state snapshots until the job reaches a terminal state."""
        terminal = {"completed", "failed", "cancelled"}
        while True:
            job = self.get(job_id)
            yield job
            if job.get("status") in terminal:
                return
            time.sleep(poll_interval)


class TrainingClient:
    """Client for the routing-model training endpoints (`/v1/training`)."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        headers = {"Content-Type": "application/json"}
        key = api_key or os.environ.get("LUNAR_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        self._http = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "TrainingClient":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def _request(self, method: str, path: str, **kwargs) -> Any:
        r = self._http.request(method, path, **kwargs)
        if r.status_code >= 400:
            raise DistillerError(f"{method} {path} → {r.status_code}: {r.text}")
        return r.json() if r.content else None

    def status(self) -> Dict[str, Any]:
        return self._request("GET", "/v1/training/status")

    def runs(self, limit: int = 20) -> Any:
        return self._request("GET", "/v1/training/runs", params={"limit": limit})

    def run_now(self) -> Dict[str, Any]:
        """Trigger a manual training run (admin-only)."""
        return self._request("POST", "/v1/training/run_now")

    def pause(self) -> Dict[str, Any]:
        return self._request("POST", "/v1/training/pause")

    def resume(self) -> Dict[str, Any]:
        return self._request("POST", "/v1/training/resume")


__all__ = ["Distiller", "TrainingClient", "DistillerError"]
