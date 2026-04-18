"""
Go Engine Client: Thin Python SDK for the Lunar Router Go backend.

Starts the Go binary as a subprocess and wraps HTTP calls in a Pythonic API.
Users get the same interface but with Go-level performance.

Usage:
    from lunar_router.engine import GoEngine

    engine = GoEngine()
    engine.start()

    decision = engine.route("Explain quantum computing")
    print(decision["selected_model"])

    engine.stop()

Or as a context manager:

    with GoEngine() as engine:
        decision = engine.route("What is 2+2?")
"""

import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class GoEngine:
    """
    Manages the Go inference engine binary.

    Starts lunar-engine as a subprocess, communicates via localhost HTTP.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        port: Optional[int] = None,
        host: str = "127.0.0.1",
        binary_path: Optional[str] = None,
        config_path: Optional[str] = None,
        no_embedder: bool = False,
    ):
        """
        Initialize the Go engine client.

        Args:
            weights_path: Path to weights directory. Auto-detected if None.
            port: Server port. Auto-assigned if None.
            host: Server host (default: 127.0.0.1).
            binary_path: Path to lunar-engine binary. Auto-detected if None.
            config_path: Path to YAML config file.
            no_embedder: Disable ONNX embedder (embedding-only mode).
        """
        self._port = port or _find_free_port()
        self._host = host
        self._base_url = f"http://{host}:{self._port}"
        self._process: Optional[subprocess.Popen] = None
        self._binary_path = binary_path
        self._weights_path = weights_path
        self._config_path = config_path
        self._no_embedder = no_embedder

    @property
    def base_url(self) -> str:
        """Return the base URL of the running engine."""
        return self._base_url

    @property
    def is_running(self) -> bool:
        """Check if the engine process is running."""
        return self._process is not None and self._process.poll() is None

    def start(self, timeout: float = 15.0) -> None:
        """
        Start the Go engine binary as a subprocess.

        Args:
            timeout: Max seconds to wait for the engine to become healthy.

        Raises:
            FileNotFoundError: If the binary is not found.
            RuntimeError: If the engine fails to start.
        """
        if self.is_running:
            return

        binary = self._binary_path or _find_binary()
        weights = self._weights_path or _find_weights()

        # Make sure the ONNX embedder has what it needs: a shared library path
        # (for onnxruntime_go to dlopen) and model.onnx + vocab.txt inside the
        # weights directory. Both are shipped in the wheel alongside the binary.
        _ensure_onnx_assets(weights)
        env = os.environ.copy()
        bundled_lib = _find_bundled_onnxruntime()
        if bundled_lib and "ONNXRUNTIME_LIB_PATH" not in env:
            env["ONNXRUNTIME_LIB_PATH"] = str(bundled_lib)

        args = [
            binary,
            "--host", self._host,
            "--port", str(self._port),
            "--weights", str(weights),
        ]
        if self._config_path:
            args.extend(["--config", self._config_path])
        if self._no_embedder:
            args.append("--no-embedder")

        self._process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        # Wait for health check
        if not self._wait_healthy(timeout):
            self.stop()
            stderr = ""
            if self._process and self._process.stderr:
                stderr = self._process.stderr.read().decode(errors="replace")
            raise RuntimeError(
                f"Engine failed to start within {timeout}s.\n"
                f"Binary: {binary}\n"
                f"Weights: {weights}\n"
                f"Stderr: {stderr}"
            )

    def stop(self) -> None:
        """Gracefully stop the Go engine."""
        if self._process is None:
            return

        try:
            self._process.send_signal(signal.SIGTERM)
            self._process.wait(timeout=5)
        except (subprocess.TimeoutExpired, OSError):
            self._process.kill()
            self._process.wait(timeout=2)
        finally:
            self._process = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    # --- API Methods ---

    def route(
        self,
        prompt: str,
        available_models: Optional[List[str]] = None,
        cost_weight: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Route a prompt to the best model.

        Args:
            prompt: The text prompt to route.
            available_models: Restrict to these model IDs.
            cost_weight: Override cost weight (0=quality, 1=cost).

        Returns:
            Dict with selected_model, expected_error, cluster_id, all_scores, usage.
        """
        body: Dict[str, Any] = {"prompt": prompt}
        if available_models is not None:
            body["available_models"] = available_models
        if cost_weight is not None:
            body["cost_weight"] = cost_weight
        return self._post("/v1/route", body)

    def route_embedding(
        self,
        embedding: List[float],
        available_models: Optional[List[str]] = None,
        cost_weight: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Route a pre-computed embedding to the best model.

        Args:
            embedding: Pre-computed embedding vector (384 dims).
            available_models: Restrict to these model IDs.
            cost_weight: Override cost weight.

        Returns:
            Routing decision dict.
        """
        body: Dict[str, Any] = {"embedding": embedding}
        if available_models is not None:
            body["available_models"] = available_models
        if cost_weight is not None:
            body["cost_weight"] = cost_weight
        return self._post("/v1/route", body)

    def route_batch(
        self,
        prompts: List[str],
        available_models: Optional[List[str]] = None,
        cost_weight: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Route multiple prompts."""
        return [
            self.route(prompt, available_models, cost_weight)
            for prompt in prompts
        ]

    def models(self) -> Dict[str, Any]:
        """List all available models with profiles."""
        return self._get("/v1/models")

    def model(self, model_id: str) -> Dict[str, Any]:
        """Get info for a specific model."""
        return self._get(f"/v1/models/{model_id}")

    def health(self) -> Dict[str, Any]:
        """Check engine health."""
        return self._get("/health")

    def metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics summary."""
        return self._get("/v1/metrics")

    def metrics_recent(self, n: int = 20) -> List[Dict[str, Any]]:
        """Get last N raw request metrics."""
        return self._get(f"/v1/metrics/recent?n={n}")

    def metrics_reset(self) -> None:
        """Reset all metrics."""
        self._post("/v1/metrics/reset", {})

    def cache_stats(self) -> Dict[str, Any]:
        """Get routing cache statistics (size, hits, misses, hit_rate)."""
        return self._get("/v1/cache")

    def cache_clear(self) -> None:
        """Clear the routing cache."""
        self._post("/v1/cache/clear", {})

    # --- Chat Completions (Gateway) ---

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str = "auto",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Send a chat completion request via the gateway.

        Supports multimodal messages with images (base64 or URL):

            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
                ]
            }]

        Args:
            messages: List of message dicts. Content can be a string or an
                array of content parts (text + image_url) for vision models.
            model: Model name, or "auto" for semantic routing.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            stream: If True, returns a generator of SSE chunks.
            **kwargs: Additional OpenAI-compatible params.

        Returns:
            OpenAI-compatible chat completion response dict.
            If stream=True, returns a generator instead.

        Example:
            >>> response = engine.chat(
            ...     messages=[{"role": "user", "content": "Hello!"}],
            ...     model="gpt-4o-mini",
            ... )
            >>> print(response["choices"][0]["message"]["content"])
        """
        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature
        body.update(kwargs)

        if stream:
            return self._stream_chat(body)

        return self._post("/v1/chat/completions", body)

    def _stream_chat(self, body: dict):
        """Stream chat completion, yielding parsed SSE chunks."""
        import urllib.request
        url = self._base_url + "/v1/chat/completions"
        data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")

        resp = urllib.request.urlopen(req, timeout=120)
        try:
            for line in resp:
                line = line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        yield json.loads(data_str, strict=False)
                    except json.JSONDecodeError:
                        continue
        finally:
            resp.close()

    def generate(
        self,
        prompt: str,
        model: str = "auto",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> str:
        """
        Generate a text response. Convenience wrapper around chat().

        Args:
            prompt: The user prompt.
            model: Model name, or "auto" for semantic routing.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            The generated text string.
        """
        resp = self.chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        return resp["choices"][0]["message"]["content"]

    def vision(
        self,
        image: str,
        prompt: str = "Describe this image.",
        model: str = "auto",
        max_tokens: Optional[int] = None,
        detail: str = "auto",
    ) -> Dict[str, Any]:
        """
        Send an image to a vision model and get a response.

        Args:
            image: Base64-encoded image string, data URI, or URL.
            prompt: Text prompt to accompany the image.
            model: Model name, or "auto" for semantic routing.
            max_tokens: Maximum tokens to generate.
            detail: Image detail level ("low", "high", "auto").

        Returns:
            OpenAI-compatible chat completion response dict.

        Example:
            >>> import base64
            >>> with open("photo.jpg", "rb") as f:
            ...     img_b64 = base64.b64encode(f.read()).decode()
            >>> resp = engine.vision(img_b64, "What animal is this?", model="gpt-4o")
            >>> print(resp["choices"][0]["message"]["content"])
        """
        # Normalize image to data URI if raw base64
        if not image.startswith(("data:", "http://", "https://")):
            image = f"data:image/jpeg;base64,{image}"

        messages: List[Dict[str, Any]] = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image, "detail": detail}},
            ],
        }]
        return self.chat(messages=messages, model=model, max_tokens=max_tokens)

    def smart_generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        cost_weight: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Route to best model and generate in one call.

        Returns:
            Dict with 'text', 'model', 'routing'.
        """
        decision = self.route(prompt, cost_weight=cost_weight)
        resp = self.chat(
            messages=[{"role": "user", "content": prompt}],
            model=decision["selected_model"],
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        return {
            "text": resp["choices"][0]["message"]["content"],
            "model": decision["selected_model"],
            "routing": decision,
            "usage": resp.get("usage"),
        }

    # --- Internal ---

    def _get(self, path: str) -> Any:
        import urllib.request
        url = self._base_url + path
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode(), strict=False)

    def _post(self, path: str, body: dict) -> Any:
        import urllib.request
        url = self._base_url + path
        data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode(), strict=False)

    def _wait_healthy(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._process and self._process.poll() is not None:
                return False  # Process exited
            try:
                resp = self._get("/health")
                if resp.get("status") == "healthy":
                    return True
            except Exception:
                pass
            time.sleep(0.1)
        return False


def _find_free_port() -> int:
    """Find a free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _find_binary() -> str:
    """
    Find the lunar-engine binary, checking:
    1. go/bin/lunar-engine in the project (dev mode)
    2. Bundled in the package (_bin/)
    3. PATH
    """
    # Dev mode: built binary in go/bin/
    project_root = Path(__file__).parent.parent
    dev_binary = project_root / "go" / "bin" / "lunar-engine"
    if dev_binary.exists():
        return str(dev_binary)

    # Bundled in package
    pkg_binary = Path(__file__).parent / "_bin" / f"lunar-engine-{_platform_tag()}"
    if pkg_binary.exists():
        return str(pkg_binary)

    # On PATH
    path_binary = shutil.which("lunar-engine")
    if path_binary:
        return path_binary

    raise FileNotFoundError(
        f"lunar-engine binary not bundled for this platform ({_platform_tag()}).\n"
        f"Supported platforms get the binary automatically via `pip install lunar-router`.\n"
        f"If you hit this, options:\n"
        f"  1. File an issue — https://github.com/lunar-org-ai/lunar-router/issues\n"
        f"  2. Build from source: cd go && make build  (requires Go toolchain)\n"
        f"  3. Use the Python backend instead: load_router(engine='python')"
    )


def _find_bundled_onnxruntime() -> Optional[Path]:
    """Return the path to the bundled libonnxruntime shared library, if any.

    The Go engine uses onnxruntime_go which dlopens onnxruntime at runtime.
    Ship the library inside ``lunar_router/_bin/`` so users don't need to
    install it system-wide.
    """
    bin_dir = Path(__file__).parent / "_bin"
    # Preferred names per platform (Linux: .so, macOS: .dylib, Windows: .dll)
    candidates = [
        bin_dir / "libonnxruntime.so",
        bin_dir / "libonnxruntime.so.1",
        bin_dir / "libonnxruntime.dylib",
        bin_dir / "onnxruntime.dll",
    ]
    # Also match any version-suffixed .so (e.g. libonnxruntime.so.1.19.2)
    if bin_dir.exists():
        for p in bin_dir.iterdir():
            name = p.name
            if name.startswith("libonnxruntime.so") or name.startswith("libonnxruntime.") or name == "onnxruntime.dll":
                candidates.append(p)
    for c in candidates:
        if c.exists():
            return c
    return None


def _ensure_onnx_assets(weights_path: str) -> None:
    """Ensure the weights directory has the ONNX model + vocab the engine needs.

    The Go engine expects ``<weights>/onnx/model.onnx`` and
    ``<weights>/onnx/vocab.txt``. When the wheel ships them under
    ``lunar_router/_onnx/``, copy them on first start so the user never has to
    manage ONNX assets manually.
    """
    target = Path(weights_path) / "onnx"
    model_dst = target / "model.onnx"
    vocab_dst = target / "vocab.txt"
    if model_dst.exists() and vocab_dst.exists():
        return

    src = Path(__file__).parent / "_onnx"
    model_src = src / "model.onnx"
    vocab_src = src / "vocab.txt"
    if not (model_src.exists() and vocab_src.exists()):
        return  # nothing to wire up; engine will surface its own error

    target.mkdir(parents=True, exist_ok=True)
    if not model_dst.exists():
        shutil.copy2(model_src, model_dst)
    if not vocab_dst.exists():
        shutil.copy2(vocab_src, vocab_dst)


def _find_weights() -> str:
    """Find the weights directory."""
    # Check common locations
    candidates = []

    # macOS
    app_support = Path.home() / "Library" / "Application Support" / "lunar_router"
    if app_support.exists():
        for name in ["weights-mmlu-v1", "weights-default"]:
            p = app_support / name
            if p.exists() and (p / "clusters").exists():
                candidates.append(p)

    # Linux
    local_share = Path.home() / ".local" / "share" / "lunar_router"
    if local_share.exists():
        for name in ["weights-mmlu-v1", "weights-default"]:
            p = local_share / name
            if p.exists() and (p / "clusters").exists():
                candidates.append(p)

    # Environment variable
    env_path = os.environ.get("LUNAR_WEIGHTS_PATH")
    if env_path and Path(env_path).exists():
        return env_path

    if candidates:
        return str(candidates[0])

    raise FileNotFoundError(
        "Weights not found. Download with:\n"
        "  lunar-router download weights-mmlu-v1"
    )


def _platform_tag() -> str:
    """Return platform tag for binary name (e.g., 'darwin-arm64')."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        machine = "amd64"
    elif machine in ("aarch64", "arm64"):
        machine = "arm64"
    return f"{system}-{machine}"
