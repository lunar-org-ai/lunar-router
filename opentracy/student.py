"""High-level wrapper for trained student models.

A :class:`Student` is a callable that owns everything needed to run inference
on a locally-hosted model — either a LoRA adapter loaded through
Transformers + PEFT, or a quantized GGUF file running through
``llama-cpp-python``. The user doesn't see tokenizers, chat templates, or
base-model plumbing.

Minimum usage::

    import opentracy as ot

    student = ot.load_student("./my-trained-adapter")
    label = student("Classify this ticket: 'Please refund me'")

    # Use like any other OpenTracy model
    resp = ot.completion(model=student, messages=[{"role": "user", "content": "..."}])

The module is safe to import on the base wheel (no torch, peft, llama-cpp-python
required). The heavy imports are deferred until a :class:`Student` is actually
loaded or called.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, Union

__all__ = ["Student", "load_student", "StudentError"]


class StudentError(RuntimeError):
    """Raised for problems specific to Student loading or inference."""


Backend = Literal["peft", "gguf"]


@dataclass
class Student:
    """A callable wrapper around a trained student model.

    Either instantiate directly with known fields or use :func:`load_student`
    to auto-detect the backend from a path or HuggingFace repo ID.

    Attributes:
        backend: ``"peft"`` (Transformers + LoRA adapter) or ``"gguf"``
            (llama.cpp quantized file).
        model_path: Absolute path to the adapter directory or ``.gguf`` file.
        base_model: For ``peft`` backend, the HF ID of the base model the
            adapter was trained against (read from ``adapter_config.json``).
        metadata: Free-form extra metadata — populated when available.
    """

    backend: Backend
    model_path: str
    base_model: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    # Lazily-loaded heavy objects. Not part of the dataclass repr.
    _tokenizer: Any = field(default=None, repr=False, compare=False)
    _model: Any = field(default=None, repr=False, compare=False)
    _n_ctx: int = field(default=2048, repr=False, compare=False)

    # ------------------------------------------------------------------ #
    # Identity helpers
    # ------------------------------------------------------------------ #

    @property
    def id(self) -> str:
        """A human-readable identifier — the last path component."""
        return Path(self.model_path).name or self.backend

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"Student(backend={self.backend!r}, id={self.id!r}, "
            f"base_model={self.base_model!r})"
        )

    # ------------------------------------------------------------------ #
    # Lazy loading
    # ------------------------------------------------------------------ #

    def _ensure_loaded(self) -> None:
        """Load the underlying model on first use. Idempotent."""
        if self._model is not None:
            return
        if self.backend == "peft":
            self._load_peft()
        elif self.backend == "gguf":
            self._load_gguf()
        else:  # pragma: no cover - dataclass Literal guards this
            raise StudentError(f"Unknown backend: {self.backend!r}")

    def _load_peft(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
        except ImportError as e:
            raise StudentError(
                "PEFT backend needs torch + transformers + peft. "
                "Install with: pip install opentracy[distill]"
            ) from e

        # apply_chat_template needs jinja2>=3.1. Old envs (especially stale
        # system Pythons picked up by a `uvicorn` on PATH) can have 3.0.x
        # sitting around and the transformers ImportError that surfaces from
        # inside a generate() call is impossible to diagnose without reading
        # the traceback. Fail loudly at load time instead.
        try:
            import jinja2
            parts = tuple(int(p) for p in jinja2.__version__.split(".")[:2] if p.isdigit())
            if parts and parts < (3, 1):
                raise StudentError(
                    f"jinja2 is too old for transformers.apply_chat_template "
                    f"(found {jinja2.__version__}, need >=3.1). "
                    f"Upgrade in the same interpreter you're running: "
                    f"`{sys.executable} -m pip install -U 'jinja2>=3.1'`."
                )
        except ImportError:
            raise StudentError(
                f"jinja2 is not installed (transformers.apply_chat_template needs it). "
                f"Install with: `{sys.executable} -m pip install 'jinja2>=3.1'`."
            )

        if not self.base_model:
            raise StudentError(
                f"PEFT adapter at {self.model_path} has no base_model recorded. "
                "Set Student(..., base_model='...') explicitly."
            )

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device_map = "auto" if torch.cuda.is_available() else None

        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        # Some tokenizers ship without a pad token — fall back to EOS.
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        base = AutoModelForCausalLM.from_pretrained(
            self.base_model, torch_dtype=dtype, device_map=device_map,
        )
        self._model = PeftModel.from_pretrained(base, self.model_path)
        self._model.eval()

    def _load_gguf(self) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise StudentError(
                "GGUF backend needs llama-cpp-python. "
                "Install with: pip install llama-cpp-python"
            ) from e

        self._model = Llama(
            model_path=self.model_path,
            n_ctx=self._n_ctx,
            verbose=False,
        )

    # ------------------------------------------------------------------ #
    # Public inference surface
    # ------------------------------------------------------------------ #

    def __call__(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        """Run inference on a single prompt string. Returns the text response."""
        messages = [{"role": "user", "content": prompt}]
        resp = self.generate(
            messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )
        return resp["choices"][0]["message"]["content"]

    def batch(
        self,
        prompts: Iterable[str],
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> list[str]:
        """Run inference on many prompts. Sequential; batched kernels TBD."""
        return [
            self(p, max_new_tokens=max_new_tokens, temperature=temperature, **kwargs)
            for p in prompts
        ]

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, list[str]]] = None,
        **kwargs: Any,
    ) -> dict:
        """Full-messages completion. Returns an OpenAI-shaped dict.

        This is what :func:`opentracy.completion` dispatches to when given
        ``model=<Student instance>``.
        """
        self._ensure_loaded()
        start = time.time()

        if self.backend == "peft":
            text, prompt_tokens, completion_tokens = self._generate_peft(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
            )
        else:
            text, prompt_tokens, completion_tokens = self._generate_gguf(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
            )

        latency_ms = (time.time() - start) * 1000
        return {
            "id": f"student-{int(start * 1000)}",
            "object": "chat.completion",
            "created": int(start),
            "model": self.id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            # OpenTracy-specific metadata. completion() re-wraps this into a
            # ModelResponse and surfaces _cost/_latency_ms/_provider on top.
            "_provider": "opentracy",
            "_cost": 0.0,  # local inference — no provider fee
            "_latency_ms": latency_ms,
        }

    # ------------------------------------------------------------------ #
    # Backend-specific generation
    # ------------------------------------------------------------------ #

    def _generate_peft(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: Optional[float],
        stop: Optional[Union[str, list[str]]],
    ) -> tuple[str, int, int]:
        import torch

        inputs = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
        )
        device = getattr(self._model, "device", "cpu")
        inputs = inputs.to(device)

        do_sample = temperature > 0
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            if top_p is not None:
                gen_kwargs["top_p"] = top_p

        with torch.no_grad():
            out = self._model.generate(inputs, **gen_kwargs)

        prompt_tokens = int(inputs.shape[-1])
        completion_tokens = int(out.shape[-1]) - prompt_tokens
        text = self._tokenizer.decode(
            out[0][prompt_tokens:], skip_special_tokens=True,
        ).strip()

        if stop:
            text = _apply_stop(text, stop)
        return text, prompt_tokens, completion_tokens

    def _generate_gguf(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: Optional[float],
        stop: Optional[Union[str, list[str]]],
    ) -> tuple[str, int, int]:
        call_kwargs: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if top_p is not None:
            call_kwargs["top_p"] = top_p
        if stop is not None:
            call_kwargs["stop"] = stop if isinstance(stop, list) else [stop]

        resp = self._model.create_chat_completion(**call_kwargs)
        text = resp["choices"][0]["message"]["content"]
        usage = resp.get("usage", {})
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        return text, prompt_tokens, completion_tokens

    # ------------------------------------------------------------------ #
    # Lifecycle / deployment stubs
    # ------------------------------------------------------------------ #

    def save(self, path: Union[str, Path]) -> Path:
        """Copy the model files to ``path``. Returns the resolved destination.

        For a PEFT student this copies the adapter directory; for a GGUF
        student it copies the single ``.gguf`` file. In both cases
        ``metadata.json`` is refreshed so a later :func:`load_student`
        round-trips faithfully.
        """
        import shutil

        dst = Path(path).resolve()
        src = Path(self.model_path).resolve()
        dst.parent.mkdir(parents=True, exist_ok=True)

        if src.is_file():
            if dst.suffix == "":
                dst = dst / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            meta_dst = dst.parent / f"{dst.stem}.opentracy.json"
        else:
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            meta_dst = dst / "opentracy_meta.json"

        meta_dst.write_text(
            json.dumps(
                {
                    "backend": self.backend,
                    "base_model": self.base_model,
                    "metadata": self.metadata,
                },
                indent=2,
            )
        )
        return dst

    def deploy(self, alias: str, engine_url: Optional[str] = None) -> dict:
        """Register this student behind a routing alias.

        After this call, ``opentracy.completion(model=<alias>, ...)`` will
        dispatch to this student's local inference path — no provider call,
        no HTTP hop.

        The registration is written to the local aliases file (default
        ``~/.opentracy/aliases.json``). If ``engine_url`` is provided, we
        additionally try to POST to the engine's ``/v1/models/register``
        endpoint so server-side callers see the alias too — failures there
        emit a warning but do not raise, since the local dispatch will
        still work.

        Args:
            alias: Logical name. Overwrites any prior alias with the same name.
            engine_url: Optional remote engine to also notify (best-effort).

        Returns:
            The registered alias entry (as stored).
        """
        from . import aliases

        entry = aliases.set_alias(
            alias,
            backend=self.backend,
            model_path=self.model_path,
            base_model=self.base_model,
            metadata=self.metadata,
        )

        if engine_url:
            try:
                _notify_engine_register(engine_url, alias, entry)
            except Exception as e:  # pragma: no cover - best-effort path
                import warnings
                warnings.warn(
                    f"Local alias {alias!r} set, but engine registration at "
                    f"{engine_url} failed: {e}. Local completion calls still work."
                )

        return entry


# ---------------------------------------------------------------------- #
# Factory
# ---------------------------------------------------------------------- #


def load_student(path_or_id: Union[str, Path]) -> Student:
    """Load a trained student from a local path or HuggingFace repo ID.

    Accepted inputs (checked in order):

    * A local ``.gguf`` file → ``backend="gguf"``.
    * A local directory containing an ``adapter_config.json`` → ``backend="peft"``.
      ``base_model`` is read from the config.
    * A local directory containing one or more ``.gguf`` files → ``backend="gguf"``
      (picks the smallest file by size, which is usually the heaviest quantization).
    * Anything else (string that looks like ``org/name``) is treated as a
      HuggingFace repo ID; the repo is downloaded to the HF cache and the
      rules above are applied to the snapshot.

    Raises:
        StudentError: If the path can't be resolved or lacks the files
            required by either backend.
    """
    raw = str(path_or_id)
    path = Path(raw).expanduser()

    if path.exists():
        return _load_from_local(path)

    # Looks like a HuggingFace repo ID — download the snapshot and retry.
    if "/" in raw and not raw.startswith(("/", ".")):
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise StudentError(
                f"{raw!r} doesn't exist locally and huggingface_hub is not "
                "installed. `pip install huggingface_hub` or pass a local path."
            ) from e
        local_snap = Path(snapshot_download(repo_id=raw))
        return _load_from_local(local_snap)

    raise StudentError(f"Path does not exist: {raw}")


def _load_from_local(path: Path) -> Student:
    # Case 1: direct GGUF file
    if path.is_file() and path.suffix == ".gguf":
        return _student_from_gguf(path)

    if not path.is_dir():
        raise StudentError(
            f"{path} is neither a .gguf file nor a directory."
        )

    # Case 2: PEFT adapter directory
    adapter_cfg = path / "adapter_config.json"
    if adapter_cfg.exists():
        return _student_from_peft(path, adapter_cfg)

    # Case 3: directory containing at least one GGUF — pick the smallest
    ggufs = sorted(path.rglob("*.gguf"), key=lambda p: p.stat().st_size)
    if ggufs:
        return _student_from_gguf(ggufs[0])

    raise StudentError(
        f"{path} has no adapter_config.json and no .gguf files. "
        "Either pass a PEFT adapter directory or a GGUF file."
    )


def _student_from_peft(path: Path, cfg_path: Path) -> Student:
    cfg = json.loads(cfg_path.read_text())
    base_model = cfg.get("base_model_name_or_path")
    if not base_model:
        raise StudentError(
            f"adapter_config.json at {cfg_path} is missing "
            "'base_model_name_or_path'. Cannot resolve base model."
        )
    meta_path = path / "opentracy_meta.json"
    metadata = json.loads(meta_path.read_text()).get("metadata", {}) if meta_path.exists() else {}
    return Student(
        backend="peft",
        model_path=str(path.resolve()),
        base_model=base_model,
        metadata=metadata,
    )


def _student_from_gguf(gguf_path: Path) -> Student:
    meta_path = gguf_path.parent / f"{gguf_path.stem}.opentracy.json"
    metadata = json.loads(meta_path.read_text()).get("metadata", {}) if meta_path.exists() else {}
    base_model = None
    if meta_path.exists():
        base_model = json.loads(meta_path.read_text()).get("base_model")
    return Student(
        backend="gguf",
        model_path=str(gguf_path.resolve()),
        base_model=base_model,
        metadata=metadata,
    )


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


def _apply_stop(text: str, stop: Union[str, list[str]]) -> str:
    """Trim ``text`` at the first occurrence of any stop sequence."""
    if isinstance(stop, str):
        stop = [stop]
    cut = len(text)
    for s in stop:
        if not s:
            continue
        idx = text.find(s)
        if idx != -1 and idx < cut:
            cut = idx
    return text[:cut]


def _notify_engine_register(engine_url: str, alias: str, entry: dict) -> None:
    """POST an alias entry to a running engine's /v1/models/register.

    Best-effort — the caller catches and downgrades to a warning. Uses urllib
    so we don't pull in an extra HTTP dep for what is essentially a one-shot
    fire-and-forget.
    """
    import urllib.request

    body = {
        "alias": alias,
        "backend": entry["backend"],
        "model_path": entry["model_path"],
        "base_model": entry.get("base_model"),
        "metadata": entry.get("metadata", {}),
    }
    req = urllib.request.Request(
        f"{engine_url.rstrip('/')}/v1/models/register",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5.0) as r:
        if not (200 <= r.status < 300):
            raise RuntimeError(f"Engine returned HTTP {r.status}")
