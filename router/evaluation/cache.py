"""Response cache for offline evaluation.

Caches model responses keyed by ``(prompt_hash, model_id)`` so that
once all models have been run on a benchmark dataset, evaluation is
free and instant — no LLM calls needed to re-evaluate at different
lambda values.

Storage format: JSONL file, one entry per line. Default location:
``evals/_response_cache/cache.jsonl`` (matches the project's eval
report convention).

P15.3.6 ports the surface but **drops** ``populate_from_dataset`` —
the explicit one-shot replay lives in ``tools/populate_response_cache.py``
(deferred). The cache here is read-only at eval time.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import hashlib
import json
import logging


logger = logging.getLogger("router.evaluation.cache")


DEFAULT_CACHE_PATH = Path("evals") / "_response_cache" / "cache.jsonl"


@dataclass
class CachedResponse:
    """A single cached model response."""

    prompt_hash: str
    model_id: str
    response_text: str
    loss: float  # 0.0 = correct, 1.0 = error
    latency_ms: Optional[float] = None
    tokens_used: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "prompt_hash": self.prompt_hash,
            "model_id": self.model_id,
            "response_text": self.response_text,
            "loss": self.loss,
        }
        if self.latency_ms is not None:
            d["latency_ms"] = self.latency_ms
        if self.tokens_used is not None:
            d["tokens_used"] = self.tokens_used
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "CachedResponse":
        return cls(
            prompt_hash=data["prompt_hash"],
            model_id=data["model_id"],
            response_text=data["response_text"],
            loss=data["loss"],
            latency_ms=data.get("latency_ms"),
            tokens_used=data.get("tokens_used"),
            metadata=data.get("metadata", {}),
        )


class ResponseCache:
    """Cache of model responses for offline evaluation.

    Stores ``(prompt_hash, model_id) -> CachedResponse`` mappings in a
    JSONL file. Once cached, the evaluator can sweep λ values without
    any LLM calls.
    """

    def __init__(self, path: Optional[str | Path] = None):
        self._entries: dict[tuple[str, str], CachedResponse] = {}
        self._path = Path(path) if path else None
        if self._path and self._path.exists():
            self._load()

    @staticmethod
    def hash_prompt(prompt: str) -> str:
        """Deterministic hash for a prompt string."""
        return hashlib.md5(prompt.encode("utf-8")).hexdigest()

    def add(
        self,
        prompt: str,
        model_id: str,
        response_text: str,
        loss: float,
        latency_ms: Optional[float] = None,
        tokens_used: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Add a response to the cache."""
        prompt_hash = self.hash_prompt(prompt)
        entry = CachedResponse(
            prompt_hash=prompt_hash,
            model_id=model_id,
            response_text=response_text,
            loss=loss,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            metadata=metadata or {},
        )
        self._entries[(prompt_hash, model_id)] = entry

    def get(self, prompt: str, model_id: str) -> Optional[CachedResponse]:
        prompt_hash = self.hash_prompt(prompt)
        return self._entries.get((prompt_hash, model_id))

    def get_by_hash(self, prompt_hash: str, model_id: str) -> Optional[CachedResponse]:
        return self._entries.get((prompt_hash, model_id))

    def get_all_models(self, prompt: str) -> dict[str, CachedResponse]:
        prompt_hash = self.hash_prompt(prompt)
        return self.get_all_models_by_hash(prompt_hash)

    def get_all_models_by_hash(self, prompt_hash: str) -> dict[str, CachedResponse]:
        result = {}
        for (ph, mid), entry in self._entries.items():
            if ph == prompt_hash:
                result[mid] = entry
        return result

    def has(self, prompt: str, model_id: str) -> bool:
        return (self.hash_prompt(prompt), model_id) in self._entries

    @property
    def model_ids(self) -> set[str]:
        return {mid for _, mid in self._entries}

    @property
    def prompt_hashes(self) -> set[str]:
        return {ph for ph, _ in self._entries}

    def __len__(self) -> int:
        return len(self._entries)

    def coverage(self, model_id: str) -> int:
        """Number of prompts with a cached response for this model."""
        return sum(1 for _, mid in self._entries if mid == model_id)

    def purge_model(self, model_id: str) -> int:
        """Drop every entry for ``model_id`` (e.g., after a model upgrade)."""
        keys = [k for k in self._entries if k[1] == model_id]
        for k in keys:
            del self._entries[k]
        return len(keys)

    def save(self, path: Optional[str | Path] = None) -> None:
        """Save cache to JSONL file."""
        save_path = Path(path) if path else self._path
        if save_path is None:
            raise ValueError("No path specified for saving")

        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as f:
            for entry in self._entries.values():
                f.write(json.dumps(entry.to_dict()) + "\n")

        logger.info("saved %d cached responses to %s", len(self._entries), save_path)

    def _load(self) -> None:
        if not self._path or not self._path.exists():
            return

        count = 0
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("skip malformed JSON line in %s", self._path)
                    continue
                entry = CachedResponse.from_dict(data)
                self._entries[(entry.prompt_hash, entry.model_id)] = entry
                count += 1

        logger.info("loaded %d cached responses from %s", count, self._path)
