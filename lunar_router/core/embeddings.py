"""
Text embedding components for UniRoute.

Implements the prompt embedding function φ(x) → R^d.
Supports multiple backends: OpenAI, SentenceTransformers, etc.
"""

from abc import ABC, abstractmethod
from typing import Protocol, Optional, runtime_checkable
import hashlib
import numpy as np


@runtime_checkable
class EmbeddingProvider(Protocol):
    """
    Protocol for embedding providers.

    Any class implementing embed() and embed_batch() can be used as a provider.
    """

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text and return a numpy array."""
        ...

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed multiple texts and return a 2D numpy array (N, d)."""
        ...

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...


class PromptEmbedder:
    """
    Main class for generating text embeddings.

    Wraps an EmbeddingProvider and optionally adds caching.

    Attributes:
        provider: The underlying embedding provider.
        cache: Optional dictionary cache for embeddings.
    """

    def __init__(
        self,
        provider: EmbeddingProvider,
        cache_enabled: bool = True,
        cache_max_size: int = 10000,
    ):
        """
        Initialize the embedder.

        Args:
            provider: The embedding provider to use.
            cache_enabled: Whether to cache embeddings.
            cache_max_size: Maximum number of embeddings to cache.
        """
        self.provider = provider
        self.cache_enabled = cache_enabled
        self.cache_max_size = cache_max_size
        self._cache: dict[str, np.ndarray] = {} if cache_enabled else {}

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self.provider.dimension

    def _cache_key(self, text: str) -> str:
        """Generate a cache key for the text."""
        return hashlib.md5(text.encode()).hexdigest()

    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            Embedding vector φ(x) ∈ R^d.
        """
        if self.cache_enabled:
            key = self._cache_key(text)
            if key in self._cache:
                return self._cache[key]

        embedding = self.provider.embed(text)

        if self.cache_enabled:
            if len(self._cache) >= self.cache_max_size:
                # Simple eviction: remove oldest entries (at least 1)
                num_to_remove = max(1, self.cache_max_size // 10)
                keys_to_remove = list(self._cache.keys())[:num_to_remove]
                for k in keys_to_remove:
                    del self._cache[k]
            self._cache[key] = embedding

        return embedding

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            2D numpy array of shape (N, d) where N is len(texts).
        """
        if not texts:
            return np.array([])

        if self.cache_enabled:
            # Check which texts are cached
            cached_embeddings = {}
            uncached_texts = []
            uncached_indices = []

            for i, text in enumerate(texts):
                key = self._cache_key(text)
                if key in self._cache:
                    cached_embeddings[i] = self._cache[key]
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)

            # Get embeddings for uncached texts
            if uncached_texts:
                new_embeddings = self.provider.embed_batch(uncached_texts)

                # Cache the new embeddings
                for j, text in enumerate(uncached_texts):
                    key = self._cache_key(text)
                    self._cache[key] = new_embeddings[j]
                    cached_embeddings[uncached_indices[j]] = new_embeddings[j]

            # Reconstruct in original order
            result = np.array([cached_embeddings[i] for i in range(len(texts))])
            return result
        else:
            return self.provider.embed_batch(texts)

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()


# --- Concrete Embedding Providers ---


class OpenAIEmbeddingProvider:
    """
    Embedding provider using OpenAI's API.

    Uses text-embedding-3-small by default.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        dimensions: Optional[int] = None,
    ):
        """
        Initialize OpenAI embedding provider.

        Args:
            model: OpenAI embedding model name.
            api_key: Optional API key (defaults to OPENAI_API_KEY env var).
            dimensions: Optional dimension override for models that support it.
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

        self.model = model
        self._dimensions = dimensions
        self.client = OpenAI(api_key=api_key)

        # Known dimensions for common models
        self._model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

    @property
    def dimension(self) -> int:
        if self._dimensions:
            return self._dimensions
        return self._model_dimensions.get(self.model, 1536)

    def embed(self, text: str) -> np.ndarray:
        kwargs = {"input": text, "model": self.model}
        if self._dimensions:
            kwargs["dimensions"] = self._dimensions

        response = self.client.embeddings.create(**kwargs)
        return np.array(response.data[0].embedding)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        kwargs = {"input": texts, "model": self.model}
        if self._dimensions:
            kwargs["dimensions"] = self._dimensions

        response = self.client.embeddings.create(**kwargs)

        # Sort by index to maintain order
        embeddings = sorted(response.data, key=lambda x: x.index)
        return np.array([e.embedding for e in embeddings])


class SentenceTransformerProvider:
    """
    Embedding provider using SentenceTransformers library.

    Supports any model from the sentence-transformers library.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        """
        Initialize SentenceTransformer provider.

        Args:
            model_name: Name of the sentence-transformers model.
            device: Device to use (e.g., "cuda", "cpu"). Auto-detected if None.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers package required. "
                "Install with: pip install sentence-transformers"
            )

        # Workaround for HuggingFace Hub API issues with some model metadata
        import warnings
        import logging

        # Suppress HTTP errors for missing metadata (like additional_chat_templates)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            try:
                # Try loading with trust_remote_code=False to avoid some issues
                self.model = SentenceTransformer(model_name, device=device, trust_remote_code=False)
            except TypeError:
                # Older versions don't have trust_remote_code parameter
                self.model = SentenceTransformer(model_name, device=device)

        self._dimension = self.model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> np.ndarray:
        return self.model.encode(text, convert_to_numpy=True)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)


class MockEmbeddingProvider:
    """
    Mock embedding provider for testing.

    Generates deterministic random embeddings based on text hash.
    """

    def __init__(self, dimension: int = 384):
        """
        Initialize mock provider.

        Args:
            dimension: Dimension of embeddings to generate.
        """
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _text_to_seed(self, text: str) -> int:
        """Convert text to a deterministic seed."""
        return int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)

    def embed(self, text: str) -> np.ndarray:
        np.random.seed(self._text_to_seed(text))
        vec = np.random.randn(self._dimension)
        # Normalize to unit length
        return vec / np.linalg.norm(vec)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.array([self.embed(t) for t in texts])
