"""
LLM Registry: Manages the pool of available LLMs for routing.

The registry holds LLMProfile objects and provides methods for
querying and filtering available models.
"""

from pathlib import Path
from typing import Optional, Iterator
import json

from .llm_profile import LLMProfile


class LLMRegistry:
    """
    Registry of LLM profiles available for routing.

    Manages a pool of LLMs and their profiles, supporting dynamic
    addition and removal of models.

    Attributes:
        _profiles: Dictionary mapping model_id to LLMProfile.
        _default_model: ID of the default model to use as fallback.
    """

    def __init__(self):
        """Initialize an empty registry."""
        self._profiles: dict[str, LLMProfile] = {}
        self._default_model: Optional[str] = None

    def __len__(self) -> int:
        """Return the number of registered models."""
        return len(self._profiles)

    def __iter__(self) -> Iterator[LLMProfile]:
        """Iterate over all profiles."""
        return iter(self._profiles.values())

    def __contains__(self, model_id: str) -> bool:
        """Check if a model is registered."""
        return model_id in self._profiles

    def register(self, profile: LLMProfile) -> None:
        """
        Register a new model profile.

        If this is the first model, it becomes the default.

        Args:
            profile: The LLMProfile to register.
        """
        self._profiles[profile.model_id] = profile

        if self._default_model is None:
            self._default_model = profile.model_id

    def unregister(self, model_id: str) -> Optional[LLMProfile]:
        """
        Remove a model from the registry.

        Args:
            model_id: ID of the model to remove.

        Returns:
            The removed profile, or None if not found.
        """
        profile = self._profiles.pop(model_id, None)

        # Update default if necessary
        if self._default_model == model_id:
            if self._profiles:
                self._default_model = next(iter(self._profiles.keys()))
            else:
                self._default_model = None

        return profile

    def get(self, model_id: str) -> Optional[LLMProfile]:
        """
        Get a profile by model ID.

        Args:
            model_id: ID of the model.

        Returns:
            The LLMProfile, or None if not found.
        """
        return self._profiles.get(model_id)

    def get_all(self) -> list[LLMProfile]:
        """Return list of all registered profiles."""
        return list(self._profiles.values())

    def get_model_ids(self) -> list[str]:
        """Return list of all registered model IDs."""
        return list(self._profiles.keys())

    def get_available_models(
        self,
        model_ids: Optional[list[str]] = None,
    ) -> list[LLMProfile]:
        """
        Get profiles for specific models, or all if none specified.

        Args:
            model_ids: Optional list of model IDs to filter by.

        Returns:
            List of LLMProfile objects.
        """
        if model_ids is None:
            return self.get_all()

        return [
            self._profiles[mid]
            for mid in model_ids
            if mid in self._profiles
        ]

    def set_default(self, model_id: str) -> None:
        """
        Set the default model.

        Args:
            model_id: ID of the model to set as default.

        Raises:
            ValueError: If model is not registered.
        """
        if model_id not in self._profiles:
            raise ValueError(f"Model '{model_id}' not registered")
        self._default_model = model_id

    def get_default(self) -> Optional[LLMProfile]:
        """
        Get the default model profile.

        Returns:
            The default LLMProfile, or None if no models registered.
        """
        if self._default_model is None:
            return None
        return self._profiles.get(self._default_model)

    @property
    def default_model_id(self) -> Optional[str]:
        """Return the default model ID."""
        return self._default_model

    def filter_by_cost(
        self,
        max_cost: float,
    ) -> list[LLMProfile]:
        """
        Get profiles with cost at or below the threshold.

        Args:
            max_cost: Maximum cost per 1k tokens.

        Returns:
            List of profiles meeting the cost constraint.
        """
        return [
            p for p in self._profiles.values()
            if p.cost_per_1k_tokens <= max_cost
        ]

    def filter_by_accuracy(
        self,
        min_accuracy: float,
    ) -> list[LLMProfile]:
        """
        Get profiles with overall accuracy at or above the threshold.

        Args:
            min_accuracy: Minimum overall accuracy (0 to 1).

        Returns:
            List of profiles meeting the accuracy constraint.
        """
        return [
            p for p in self._profiles.values()
            if p.overall_accuracy >= min_accuracy
        ]

    def get_cheapest(self) -> Optional[LLMProfile]:
        """Get the cheapest registered model."""
        if not self._profiles:
            return None
        return min(self._profiles.values(), key=lambda p: p.cost_per_1k_tokens)

    def get_most_accurate(self) -> Optional[LLMProfile]:
        """Get the most accurate registered model."""
        if not self._profiles:
            return None
        return max(self._profiles.values(), key=lambda p: p.overall_accuracy)

    def get_best_for_cluster(self, cluster_id: int) -> Optional[LLMProfile]:
        """
        Get the model with lowest error for a specific cluster.

        Args:
            cluster_id: The cluster index.

        Returns:
            The best profile for that cluster, or None if empty.
        """
        if not self._profiles:
            return None

        return min(
            self._profiles.values(),
            key=lambda p: p.get_cluster_error(cluster_id)
        )

    def save(self, directory: str | Path) -> None:
        """
        Save all profiles to a directory.

        Creates one JSON file per profile.

        Args:
            directory: Directory to save profiles to.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Save registry metadata
        metadata = {
            "default_model": self._default_model,
            "model_ids": list(self._profiles.keys()),
        }
        with open(directory / "_registry.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save each profile
        for profile in self._profiles.values():
            # Sanitize model_id for filename
            safe_name = profile.model_id.replace("/", "_").replace(":", "_")
            profile.save(directory / f"{safe_name}.json")

    @classmethod
    def load(cls, directory: str | Path) -> "LLMRegistry":
        """
        Load registry from a directory.

        Args:
            directory: Directory containing profile JSON files.

        Returns:
            LLMRegistry instance with loaded profiles.
        """
        directory = Path(directory)
        registry = cls()

        # Load registry metadata if exists
        metadata_path = directory / "_registry.json"
        default_model = None
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
                default_model = metadata.get("default_model")

        # Load all profile files
        for path in directory.glob("*.json"):
            if path.name == "_registry.json":
                continue
            try:
                profile = LLMProfile.load(path)
                registry.register(profile)
            except Exception as e:
                # Skip invalid files
                print(f"Warning: Could not load {path}: {e}")

        # Restore default model
        if default_model and default_model in registry:
            registry.set_default(default_model)

        return registry

    def summary(self) -> str:
        """
        Generate a summary of all registered models.

        Returns:
            Formatted string with model statistics.
        """
        if not self._profiles:
            return "Registry is empty"

        lines = [
            f"LLM Registry: {len(self._profiles)} models",
            "-" * 60,
            f"{'Model ID':<30} {'Accuracy':>10} {'Cost/1k':>10}",
            "-" * 60,
        ]

        for profile in sorted(self._profiles.values(), key=lambda p: -p.overall_accuracy):
            lines.append(
                f"{profile.model_id:<30} "
                f"{profile.overall_accuracy:>9.1%} "
                f"${profile.cost_per_1k_tokens:>9.4f}"
            )

        lines.append("-" * 60)
        if self._default_model:
            lines.append(f"Default: {self._default_model}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"LLMRegistry(models={list(self._profiles.keys())})"
