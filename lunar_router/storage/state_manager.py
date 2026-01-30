"""
State Manager: Handles persistence of UniRoute system state.

Manages saving and loading of:
- Cluster assigners (centroids, θ)
- LLM profiles (Ψ vectors)
- Configuration and metadata
"""

from pathlib import Path
from typing import Optional
import json
import shutil
from datetime import datetime

from ..core.clustering import (
    ClusterAssigner,
    KMeansClusterAssigner,
    LearnedMapClusterAssigner,
    load_cluster_assigner,
)
from ..models.llm_profile import LLMProfile
from ..models.llm_registry import LLMRegistry
from ..data.dataset import PromptDataset


class StateManager:
    """
    Manages persistence of UniRoute system state.

    Provides a unified interface for saving/loading all components
    needed to reconstruct the routing system.

    Directory structure:
        base_path/
        ├── clusters/
        │   └── default.npz      # Cluster assigner state
        ├── profiles/
        │   ├── gpt-4o.json
        │   └── claude-3.json
        ├── datasets/
        │   ├── s_tr.json
        │   └── s_val.json
        ├── config/
        │   └── settings.json
        └── metadata.json

    Attributes:
        base_path: Root directory for all state files.
    """

    def __init__(self, base_path: str | Path):
        """
        Initialize the state manager.

        Args:
            base_path: Root directory for state storage.
        """
        self.base_path = Path(base_path)
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create necessary subdirectories."""
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.clusters_dir = self.base_path / "clusters"
        self.profiles_dir = self.base_path / "profiles"
        self.datasets_dir = self.base_path / "datasets"
        self.config_dir = self.base_path / "config"

        for d in [self.clusters_dir, self.profiles_dir, self.datasets_dir, self.config_dir]:
            d.mkdir(exist_ok=True)

    # --- Metadata ---

    def save_metadata(self, metadata: dict) -> None:
        """Save system metadata."""
        metadata["last_updated"] = datetime.now().isoformat()
        with open(self.base_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def load_metadata(self) -> dict:
        """Load system metadata."""
        path = self.base_path / "metadata.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {}

    # --- Cluster Assigners ---

    def save_cluster_assigner(
        self,
        assigner: ClusterAssigner,
        name: str = "default",
    ) -> Path:
        """
        Save a cluster assigner.

        Args:
            assigner: The cluster assigner to save.
            name: Name for the saved state (default: "default").

        Returns:
            Path to the saved file.
        """
        path = self.clusters_dir / f"{name}.npz"
        assigner.save(path)
        return path

    def load_cluster_assigner(self, name: str = "default") -> ClusterAssigner:
        """
        Load a cluster assigner.

        Args:
            name: Name of the saved state.

        Returns:
            Loaded ClusterAssigner (either KMeans or LearnedMap).
        """
        path = self.clusters_dir / f"{name}.npz"
        return load_cluster_assigner(path)

    def has_cluster_assigner(self, name: str = "default") -> bool:
        """Check if a cluster assigner exists."""
        return (self.clusters_dir / f"{name}.npz").exists()

    def list_cluster_assigners(self) -> list[str]:
        """List available cluster assigner names."""
        return [p.stem for p in self.clusters_dir.glob("*.npz")]

    # --- LLM Profiles ---

    def save_profile(self, profile: LLMProfile) -> Path:
        """
        Save an LLM profile.

        Args:
            profile: The profile to save.

        Returns:
            Path to the saved file.
        """
        # Sanitize model_id for filename
        safe_name = profile.model_id.replace("/", "_").replace(":", "_")
        path = self.profiles_dir / f"{safe_name}.json"
        profile.save(path)
        return path

    def load_profile(self, model_id: str) -> LLMProfile:
        """
        Load an LLM profile by model ID.

        Args:
            model_id: The model ID.

        Returns:
            Loaded LLMProfile.
        """
        safe_name = model_id.replace("/", "_").replace(":", "_")
        path = self.profiles_dir / f"{safe_name}.json"
        return LLMProfile.load(path)

    def load_all_profiles(self) -> list[LLMProfile]:
        """Load all available profiles."""
        profiles = []
        for path in self.profiles_dir.glob("*.json"):
            try:
                profiles.append(LLMProfile.load(path))
            except Exception as e:
                print(f"Warning: Could not load {path}: {e}")
        return profiles

    def has_profile(self, model_id: str) -> bool:
        """Check if a profile exists for the given model."""
        safe_name = model_id.replace("/", "_").replace(":", "_")
        return (self.profiles_dir / f"{safe_name}.json").exists()

    def list_profiles(self) -> list[str]:
        """List available profile model IDs."""
        profiles = []
        for path in self.profiles_dir.glob("*.json"):
            try:
                profile = LLMProfile.load(path)
                profiles.append(profile.model_id)
            except Exception:
                pass
        return profiles

    def delete_profile(self, model_id: str) -> bool:
        """
        Delete a profile.

        Args:
            model_id: The model ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        safe_name = model_id.replace("/", "_").replace(":", "_")
        path = self.profiles_dir / f"{safe_name}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    # --- Registry ---

    def load_registry(self) -> LLMRegistry:
        """
        Load all profiles into a registry.

        Returns:
            LLMRegistry with all available profiles.
        """
        registry = LLMRegistry()
        for profile in self.load_all_profiles():
            registry.register(profile)
        return registry

    def save_registry(self, registry: LLMRegistry) -> None:
        """
        Save all profiles from a registry.

        Args:
            registry: The registry to save.
        """
        for profile in registry.get_all():
            self.save_profile(profile)

    # --- Datasets ---

    def save_dataset(
        self,
        dataset: PromptDataset,
        name: Optional[str] = None,
    ) -> Path:
        """
        Save a dataset.

        Args:
            dataset: The dataset to save.
            name: Name for the file (defaults to dataset.name).

        Returns:
            Path to the saved file.
        """
        name = name or dataset.name
        path = self.datasets_dir / f"{name}.json"
        dataset.save(path)
        return path

    def load_dataset(self, name: str) -> PromptDataset:
        """
        Load a dataset by name.

        Args:
            name: Name of the dataset.

        Returns:
            Loaded PromptDataset.
        """
        path = self.datasets_dir / f"{name}.json"
        return PromptDataset.load(path)

    def has_dataset(self, name: str) -> bool:
        """Check if a dataset exists."""
        return (self.datasets_dir / f"{name}.json").exists()

    def list_datasets(self) -> list[str]:
        """List available dataset names."""
        return [p.stem for p in self.datasets_dir.glob("*.json")]

    # --- Configuration ---

    def save_config(self, config: dict, name: str = "settings") -> Path:
        """
        Save configuration.

        Args:
            config: Configuration dictionary.
            name: Name for the config file.

        Returns:
            Path to the saved file.
        """
        path = self.config_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        return path

    def load_config(self, name: str = "settings") -> dict:
        """
        Load configuration.

        Args:
            name: Name of the config file.

        Returns:
            Configuration dictionary.
        """
        path = self.config_dir / f"{name}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {}

    # --- Full System State ---

    def export_state(self, export_path: str | Path) -> Path:
        """
        Export complete system state to a zip file.

        Args:
            export_path: Path for the zip file.

        Returns:
            Path to the created zip file.
        """
        export_path = Path(export_path)
        if export_path.suffix != ".zip":
            export_path = export_path.with_suffix(".zip")

        # Create zip from base_path
        shutil.make_archive(
            str(export_path.with_suffix("")),
            "zip",
            self.base_path,
        )

        return export_path

    def import_state(self, import_path: str | Path) -> None:
        """
        Import system state from a zip file.

        Args:
            import_path: Path to the zip file.
        """
        import_path = Path(import_path)

        # Clear existing state
        shutil.rmtree(self.base_path, ignore_errors=True)

        # Extract zip to base_path
        shutil.unpack_archive(import_path, self.base_path)

        # Ensure directories exist
        self._ensure_directories()

    def clear_all(self) -> None:
        """Clear all stored state."""
        shutil.rmtree(self.base_path, ignore_errors=True)
        self._ensure_directories()

    def get_summary(self) -> dict:
        """
        Get a summary of stored state.

        Returns:
            Dictionary with counts and info about stored state.
        """
        return {
            "base_path": str(self.base_path),
            "cluster_assigners": self.list_cluster_assigners(),
            "profiles": self.list_profiles(),
            "datasets": self.list_datasets(),
            "metadata": self.load_metadata(),
        }

    def __repr__(self) -> str:
        summary = self.get_summary()
        return (
            f"StateManager(path='{self.base_path}', "
            f"clusters={len(summary['cluster_assigners'])}, "
            f"profiles={len(summary['profiles'])}, "
            f"datasets={len(summary['datasets'])})"
        )
