"""
Settings: Configuration management for Lunar Router.

Uses pydantic-settings for environment variable support and validation.
"""

from typing import Optional
from pathlib import Path


class Settings:
    """
    Lunar Router configuration settings.

    Can be configured via environment variables with UNIROUTE_ prefix.

    Attributes:
        embedding_provider: Embedding provider ("openai", "sentence_transformers").
        embedding_model: Model name for embeddings.
        embedding_dimension: Embedding vector dimension.
        num_clusters: Number of clusters K.
        use_learned_map: Use learned cluster map vs K-Means.
        learned_map_temperature: Temperature for learned map softmax.
        default_cost_weight: Default λ for cost penalty.
        use_soft_assignment: Use soft vs hard cluster assignment.
        state_path: Path for storing system state.
        api_host: API server host.
        api_port: API server port.
        openai_api_key: OpenAI API key.
        anthropic_api_key: Anthropic API key.
    """

    def __init__(
        self,
        # Embedding settings
        embedding_provider: str = "openai",
        embedding_model: str = "text-embedding-3-small",
        embedding_dimension: int = 1536,

        # Clustering settings
        num_clusters: int = 100,
        use_learned_map: bool = False,
        learned_map_temperature: float = 1.0,

        # Router settings
        default_cost_weight: float = 0.0,
        use_soft_assignment: bool = True,

        # Storage settings
        state_path: str = "./uniroute_state",

        # API settings
        api_host: str = "0.0.0.0",
        api_port: int = 8000,

        # API keys
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
    ):
        """Initialize settings."""
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension

        self.num_clusters = num_clusters
        self.use_learned_map = use_learned_map
        self.learned_map_temperature = learned_map_temperature

        self.default_cost_weight = default_cost_weight
        self.use_soft_assignment = use_soft_assignment

        self.state_path = state_path

        self.api_host = api_host
        self.api_port = api_port

        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key

    @classmethod
    def from_env(cls) -> "Settings":
        """
        Create settings from environment variables.

        Environment variables should be prefixed with UNIROUTE_.
        For example: UNIROUTE_EMBEDDING_PROVIDER, UNIROUTE_NUM_CLUSTERS, etc.

        Returns:
            Settings instance configured from environment.
        """
        import os

        def get_env(key: str, default: str) -> str:
            return os.environ.get(f"UNIROUTE_{key}", default)

        def get_env_int(key: str, default: int) -> int:
            val = os.environ.get(f"UNIROUTE_{key}")
            return int(val) if val else default

        def get_env_float(key: str, default: float) -> float:
            val = os.environ.get(f"UNIROUTE_{key}")
            return float(val) if val else default

        def get_env_bool(key: str, default: bool) -> bool:
            val = os.environ.get(f"UNIROUTE_{key}")
            if val is None:
                return default
            return val.lower() in ("true", "1", "yes")

        return cls(
            embedding_provider=get_env("EMBEDDING_PROVIDER", "openai"),
            embedding_model=get_env("EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_dimension=get_env_int("EMBEDDING_DIMENSION", 1536),

            num_clusters=get_env_int("NUM_CLUSTERS", 100),
            use_learned_map=get_env_bool("USE_LEARNED_MAP", False),
            learned_map_temperature=get_env_float("LEARNED_MAP_TEMPERATURE", 1.0),

            default_cost_weight=get_env_float("DEFAULT_COST_WEIGHT", 0.0),
            use_soft_assignment=get_env_bool("USE_SOFT_ASSIGNMENT", True),

            state_path=get_env("STATE_PATH", "./uniroute_state"),

            api_host=get_env("API_HOST", "0.0.0.0"),
            api_port=get_env_int("API_PORT", 8000),

            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "Settings":
        """
        Load settings from a JSON file.

        Args:
            path: Path to the JSON settings file.

        Returns:
            Settings instance.
        """
        import json

        with open(path) as f:
            data = json.load(f)

        return cls(**data)

    def to_dict(self) -> dict:
        """Convert settings to dictionary."""
        return {
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "num_clusters": self.num_clusters,
            "use_learned_map": self.use_learned_map,
            "learned_map_temperature": self.learned_map_temperature,
            "default_cost_weight": self.default_cost_weight,
            "use_soft_assignment": self.use_soft_assignment,
            "state_path": self.state_path,
            "api_host": self.api_host,
            "api_port": self.api_port,
            # Don't include API keys in serialization
        }

    def save(self, path: str | Path) -> None:
        """
        Save settings to a JSON file.

        Args:
            path: Path to save the settings.
        """
        import json

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def __repr__(self) -> str:
        return (
            f"Settings("
            f"provider={self.embedding_provider}, "
            f"clusters={self.num_clusters}, "
            f"cost_weight={self.default_cost_weight})"
        )


def get_settings() -> Settings:
    """
    Get settings instance.

    Tries to load from environment, falls back to defaults.

    Returns:
        Settings instance.
    """
    return Settings.from_env()
