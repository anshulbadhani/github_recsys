"""
Centralized configuration for the GitHub RecSys project.
Modify these settings to adapt to different datasets or projects.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PathConfig:
    """Configuration for all file paths in the project."""

    # Directories
    project_root: Path = Path(__file__).parent.parent.parent
    data_dir: Path = field(
        init=False
    )  # init=False means do not handle in the constructor itself
    outputs_dir: Path = field(init=False)  # TODO: Setup outputs dir later
    # for logs of csv (but I am not planning to use CSVs)

    raw_data_dir: Path = field(init=False)

    processed_data_dir: Path = field(init=False)
    repos_dir: Path = field(init=False)
    users_dir: Path = field(init=False)

    embeddings_dir: Path = field(init=False)

    def __post_init__(self):
        """
        To initialize the remaining paths after PathConfig obj creation.
        """
        self.data_dir = self.project_root / "data"
        self.outputs_dir = (
            self.project_root / "outputs"
        )  # TODO: Setup outputs dir later
        # for logs of csv (but I am not planning to use CSVs)

        self.raw_data_dir = self.data_dir / "raw"

        self.processed_data_dir = self.data_dir / "processed"
        self.repos_dir = self.processed_data_dir / "repos"
        self.users_dir = self.processed_data_dir / "users"

        self.embeddings_dir = self.data_dir / "embeddings"

    def get_raw_data_path(self, min_freq: int = 160) -> Path:
        """Get path to raw preprocessed data."""
        return self.raw_data_dir / f"min_freq_{min_freq}_preprocessed_data.pkl"

    def get_repo_metadata_path(self, min_freq: int = 160) -> Path:
        """Get path to cleaned repo metadata."""
        return self.repos_dir / f"min_freq_{min_freq}_cleaned_repos.pkl"

    def get_user_history_path(self, min_freq: int = 160) -> Path:
        """Get path to cleaned user history."""
        return self.users_dir / f"min_freq_{min_freq}_cleaned_users.pkl"

    def get_repo_embeddings_path(self, backend: str = "") -> Path:
        """Get path to repo embeddings."""
        return self.embeddings_dir / f"repo_embeddings{'_' + backend}.pkl"

    def get_user_embeddings_path(self, backend: str = "") -> Path:
        """Get path to user embeddings."""
        return self.embeddings_dir / f"user_embeddings{'_' + backend}.pkl"

    def get_csv_output_path(self, split: str) -> Path:
        """Get path for stats or logs or others."""
        return self.outputs_dir / f"{split}.csv"

    def ensure_dirs(self):
        """Create all necessary directories."""
        dirs = [
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.repos_dir,
            self.users_dir,
            self.embeddings_dir,
            self.outputs_dir,
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """Configuration for embedding models and parameters."""

    # Embedding model
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    device: str = "cpu"  # "cpu", "cuda", "openvino", "auto"
    batch_size: int = 64
    normalize_embeddings: bool = True

    # TODO: Review this later
    # User aggregation method
    user_aggregation: str = "mean"  # "mean", "weighted", "last_k"
    user_last_k: Optional[int] = None  # Only used if aggregation is "last_k"


@dataclass
class DataConfig:
    """Configuration for data filtering and processing."""

    # Data version
    min_freq: int = 160  # Minimum frequency threshold (40 or 160)
    # 160 is smaller and faster to process

    # Filtering thresholds (currently not used but available for future)
    # To avoid cold start while prototyping
    user_interaction_threshold: int = 5
    repo_interaction_threshold: int = 10

    # Data splits
    # According to data set by *Kim et al.*
    train_split: str = "train"
    valid_split: str = "valid"
    test_split: str = "test"


@dataclass
class Config:
    """Main configuration object combining all sub-configs."""

    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def __post_init__(self):
        """Ensure all directories exist."""
        self.paths.ensure_dirs()


# Global config instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def update_config(**kwargs):
    """
    Update configuration parameters.

    Example:
        update_config(min_freq=160, embedding_model_name="all-mpnet-base-v2")
    """
    global config

    # Update data config
    for key in ["min_freq", "user_interaction_threshold", "repo_interaction_threshold"]:
        if key in kwargs:
            setattr(config.data, key, kwargs[key])

    # Update model config
    for key in ["embedding_model_name", "embedding_dim", "device", "batch_size"]:
        if key in kwargs:
            setattr(config.model, key, kwargs[key])

    # Ensure directories exist after config update
    config.paths.ensure_dirs()


if __name__ == "__main__":
    # Print current configuration
    cfg = get_config()
    print("=" * 60)
    print("CURRENT CONFIGURATION")
    print("=" * 60)
    print(f"\nData Config:")
    print(f"  Min Frequency: {cfg.data.min_freq}")
    print(f"  Train Split: {cfg.data.train_split}")

    print(f"\nModel Config:")
    print(f"  Embedding Model: {cfg.model.embedding_model_name}")
    print(f"  Embedding Dim: {cfg.model.embedding_dim}")
    print(f"  Device: {cfg.model.device}")

    print(f"\nPath Config:")
    print(f"  Project Root: {cfg.paths.project_root}")
    print(f"  Raw Data: {cfg.paths.get_raw_data_path(cfg.data.min_freq)}")
    print(f"  Repo Metadata: {cfg.paths.get_repo_metadata_path(cfg.data.min_freq)}")
    print(f"  User History: {cfg.paths.get_user_history_path(cfg.data.min_freq)}")
