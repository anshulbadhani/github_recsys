"""Configuration package for github-recsys."""

# Absolute imports (recommended when config is inside recsys/)
from .config import (
    Config,
    PathConfig,
    ModelConfig,
    DataConfig,
    get_config,
    update_config
)

__all__ = ['Config', 'PathConfig', 'ModelConfig', 'DataConfig', 'get_config', 'update_config']