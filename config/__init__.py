"""Configuration package for github-recsys."""
from .config import Config, PathConfig, ModelConfig, DataConfig, get_config, update_config
 
__all__ = ['Config', 'PathConfig', 'ModelConfig', 'DataConfig', 'get_config', 'update_config']