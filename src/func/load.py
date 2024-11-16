import yaml
import os

from src.config.prune_config import PruneConfig


# This method loads the configuration file from the given path
# and returns the configuration as a PruneConfig object.
def load_config(config_path: str) -> PruneConfig:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at path: {config_path}")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        config = PruneConfig(**config)
    return config
