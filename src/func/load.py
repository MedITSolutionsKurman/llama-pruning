import yaml
import os

from src.config.prune_config import PruneConfig
from src.config.prune_ft_config import PruneFTConfig
from src.config.prune_eval_config import PruneEvalConfig
from src.config.prune_grid_search_config import PruneGridSearchConfig
from src.config.method_config import MethodConfig
from dacite import from_dict, Config


# This method loads the configuration file from the given path
# and returns the configuration as a PruneConfig object.
def load_config(config_path: str) -> PruneConfig:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at path: {config_path}")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        config = from_dict(
            data_class=PruneConfig,
            data=config,
            config=Config(cast=[PruneFTConfig, PruneEvalConfig, PruneGridSearchConfig, MethodConfig, bool, float, int]),
        )
    return config
