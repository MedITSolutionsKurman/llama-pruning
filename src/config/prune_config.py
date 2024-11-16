from dataclasses import dataclass, field
from typing import Optional, List, Union
from config.prune_grid_search_config import PruneGridSearchConfig
from config.prune_method import PruneMethod
from dataset.map import map_iio, map_conversations
from datasets import load_dataset, Dataset

import logging
import os

logger = logging.getLogger(__name__)


# Dataclass to store the prune configuration.
@dataclass
class PruneConfig:
    model_name: str = "meditsolutions/Llama-3.2-SUN-1B-chat"
    dtype: str = "float32"
    device: str = "cuda"
    output: str = "results/pruned_model"
    apply_chat_template: bool = False
    prompt: str = "What is the capital of France?"
    apply_chat_template: bool = False
    max_new_tokens: int = 50
    prune_percent: float = 0.2
    prune_method: str = "mk_prune"
    use_normalized_weights: bool = False
    use_layer_norm_tweaks: bool = False
    layer_norm_scale: float = 4.0
    log_dir: str = "logs"
    stop_logging: bool = False
    test_only: bool = False
    print_summary: bool = False
    quiet: bool = False
    cache_dir: Optional[str] = None
    target_size: Optional[int] = None
    use_full_precision: bool = False
    gate_up_down_weight_weights: Optional[List[float]] = field(
        default_factory=lambda: [1.0, 1.0, 1.0]
    )
    eval_dataset: Optional[Union[str, Dataset]] = None
    eval_dataset_size: Optional[int] = None
    grid_search: Optional[PruneGridSearchConfig] = None

    def __post_init__(self):
        # Check if the prune method is valid.
        if self.prune_method in ["mk", PruneMethod.MK_PRUNE]:
            self.prune_method = PruneMethod.MK_PRUNE
        elif self.prune_method in ["mka", PruneMethod.MK_PRUNE_ADJUSTED]:
            self.prune_method = PruneMethod.MK_PRUNE_ADJUSTED
        elif self.prune_method in ["mka2", PruneMethod.MK_PRUNE_ADJUSTED_2]:
            self.prune_method = PruneMethod.MK_PRUNE_ADJUSTED_2
        elif self.prune_method in [
            "mka2g",
            PruneMethod.MK_PRUNE_ADJUSTED_2_WITH_GRADIENTS,
        ]:
            self.prune_method = PruneMethod.MK_PRUNE_ADJUSTED_2_WITH_GRADIENTS

            if self.eval_dataset is None:
                raise ValueError(
                    "eval_dataset with `conversations` column is required for this method if use_chat_template is True. `text` column is required otherwise."
                )
        else:
            raise ValueError(f"Unknown prune method: {self.prune_method}")

        # Validate parameters
        if self.use_layer_norm_tweaks and self.layer_norm_scale <= 0:
            raise ValueError("layer_norm_scale must be greater than 0.")

        if self.gate_up_down_weight_weights is not None:
            if len(self.gate_up_down_weight_weights) != 3:
                raise ValueError("gate_up_down_weight_weights must have 3 values.")
            if any(
                [
                    weight < 0 or weight > 1.0
                    for weight in self.gate_up_down_weight_weights
                ]
            ):
                raise ValueError("gate_up_down_weight_weights must be between 0 and 1")

        if self.target_size is not None:
            if self.target_size <= 0:
                raise ValueError("target_size must be greater than 0.")

        if self.prune_percent < 0 or self.prune_percent > 1.0:
            raise ValueError("prune_percent must be between 0 and 1")

        if self.max_new_tokens <= 10:
            raise ValueError("max_new_tokens must be greater than 10")

        if self.cache_dir is not None:
            if not os.path.exists(self.cache_dir):
                raise ValueError("cache_dir does not exist.")

        if not os.path.exists(self.output):
            os.makedirs(self.output)

        if self.eval_dataset is not None and self.eval_dataset_size is None:
            logger.warning("eval_dataset_size is not set. Defaulting to 20.")
            self.eval_dataset_size = 20

        if self.eval_dataset is not None:
            self.eval_dataset = load_dataset(
                self.eval_dataset,
                split=f"train[:{self.eval_dataset_size}]",
                cache_dir=self.cache_dir,
            )

            # Get num of available processors
            num_proc = os.cpu_count()

            if (
                self.apply_chat_template
                and "conversations" not in self.eval_dataset.column_names
            ):
                if (
                    "input" not in self.eval_dataset.column_names
                    and "output" not in self.eval_dataset.column_names
                ):
                    raise ValueError(
                        "eval_dataset must have 'conversations' or 'input' and 'output' columns if use_chat_template is True."
                    )
                else:
                    self.eval_dataset = (
                        self.eval_dataset.map(
                            lambda x: {"conversations": map_iio(x)},
                            num_proc=num_proc,
                            remove_columns=[
                                x
                                for x in self.eval_dataset.column_names
                                if x not in ["conversations"]
                            ],
                        )
                        .filter(lambda x: x["conversations"] is not None)
                        .map(
                            lambda x: {
                                "conversations": map_conversations(x["conversations"])
                            },
                            num_proc=num_proc,
                        )
                        .filter(lambda x: x["conversations"] is not None)
                    )

            elif (
                self.apply_chat_template
                and "conversations" in self.eval_dataset.column_names
            ):
                self.eval_dataset = self.eval_dataset.map(
                    lambda x: {"conversations": map_conversations(x["conversations"])},
                    remove_columns=[
                        x
                        for x in self.eval_dataset.column_names
                        if x not in ["conversations"]
                    ],
                    num_proc=num_proc,
                ).filter(lambda x: x["conversations"] is not None)

            if (
                not self.apply_chat_template
                and "text" not in self.eval_dataset.column_names
            ):
                raise ValueError(
                    "eval_dataset must have 'text' column if use_chat_template is False."
                )

        if self.grid_search is not None and self.eval_dataset is None:
            raise ValueError("Grid search (AutoML) requires an eval_dataset.")
