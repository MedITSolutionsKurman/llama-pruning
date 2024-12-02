from dataclasses import dataclass
from typing import Optional, Union
from datasets import load_dataset, Dataset
from dataset.prepare import prepare_chat, prepare_iio_chat

import logging

logger = logging.getLogger()

@dataclass
class PruneEvalConfig:
    apply_chat_template: bool = False
    cache_dir: Optional[str] = None
    eval_dataset: Optional[Union[str, Dataset]] = None
    eval_dataset_size: Optional[int] = None
    eval_max_length: Optional[int] = 128
    eval_batch_size: Optional[int] = 1

    
    def __post_init__(self):
        if self.eval_dataset is not None and self.eval_dataset_size is None:
            logger.warning("eval_dataset_size is not set. Defaulting to 20.")
            self.eval_dataset_size = 20

        if self.eval_dataset is not None:
            self.eval_dataset = load_dataset(
                self.eval_dataset,
                split=f"test[:{self.eval_dataset_size}]",
                cache_dir=self.cache_dir,
            )

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
                    self.eval_dataset = prepare_iio_chat(self.eval_dataset)

            elif (
                self.apply_chat_template
                and "conversations" in self.eval_dataset.column_names
            ):
                self.eval_dataset = prepare_chat(self.eval_dataset)

            if (
                not self.apply_chat_template
                and "text" not in self.eval_dataset.column_names
            ):
                raise ValueError(
                    "eval_dataset must have 'text' column if use_chat_template is False."
                )
