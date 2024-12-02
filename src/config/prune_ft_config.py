from dataclasses import dataclass
from typing import Optional, Union
from datasets import load_dataset, Dataset
from dataset.prepare import prepare_chat, prepare_iio_chat
from src.config.prune_ft_method import PruneFTMethod
import os

import logging

logger = logging.getLogger()

@dataclass
class PruneFTConfig:
    method: PruneFTMethod = PruneFTMethod.MLP_ONLY
    apply_chat_template: bool = False
    cache_dir: Optional[str] = None
    train_dataset: Optional[Union[str, Dataset]] = None
    train_dataset_size: Optional[int] = None
    train_max_length: Optional[int] = 128
    train_batch_size: Optional[int] = 1
    train_epochs: Optional[int] = 0
    train_learning_rate: Optional[float] = 1e-5
    train_accumulation_steps: Optional[int] = 1
    train_warmup_steps: Optional[int] = 10
    train_weight_decay: Optional[float] = 0.01
    train_save_steps: Optional[int] = 0
    train_checkpoint_dir: Optional[str] = None

    
    def __post_init__(self):
        if self.method not in PruneFTMethod.__dict__.values():
            raise ValueError(f"Invalid fine-tuning method: {self.method}")

        if self.train_dataset is not None and self.train_dataset_size is None:
            logger.warning("train_dataset_size is not set. Defaulting to 20.")
            self.train_dataset_size = 20

        if self.train_dataset is not None:
            self.train_dataset = load_dataset(
                self.train_dataset,
                split=f"train[:{self.train_dataset_size}]",
                cache_dir=self.cache_dir,
            )

            if (
                self.apply_chat_template
                and "conversations" not in self.train_dataset.column_names
            ):
                if 'messages' in self.train_dataset.column_names:
                    self.train_dataset = self.train_dataset.rename_column('messages', 'conversations')
                    self.train_dataset = prepare_chat(self.train_dataset)
                    return
                
                if (
                    "input" not in self.train_dataset.column_names
                    and "output" not in self.train_dataset.column_names
                ):
                    raise ValueError(
                        "train_dataset must have 'conversations' or 'input' and 'output' columns if use_chat_template is True."
                    )
                else:
                    self.train_dataset = prepare_iio_chat(self.train_dataset)

            elif (
                self.apply_chat_template
                and "conversations" in self.train_dataset.column_names
            ):
                self.train_dataset = prepare_chat(self.train_dataset)

            if (
                not self.apply_chat_template
                and "text" not in self.train_dataset.column_names
            ):
                raise ValueError(
                    "train_dataset must have 'text' column if use_chat_template is False."
                )
            
            if self.train_checkpoint_dir is not None:
                if not os.path.exists(self.train_checkpoint_dir):
                    os.makedirs(self.train_checkpoint_dir)

            if self.train_save_steps < 0:
                raise ValueError("train_save_steps must be greater than or equal to 0")
            
            if self.train_save_steps > 0 and self.train_checkpoint_dir is None:
                raise ValueError("train_checkpoint_dir must be set if train_save_steps is greater than 0")