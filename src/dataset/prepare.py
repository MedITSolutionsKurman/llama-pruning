from datasets import Dataset
from src.dataset.map import map_iio, map_conversations

import os


def prepare_iio_chat(dataset: Dataset) -> Dataset:
    """

    Prepares the dataset for the input, instruction, and output format.

    Args:
    - dataset: Dataset to prepare.

    Returns:
    - Prepared dataset.

    """

    # Get number of available processors
    num_proc = os.cpu_count()

    dataset = dataset.map(
        lambda x: {"conversations": map_iio(x)},
        num_proc=num_proc,
        remove_columns=[x for x in dataset.column_names if x not in ["conversations"]],
    ).filter(lambda x: x["conversations"] is not None)

    return prepare_chat(dataset)


def prepare_chat(dataset: Dataset) -> Dataset:
    """

    Prepares the dataset for the chat format.

    Args:
    - dataset: Dataset to prepare.

    Returns:
    - Prepared dataset.
    """

    # Get number of available processors
    num_proc = os.cpu_count()

    return dataset.map(
        lambda x: {"conversations": map_conversations(x["conversations"])},
        remove_columns=[x for x in dataset.column_names if x not in ["conversations"]],
        num_proc=num_proc,
    ).filter(lambda x: x["conversations"] is not None)
