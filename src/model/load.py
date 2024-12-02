from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Optional, Union
from torch import dtype

import torch
import bitsandbytes as bnb


# Load the model and tokenizer.
def load_model(
    model_name: str,
    device: str = "cuda",
    dtype: Optional[Union[str, dtype]] = torch.float32,
    cache_dir: Optional[str] = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Loads the model and tokenizer.

    Args:
    - model_name: Name of the model to load.
    - dtype: Data type to use.
    - cache_dir: Directory to cache the model.

    Returns:
    - model: Model loaded.
    - tokenizer: Tokenizer loaded.
    """
    # Load the model and tokenizer.

    if load_in_4bit or load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
        )
    else:
        quantization_config = None


    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, cache_dir=cache_dir, device_map=device, quantization_config=quantization_config
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    return model, tokenizer
