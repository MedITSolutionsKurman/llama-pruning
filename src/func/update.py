import torch
import torch.nn as nn
import bitsandbytes as bnb

from typing import Optional
from tqdm import tqdm
from logging import getLogger

from copy import deepcopy

from transformers import PreTrainedTokenizer, DataCollatorForLanguageModeling

from src.method.mk_prune import prune_neuron_pairs
from src.model.activation_hook import ActivationGradientHooks
from config.prune_method import PruneMethod
from config.prune_config import PruneConfig
from datasets import Dataset
from torch.utils.data import DataLoader
from src.model.load import load_model
from src.model.quantization import check_model_quantization
from src.func.energy.align import align_weight_energies
from src.func.energy.analyze import analyze_layer_energy
from src.func.energy.rescale import rescale_weight
from src.model.ft import train

logger = getLogger()


# Iterates throught the model layers and applies pruning.
# Note: This method was previously copied from the source given below:
# https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/6-PRUNING/6_3_pruning_structured_llama3.2-1b_OK.ipynb
# It was modified to include new ways to calculate the importance score, include the target_size parameter and take normalizations into account.
def update_model(
    model: nn.Module,
    config: PruneConfig,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    deepcopy_model: bool = False,
) -> nn.Module:
    """
    It modifies each mlp layer present in model, to retain only the most
    important neurons. Creating new smaller versions of each layer pruned.

    Args:
    - model: Model to prune.
    - config: Prune configuration.
    - deepcopy_model: If True, the model will be copied before pruning. (default: False)

    Returns:
    - model: New pruned model.
    """
    new_intermediate_size = None
    tokens = None

    logger.info(
        f"Pruning model, using {config.prune_method} method, "
        f"normalizing weights: {config.use_normalized_weights}, "
        f"layer norm tweaks: {config.use_layer_norm_tweaks}, "
        f"layer norm scale: {config.layer_norm_scale} "
        f"target size: {config.target_size}, "
        f"gate up weight weights: {config.gate_up_down_weight_weights}\n"
    )

    if config.training is not None and (config.training.train_epochs > 0 or config.prune_method == PruneMethod.MK_PRUNE_ADJUSTED_2_WITH_GRADIENTS):

        logger.info("Tokenizing eval dataset...")

        # Tokenize eval dataset
        if not config.apply_chat_template:
            tokens = config.training.train_dataset.map(
                lambda x: tokenizer(
                    x["text"],
                    return_tensors="pt",
                    max_length=config.training.train_max_length,
                    truncation=True,
                    padding="max_length",
                ),
                remove_columns=[
                    x for x in config.training.train_dataset.column_names if x != "conversations"
                ],
            )
        else:
            if 'messages' in config.training.train_dataset.column_names:
                config.training.train_dataset = config.training.train_dataset.rename_column("messages", "conversations")

            tokens = config.training.train_dataset.map(
                lambda x: tokenizer.apply_chat_template(
                    x["conversations"],
                    tokenize=True,
                    add_generation_prompt=False,
                    return_tensors="pt",
                    max_length=config.training.train_max_length,
                    truncation=True,
                    return_dict=True,
                    padding="max_length",
                ),
                remove_columns=[
                    x for x in config.training.train_dataset.column_names if x != "conversations"
                ],
            )

        tokens = tokens.with_format("torch")

        tokens = DataLoader(
            tokens.select_columns(["input_ids", "attention_mask"]),
            batch_size=config.training.train_batch_size,
            shuffle=False,
            collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )

    if config.prune_method == PruneMethod.MK_PRUNE:
        pass
    elif config.prune_method == PruneMethod.MK_PRUNE_ADJUSTED:
        pass
    elif config.prune_method == PruneMethod.MK_PRUNE_ADJUSTED_2:
        pass
    elif config.prune_method == PruneMethod.MK_PRUNE_ADJUSTED_2_WITH_GRADIENTS:
        if tokenizer is None:
            raise ValueError("Tokenizer is required for this method.")
        
        if config.training is None:
            raise ValueError("Training configuration is required for this method.")

        if config.training.train_dataset is None:
            raise ValueError("Training dataset is required for this method.")

        if "conversations" not in config.training.train_dataset.column_names:
            raise ValueError("Training dataset must have 'conversations' column.")

        hooks = ActivationGradientHooks()
        hooks.register_hooks(model)

        logger.info("Calculating activations and gradients...")

        for batch in tqdm(tokens):
            input_ids = batch["input_ids"].squeeze(1).to(config.device)
            attention_mask = batch["attention_mask"].squeeze(1).to(config.device)

            if config.device == "cuda":
                with torch.amp.autocast(
                    device_type=config.device, dtype=model.dtype
                ):  # Enable automatic mixed precision
                    outputs = model(
                        input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids,
                        return_dict=True,
                    )
                    loss = outputs.loss
            else:
                outputs = model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                    return_dict=True,
                )
                loss = outputs.loss

            loss.backward()
            model.zero_grad()

            input_ids = input_ids.detach().to("cpu")
            attention_mask = attention_mask.detach().to("cpu")

            torch.cuda.empty_cache()
        pass
    else:
        raise ValueError(f"Unknown method: {config.prune_method}")

    if deepcopy_model:
        model = deepcopy(model)

    quant_info = check_model_quantization(model)

    if quant_info["is_quantized"]:
        logger.info(
            f"Model is quantized. Pruning will be done using {model.config.torch_dtype}."
        )

        model = load_model(
            config.model_name,
            device=config.device,
            dtype=model.config.torch_dtype,
            cache_dir=config.cache_dir,
        )[0]

    # loop for each model layer.
    for idx, layer in tqdm(
        enumerate(model.model.layers), total=len(model.model.layers)
    ):
        # Since each layer is a LlamaDecoderLayer it contains multiple components
        # Attention, MLP and Layer norms. We're targetting MLP component
        # by accesing layer.mlp.
        mlp = layer.mlp

        energy_info = analyze_layer_energy(mlp)

        if config.prune_method == PruneMethod.MK_PRUNE_ADJUSTED_2_WITH_GRADIENTS:
            # Get the activations of the layer.
            name = f"layer_{idx}"
            activations, gradients = hooks.get_layer_statistics(name)

        # Call the prune_neiron_pairs with the layers and receiving the pruned.
        new_gate_proj, new_up_proj, new_down_proj, new_size = prune_neuron_pairs(
            mlp,
            config.prune_percent,
            prune_method=config.prune_method,
            use_normalized_weights=config.use_normalized_weights,
            device=config.device,
            target_size=config.target_size,
            use_full_precision=config.use_full_precision,
            gate_up_down_t_weight_weights=config.gate_up_down_weight_weights,
            activations=(
                [activation / (len(config.training.train_dataset) / config.training.train_batch_size) for activation in activations]
                if config.prune_method == PruneMethod.MK_PRUNE_ADJUSTED_2_WITH_GRADIENTS
                else None
            ),
            gradients=(
                [gradient / (len(config.training.train_dataset) / config.training.train_batch_size) for gradient in gradients]
                if config.prune_method == PruneMethod.MK_PRUNE_ADJUSTED_2_WITH_GRADIENTS
                else None
            ),
            parameters = config.parameters
        )

        if config.use_layer_norm_tweaks:
            # Update the layer normalization weights starting from the second layer.
            if idx < len(model.model.layers) - 1:
                last_layer_original_sum = (
                    torch.abs(mlp.down_proj.weight.data).sum()
                    + torch.abs(mlp.up_proj.weight.data).sum()
                    + torch.abs(mlp.gate_proj.weight.data).sum()
                )
                last_layer_pruned_sum = (
                    torch.abs(new_down_proj.weight.data).sum()
                    + torch.abs(new_up_proj.weight.data).sum()
                    + torch.abs(new_gate_proj.weight.data).sum()
                )

                # Update the next layer normalization weights.
                model.model.layers[idx + 1].input_layernorm.weight.data *= (
                    1.0
                    + (1.0 - torch.abs(last_layer_pruned_sum / last_layer_original_sum))
                    / config.layer_norm_scale
                )

        # new_gate_proj.weight.data = align_weight_energies(
        #     mlp.gate_proj.weight.data, new_gate_proj.weight.data, method="match_distribution"
        # )[1]

        # new_up_proj.weight.data = align_weight_energies(
        #     mlp.up_proj.weight.data, new_up_proj.weight.data, method="match_distribution"
        # )[1]

        # new_down_proj.weight.data = align_weight_energies(
        #     mlp.down_proj.weight.data, new_down_proj.weight.data, method="match_distribution"
        # )[1]

        # Move old layers to CPU and delete them.
        mlp.gate_proj.weight.data = mlp.gate_proj.weight.data.cpu()
        mlp.up_proj.weight.data = mlp.up_proj.weight.data.cpu()
        mlp.down_proj.weight.data = mlp.down_proj.weight.data.cpu()
        
        del mlp.gate_proj
        del mlp.up_proj
        del mlp.down_proj

        # Replace the Origiginal Layers with Pruned Layers.
        mlp.gate_proj = new_gate_proj
        mlp.up_proj = new_up_proj
        mlp.down_proj = new_down_proj

        pruned_energy_info = analyze_layer_energy(mlp)

        torch.cuda.empty_cache()

        for key in energy_info.keys():
            logger.info(
                f"Layer {idx} energy {key}: {energy_info[key].item()}, pruned energy {key}: {pruned_energy_info[key].item()}, diff: {energy_info[key].item() - pruned_energy_info[key].item()}"
            )

        # new_intermediate_size only needs to be set once
        if new_intermediate_size is None:
            new_intermediate_size = new_size

    if config.use_layer_norm_tweaks:
        # Update the last layer normalization weights.
        model.model.norm.weight.data *= (
            1.0
            + (1.0 - torch.abs(last_layer_pruned_sum / last_layer_original_sum))
            / config.layer_norm_scale
        )

    if config.prune_method == PruneMethod.MK_PRUNE_ADJUSTED_2_WITH_GRADIENTS:
        hooks.remove_hooks()
        torch.cuda.empty_cache()

    # Update the model config file.
    model.config.intermediate_size = new_intermediate_size

    if config.training is not None and config.training.train_epochs > 0 and tokens is not None:
        if config.apply_chat_template:
            # Get the generation prompt from the tokenizer.
            tmp_prompt = [{'role': 'user', 'content': 'prompt'}]
            prompt_without_generation_prompt = tokenizer.apply_chat_template(tmp_prompt, add_generation_prompt=False, tokenize=True)
            prompt_with_generation_prompt = tokenizer.apply_chat_template(tmp_prompt, add_generation_prompt=True, tokenize=True)
            generation_prompt = prompt_with_generation_prompt[len(prompt_without_generation_prompt):]
            print(f"Generation Prompt: {generation_prompt}")

        model = train(model, tokens, config.training, device=config.device, generation_prompt=generation_prompt)

        for layer in model.model.layers:
            pruned_ft_energy_info = analyze_layer_energy(layer.mlp)
            for key in energy_info.keys():
                logger.info(
                    f"Layer {idx} energy {key}: {energy_info[key].item()}, pruned energy {key}: {pruned_energy_info[key].item()}, diff: {energy_info[key].item() - pruned_energy_info[key].item()}, pruned fine-tuned energy {key}: {pruned_ft_energy_info[key].item()}, diff: {energy_info[key].item() - pruned_ft_energy_info[key].item()}"
                )

    # return final_model
    return model