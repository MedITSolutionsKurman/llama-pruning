import torch
from torch import nn
from typing import Optional
from tqdm import tqdm

from src.func.importance import get_importance, get_adjusted_importance, get_adjusted_importance_2, get_adjusted_importance_2_with_gradients
from src.func.normalize import normalize_weight
from src.config.prune_method import PruneMethod

# Methods to prune the model using Pere Martra's method with Mariusz Kurman's modification.

# Prunes a specific percentatge of neurons from the MLP (feed forward layers).
# Note: This method is copied from the source given below:
# https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/6-PRUNING/6_3_pruning_structured_llama3.2-1b_OK.ipynb
def prune_neuron_pairs(
    mlp: nn.Module,
    prune_percent: float,
    prune_method: str = "mk_prune",
    use_normalized_weights: bool = False,
    device: str = "cuda",
    use_full_precision: bool = False,
    target_size: Optional[int] = None,
    gate_up_down_t_weight_weights: Optional[list] = [1.0, 1.0],
    activations: Optional[list] = None,
    gradients: Optional[list] = None,
) -> tuple[nn.Linear, nn.Linear, nn.Linear, int]:
    """
    Reduces the dimensions of the **gate_proj**,**up_proj**, **down_proj**
    layers removing the least important neurons.

    Args:
    - mlp: Layers to prune.
    - prune_percent: Percentage of neurons to prune.
    - prune_method: Method to calculate the importance score.
    - use_normalized_weights: Use normalized weights to calculate the final weights.
    - device: Device to use.
    - use_full_precision: If True, the model will be pruned using full precision.
    - target_size: Target size for the intermediate layer. (prune_percent will be ignored)
    - gate_up_down_weight_weights: Weights for the gate, up and down weights. (default: [1.0, 1.0, 1.0])
    - activations: List of activations to use to calculate the gradients.
    - gradients: List of gradients to use to calculate the gradients.

    Returns:
    - new_gate_proj, new_up_proj, new_down_proj:  New pruned layers.
    - k: New intermediate size.

    """
    # Extract the weights from the MLP layers
    #  these weights are used to calculate each neuron's
    #  importance score in the next step.
    if use_full_precision:
        gate_weight = mlp.gate_proj.weight.data.float()
        up_weight = mlp.up_proj.weight.data.float()
        down_weight = mlp.down_proj.weight.float()
    else:
        gate_weight = mlp.gate_proj.weight.data
        up_weight = mlp.up_proj.weight.data
        down_weight = mlp.down_proj.weight

    original_dtype = mlp.gate_proj.weight.data.dtype

    # Compute importance stores. Neurons with higher importance scores
    # are considered more important and less likely to be pruned.

    if prune_method == PruneMethod.MK_PRUNE:
        importance_scores = get_importance(gate_weight, up_weight, down_weight.t().contiguous(), weights=gate_up_down_t_weight_weights)
    elif prune_method == PruneMethod.MK_PRUNE_ADJUSTED:
        importance_scores = get_adjusted_importance(gate_weight, up_weight, down_weight.t().contiguous(), weights=gate_up_down_t_weight_weights)
    elif prune_method == PruneMethod.MK_PRUNE_ADJUSTED_2:
        importance_scores = get_adjusted_importance_2(gate_weight, up_weight, down_weight.t().contiguous(), weights=gate_up_down_t_weight_weights)
    elif prune_method == PruneMethod.MK_PRUNE_ADJUSTED_2_WITH_GRADIENTS:
        importance_scores = get_adjusted_importance_2_with_gradients(gate_weight, up_weight, down_weight.contiguous(), weights=gate_up_down_t_weight_weights, activations=activations, gradients=gradients)
    else:
        raise ValueError(f"Unknown prune method: {prune_method}")
    
    torch.cuda.empty_cache()

    # Store the original number of neurons in the intermediate layer.
    original_intermediate_size = gate_weight.size(0)

    if target_size is not None:
        # Check if the target size is smaller than the original intermediate size.
        if target_size >= original_intermediate_size:
            raise ValueError(
                f"Target size must be smaller than the original intermediate size: {original_intermediate_size}"
            )

        # Set the number of neurons to keep to the target size.
        k = target_size

    else:
        # Computes the number of neurons to prune.
        num_neuron_pairs_to_prune = min(
            int(prune_percent * original_intermediate_size),
            original_intermediate_size - 1,
        )
        # Calculate the number of neurons to keep. The new intermediate size.
        k = original_intermediate_size - num_neuron_pairs_to_prune

    # Just check that there is no big error calculating k. We can't prune all the neurons.
    if k <= 0:
        raise ValueError(
            f"Invalid number of neuron pairs to keep: {k}. Adjust the prune_percent."
        )

    # Select the neuros to keep, by obtaining the indices to keep.
    _, indices_to_keep = torch.topk(importance_scores, k, largest=True, sorted=True)
    indices_to_keep = indices_to_keep.sort().values

    # create the new layers
    new_gate_proj = nn.Linear(mlp.gate_proj.in_features, k, bias=False).to(device)
    new_up_proj = nn.Linear(mlp.up_proj.in_features, k, bias=False).to(device)
    new_down_proj = nn.Linear(k, mlp.down_proj.out_features, bias=False).to(device)

    # copy weights to the new layers.
    new_gate_proj.weight.data = torch.clone(gate_weight[indices_to_keep, :])
    new_up_proj.weight.data = torch.clone(up_weight[indices_to_keep, :])
    new_down_proj.weight.data = torch.clone(down_weight[:, indices_to_keep])

    if use_normalized_weights:
        new_gate_proj.weight.data = normalize_weight(
            new_gate_proj.weight.data,
            mlp.gate_proj.weight.data[~indices_to_keep, :],
            mlp.gate_proj.weight.data,
        )
        new_up_proj.weight.data = normalize_weight(
            new_up_proj.weight.data,
            mlp.up_proj.weight.data[~indices_to_keep, :],
            mlp.up_proj.weight.data,
        )
        new_down_proj.weight.data = normalize_weight(
            new_down_proj.weight.data,
            mlp.down_proj.weight.data[:, ~indices_to_keep],
            mlp.down_proj.weight.data,
        )

    return (
        new_gate_proj.to(original_dtype),
        new_up_proj.to(original_dtype),
        new_down_proj.to(original_dtype),
        k,
    )
