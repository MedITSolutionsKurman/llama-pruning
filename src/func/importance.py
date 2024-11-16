import torch
from typing import Optional, List
from torch.nn import functional as F


# Maximum Absolute Weight:
# The maximum absolute weight in a neuron might indicate its significance.
# Note: This method is copied from the source given below:
# https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/6-PRUNING/6_3_pruning_structured_llama3.2-1b_OK.ipynb
def get_importance(
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight_t: torch.Tensor,
    weights: Optional[list] = [1.0, 1.0, 1.0],
) -> torch.Tensor:
    """
    Calculate the importance score for the weight matrix.

    Args:
    - gate_weight: Weight matrix from the gate_proj layer.
    - up_weight: Weight matrix from the up_weight layer
    - weights: Weights for each layer.

    Returns:
    - importance: Importance score for the weight matrix.
    """
    gate_weight = torch.clone(gate_weight)
    up_weight = torch.clone(up_weight)
    down_weight_t = torch.clone(down_weight_t)

    gate_max_abs = (
        torch.sqrt(
            torch.max(gate_weight, dim=1).values.pow(2)
            + torch.min(gate_weight, dim=1).values.pow(2)
        )
        * weights[0]
    )

    up_max_abs = (
        torch.sqrt(
            torch.max(up_weight, dim=1).values.pow(2)
            + torch.min(up_weight, dim=1).values.pow(2)
        )
        * weights[1]
    )

    down_weight_t = (
        torch.sqrt(
            torch.max(down_weight_t, dim=1).values.pow(2)
            + torch.min(down_weight_t, dim=1).values.pow(2)
        )
        * weights[1]
    )

    return torch.sqrt(
        gate_max_abs.pow(2).float()
        + up_max_abs.pow(2).float()
        + down_weight_t.pow(2).float()
    )


# Adjusted Importance:
# The adjusted importance score is calculated by dividing the sum of the maximum absolute weight,
# the mean of the weights, and the maximum absolute weight of the weights that are less than the mean,
# by the number of weights that are less than the mean.
def get_adjusted_importance(
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight_t: torch.Tensor,
    weights: list = [1.0, 1.0, 1.0],
) -> torch.Tensor:
    """
    compute neuron pair importance scores (Maximum Absolute Weight)

    Args:
    - gate_weight: Weight matrix from the gate_proj layer.
    - up_weight: Weight matrix from the up_weight layer.
    - weights: Weights for each layer.

    Returns:
    - importance_scores: Importance scores for each neuron pair.
    """

    gate_importance = get_adjusted_weight_importance(gate_weight) * weights[0]
    up_importance = get_adjusted_weight_importance(up_weight) * weights[1]
    down_importance = get_adjusted_weight_importance(down_weight_t) * weights[2]

    return torch.sqrt(
        gate_importance.pow(2).float()
        + up_importance.pow(2).float()
        + down_importance.pow(2).float()
    )


def get_adjusted_importance_2(
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight_t: torch.Tensor,
    weights: list = [1.0, 1.0, 1.0],
) -> torch.Tensor:
    """
    compute neuron pair importance scores (Maximum Absolute Weight)

    Args:
    - gate_weight: Weight matrix from the gate_proj layer.
    - up_weight: Weight matrix from the up_weight layer.
    - weights: Weights for each layer.

    Returns:
    - importance_scores: Importance scores for each neuron pair.
    """

    gate_importance = get_adjusted_weight_importance_2(gate_weight) * weights[0]
    up_importance = get_adjusted_weight_importance_2(up_weight) * weights[1]
    down_importance = get_adjusted_weight_importance_2(down_weight_t) * weights[2]

    return torch.sqrt(
        gate_importance.pow(2).float()
        + up_importance.pow(2).float()
        + down_importance.pow(2).float()
    )


def get_adjusted_importance_2_with_gradients(
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight_t: torch.Tensor,
    weights: list = [1.0, 1.0, 1.0],
    activations: Optional[torch.Tensor] = None,
    gradients: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    compute neuron pair importance scores (Maximum Absolute Weight)

    Args:
    - gate_weight: Weight matrix from the gate_proj layer.
    - up_weight: Weight matrix from the up_weight layer.
    - weights: Weights for each layer.
    - activations: Optional activation tensor from the layer.
    - gradients: Optional gradient tensor corresponding to the weights.

    Returns:
    - importance_scores: Importance scores for each neuron pair.
    """

    gate_importance = (
        get_adjusted_weight_importance_2_with_gradients(
            gate_weight, activations=activations[0], gradients=gradients[0]
        )
        * weights[0]
    )
    up_importance = (
        get_adjusted_weight_importance_2_with_gradients(
            up_weight, activations=activations[0], gradients=gradients[0]
        )
        * weights[1]
    )
    down_importance = (
        get_adjusted_weight_importance_2_with_gradients(
            down_weight_t.t(), activations=activations[0], gradients=gradients[0]
        )
        * weights[2]
    )

    return torch.sqrt(
        gate_importance.pow(2).float()
        + up_importance.pow(2).float()
        + down_importance.pow(2).float()
    )


# def get_adjusted_importance_2_with_gradients(
#     gate_weight: torch.Tensor,
#     up_weight: torch.Tensor,
#     down_weight: torch.Tensor,
#     weights: Optional[List[float]] = None,
#     activations: Optional[List[torch.Tensor]] = None,
#     gradients: Optional[List[torch.Tensor]] = None,
#     eps: float = 1e-6
# ) -> torch.Tensor:
#     """
#     Calculate neuron importance scores using weight magnitudes, activation statistics,
#     and gradient information.

#     Args:
#         gate_weight: Weight matrix from gate projection [intermediate_dim, input_dim]
#         up_weight: Weight matrix from up projection [intermediate_dim, input_dim]
#         down_weight: Weight matrix from down projection [output_dim, intermediate_dim]
#         weights: Optional weights for different components [gate_weight, up_weight]
#         activations: List of activation tensors from forward passes
#         gradients: List of gradient tensors from backward passes
#         eps: Small constant for numerical stability

#     Returns:
#         Tensor of importance scores for each neuron [intermediate_dim]
#     """
#     if weights is None:
#         weights = [1.0, 1.0]

#     device = gate_weight.device
#     intermediate_dim = gate_weight.size(0)

#     # Verify dimensions
#     assert gate_weight.size(0) == up_weight.size(0), "Gate and up projection must have same intermediate dimension"
#     assert gate_weight.size(0) == down_weight.size(1), "Down projection must match intermediate dimension"

#     # Calculate weight-based importance (L2 norm)
#     gate_imp = torch.norm(gate_weight, p=2, dim=1)  # [intermediate_dim]
#     up_imp = torch.norm(up_weight, p=2, dim=1)      # [intermediate_dim]
#     down_imp = torch.norm(down_weight, p=2, dim=0)  # [intermediate_dim]

#     # Combine weight-based importance scores
#     weight_imp = (
#         weights[0] * gate_imp * up_imp +
#         weights[1] * down_imp
#     )

#     if activations is not None and gradients is not None:
#         # Process activation and gradient statistics
#         act_stats, grad_stats = process_activations_and_gradients(activations, gradients, device)

#         # Each statistic tensor should have shape [3, intermediate_dim]
#         assert act_stats.size(1) == intermediate_dim, f"Activation statistics dimension mismatch: expected {intermediate_dim}, got {act_stats.size(1)}"
#         assert grad_stats.size(1) == intermediate_dim, f"Gradient statistics dimension mismatch: expected {intermediate_dim}, got {grad_stats.size(1)}"

#         # Compute mean across statistics for each neuron
#         act_imp = act_stats.mean(dim=0)   # [intermediate_dim]
#         grad_imp = grad_stats.mean(dim=0)  # [intermediate_dim]

#         # Normalize importance scores
#         act_imp = act_imp / (torch.max(act_imp) + eps)
#         grad_imp = grad_imp / (torch.max(grad_imp) + eps)
#         weight_imp = weight_imp / (torch.max(weight_imp) + eps)

#         # Combine all importance measures
#         final_imp = (
#             weight_imp *
#             (1 + act_imp) *  # Activation contribution
#             (1 + grad_imp)   # Gradient contribution
#         )
#     else:
#         final_imp = weight_imp

#     return final_imp


# Adjusted Importance:
# The adjusted importance score is calculated by dividing the sum of the maximum absolute weight,
# the mean of the weights, and the maximum absolute weight of the weights that are less than the mean,
# by the number of weights that are less than the mean.
def get_adjusted_weight_importance(weight: torch.Tensor) -> torch.Tensor:
    """
    Calculate the adjusted importance score for the weight matrix.

    Args:
    - weight: Weight matrix from the layer.

    Returns:
    - adjusted_importance: Adjusted importance score for the weight matrix.
    """
    mask = weight > 0.0
    mask = mask.to(weight.dtype)

    positive_max = (weight * mask).max(dim=1).values
    negative_max_abs = (torch.abs(weight * (~mask))).max(dim=1).values

    weight_adjusted_mean = (
        torch.abs(weight).sum(dim=1) - positive_max - negative_max_abs
    ) / (weight.size(1) - 2)

    weight_small_count = (torch.abs(weight) <= weight_adjusted_mean.unsqueeze(1)).sum(
        dim=1
    )

    adjusted_importance = torch.sqrt(
        (
            positive_max.pow(2).float()
            + negative_max_abs.pow(2).float()
            + weight_adjusted_mean.pow(2).float()
        )
        * weight.size(1)
        / weight_small_count
    )

    return adjusted_importance


def get_adjusted_weight_importance(weight: torch.Tensor) -> torch.Tensor:
    """
    Calculate the adjusted importance score for the weight matrix.

    Args:
    - weight: Weight matrix from the layer.

    Returns:
    - adjusted_importance: Adjusted importance score for the weight matrix.
    """
    mask = weight > 0.0
    mask = mask.to(weight.dtype)

    positive_max = (weight * mask).max(dim=1).values
    negative_max_abs = (torch.abs(weight * (~mask))).max(dim=1).values

    weight_adjusted_mean = (
        torch.abs(weight).sum(dim=1) - positive_max - negative_max_abs
    ) / (weight.size(1) - 2)

    weight_small_count = (torch.abs(weight) <= weight_adjusted_mean.unsqueeze(1)).sum(
        dim=1
    )

    adjusted_importance = torch.sqrt(
        (
            positive_max.pow(2).float()
            + negative_max_abs.pow(2).float()
            + weight_adjusted_mean.pow(2).float()
        )
        * weight.size(1)
        / weight_small_count
    )

    return adjusted_importance


def get_adjusted_weight_importance_2(weight: torch.Tensor) -> torch.Tensor:
    """
    Calculate importance scores focusing on activation patterns and stability.

    Args:
        weight: Weight matrix from the layer (shape [N, M])

    Returns:
        adjusted_importance: Stable importance scores (shape [N])
    """
    # Get absolute values and statistics
    abs_weight = torch.abs(weight)
    weight_mean = abs_weight.mean(dim=1)  # Size [N]
    weight_std = abs_weight.std(dim=1)  # Size [N]

    # Compute L2 norm of each neuron's weights
    weight_norm = torch.norm(weight, p=2, dim=1)  # Size [N]

    # Activation stability score
    activation_score = torch.zeros_like(weight_mean)
    for i in range(weight.size(0)):
        # Calculate activation pattern stability
        row = weight[i]
        patterns = (row > weight_mean[i]).to(weight.dtype)
        stability = patterns.mean()
        activation_score[i] = stability

    # Feature diversity through correlation
    # Normalize weights
    weight_normed = weight / (
        torch.norm(weight, p=2, dim=1, keepdim=True) + 1e-8
    )  # Shape [N, M]
    # Compute cosine similarity matrix
    similarity_matrix = torch.matmul(weight_normed, weight_normed.t())  # Shape [N, N]
    # Average diversity score for each neuron
    diversity_score = 1 - similarity_matrix.mean(dim=1)  # Size [N]

    # Combine scores with learned weights
    w1, w2, w3, w4 = 0.25, 0.25, 0.25, 0.25  # Adjust weights as needed
    combined_score = (
        w1 * weight_norm
        + w2 * activation_score
        + w3 * diversity_score
        + w4 * (weight_std / (weight_mean + 1e-8))
    )

    # Normalize the combined score
    importance = combined_score / (combined_score.max() + 1e-8)

    return importance


def get_adjusted_weight_importance_2_with_gradients(
    weight, activations=None, gradients=None, eps=1e-8
):
    # Calculate basic weight statistics
    abs_weight = torch.abs(weight)
    weight_mean = abs_weight.mean(dim=1)  # Size [N]
    weight_std = abs_weight.std(dim=1)  # Size [N]
    weight_norm = torch.norm(weight, p=2, dim=1)  # Size [N]

    # Normalize weight_norm to match other dimensions
    weight_norm = weight_norm / (weight_norm.max() + eps)

    # Activation stability score
    activation_score = torch.zeros_like(weight_mean)
    for i in range(weight.size(0)):
        row = weight[i]
        patterns = (row > weight_mean[i]).to(weight.dtype)
        stability = patterns.mean()
        activation_score[i] = stability

    # # Activation-based importance
    # if activations is not None:
    #     # Ensure activations have correct dimensions
    #     activation_importance = torch.abs(activations)
    #     if activation_importance.dim() > 1:
    #         activation_importance = activation_importance.mean(dim=0)
    #     if activation_importance.size(0) != weight_mean.size(0):
    #         activation_importance = F.interpolate(
    #             activation_importance.unsqueeze(0),
    #             size=weight_mean.size(0),
    #             mode='linear'
    #         ).squeeze().sum(dim=0)
    #     activation_importance = activation_importance / (activation_importance.max() + eps)
    # else:
    #     activation_importance = torch.zeros_like(weight_mean)

    # # Gradient-based importance
    # if gradients is not None:
    #     grad_importance = gradients.abs()
    #     if grad_importance.dim() > 1:
    #         grad_importance = grad_importance.mean(dim=1)
    #     if grad_importance.size(0) != weight_mean.size(0):
    #         grad_importance = F.interpolate(
    #             grad_importance.unsqueeze(0),
    #             size=weight_mean.size(0),
    #             mode='linear'
    #         ).squeeze().sum(dim=0)
    #     grad_importance = grad_importance / (grad_importance.max() + eps)
    # else:
    #     grad_importance = torch.zeros_like(weight_mean)

    activation_importance, grad_importance = process_activations_and_gradients(
        activations, gradients, device=weight.device
    )

    activation_importance = activation_importance.sum(dim=0)  # [intermediate_dim]
    grad_importance = grad_importance.sum(dim=0)  # [intermediate_dim]

    # Normalize importance scores
    activation_importance = activation_importance / (
        torch.max(activation_importance) + eps
    )
    grad_importance = grad_importance / (torch.max(grad_importance) + eps)

    # Combine scores with adjusted weights
    w1, w2, w3, w4, w5 = 0.5, 0.2, 0.3, 0.2, 0.1
    combined_score = (
        w1 * weight_norm
        + w2 * activation_score
        + w3 * (weight_std / (weight_mean + eps))
        + w4 * activation_importance
        + w5 * grad_importance
    )

    # Normalize the combined score
    combined_score = combined_score.pow(2)
    # importance = (combined_score - combined_score.min()) / (combined_score.max() - combined_score.min() + eps)
    importance = combined_score / (combined_score.sum() + eps)
    # importance = torch.sqrt(combined_score.pow(2) / (combined_score.pow(2).sum() + eps))

    return importance


def process_activations_and_gradients(
    activations: List[torch.Tensor], gradients: List[torch.Tensor], device: str = "cuda"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Process activations and gradients to get statistics per neuron"""
    if activations is None or gradients is None:
        raise ValueError("Both activations and gradients must be provided")

    # Stack and reshape tensors to [total_samples, hidden_dim]
    act_tensor = torch.cat(
        [act.reshape(-1, act.size(-1)).to(device) for act in activations]
    )
    grad_tensor = torch.cat(
        [grad.reshape(-1, grad.size(-1)).to(device) for grad in gradients]
    )

    # Calculate statistics for each neuron
    act_stats = torch.stack(
        [act_tensor.mean(dim=0), act_tensor.std(dim=0), act_tensor.abs().mean(dim=0)]
    )  # Shape: [3, hidden_dim]

    grad_stats = torch.stack(
        [grad_tensor.mean(dim=0), grad_tensor.std(dim=0), grad_tensor.abs().mean(dim=0)]
    )  # Shape: [3, hidden_dim]

    return act_stats, grad_stats
