import torch
import numpy as np

from typing import Optional, List
from torch.nn import functional as F

from src.config.method_config import MethodConfig
from logging import getLogger

logger = getLogger()

# Maximum Absolute Weight:
# The maximum absolute weight in a neuron might indicate its significance.
# Note: This method is copied from the source given below:
# https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/6-PRUNING/6_3_pruning_structured_llama3.2-1b_OK.ipynb
def get_importance(
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight_t: torch.Tensor,
    weights: Optional[list] = [1.0, 1.0, 1.0],
    parameters: Optional[MethodConfig] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Calculate the importance score for the weight matrix.

    Args:
    - gate_weight: Weight matrix from the gate_proj layer.
    - up_weight: Weight matrix from the up_weight layer
    - weights: Weights for each layer.
    - eps: Small constant for numerical stability.

    Returns:
    - importance: Importance score for the weight matrix.
    """
    gate_weight = torch.clone(gate_weight)
    up_weight = torch.clone(up_weight)
    down_weight_t = torch.clone(down_weight_t)

    gate_max_abs = torch.sqrt(
            torch.max(gate_weight, dim=1).values.pow(2)
            + torch.min(gate_weight, dim=1).values.pow(2)
    )

    up_max_abs = torch.sqrt(
            torch.max(up_weight, dim=1).values.pow(2)
            + torch.min(up_weight, dim=1).values.pow(2)
    )

    down_weight_abs = torch.sqrt(
            torch.max(down_weight_t, dim=1).values.pow(2)
            + torch.min(down_weight_t, dim=1).values.pow(2)
    )

    # Normalize the importance scores
    weights_sum = sum(weights)
    weights = [x / weights_sum for x in weights]

    # Apply weights
    gate_max_abs = gate_max_abs * weights[0]
    up_max_abs = up_max_abs * weights[1]
    down_weight_abs = down_weight_abs * weights[2]

    return torch.sqrt(
        gate_max_abs.pow(2).float()
        + up_max_abs.pow(2).float()
        + down_weight_abs.pow(2).float()
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
    parameters: Optional[MethodConfig] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    compute neuron pair importance scores (Maximum Absolute Weight)

    Args:
    - gate_weight: Weight matrix from the gate_proj layer.
    - up_weight: Weight matrix from the up_weight layer.
    - weights: Weights for each layer.
    - eps: Small constant for numerical stability.

    Returns:
    - importance_scores: Importance scores for each neuron pair.
    """

    gate_importance = get_adjusted_weight_importance(gate_weight)
    up_importance = get_adjusted_weight_importance(up_weight)
    down_importance = get_adjusted_weight_importance(down_weight_t)

    # Make sure all tensors have the same distribution
    gate_importance = gate_importance / (gate_importance.max() + eps)
    up_importance = up_importance / (up_importance.max() + eps)
    down_importance = down_importance / (down_importance.max() + eps)

    # Apply weights
    gate_importance = gate_importance * weights[0]
    up_importance = up_importance * weights[1]
    down_importance = down_importance * weights[2]

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
    parameters: Optional[MethodConfig] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    compute neuron pair importance scores (Maximum Absolute Weight)

    Args:
    - gate_weight: Weight matrix from the gate_proj layer.
    - up_weight: Weight matrix from the up_weight layer.
    - weights: Weights for each layer.
    - eps: Small constant for numerical stability.

    Returns:
    - importance_scores: Importance scores for each neuron pair.
    """

    gate_importance = get_adjusted_weight_importance_2(gate_weight, parameters=parameters)
    up_importance = get_adjusted_weight_importance_2(up_weight, parameters=parameters)
    down_importance = get_adjusted_weight_importance_2(down_weight_t, parameters=parameters)

    # Make sure all tensors have the same distribution
    gate_importance = gate_importance / (gate_importance.max() + eps)
    up_importance = up_importance / (up_importance.max() + eps)
    down_importance = down_importance / (down_importance.max() + eps)

    # Apply weights
    gate_importance = gate_importance * weights[0]
    up_importance = up_importance * weights[1]
    down_importance = down_importance * weights[2]

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
    activations: Optional[np.ndarray] = None,
    gradients: Optional[np.ndarray] = None,
    parameters: Optional[MethodConfig] = None,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    compute neuron pair importance scores (Maximum Absolute Weight)

    Args:
    - gate_weight: Weight matrix from the gate_proj layer.
    - up_weight: Weight matrix from the up_weight layer.
    - weights: Weights for each layer.
    - activations: Optional activation tensor from the layer.
    - gradients: Optional gradient tensor corresponding to the weights.
    - eps: Small constant for numerical stability.

    Returns:
    - importance_scores: Importance scores for each neuron pair.
    """

    gate_importance = get_adjusted_weight_importance_2_with_gradients(gate_weight, activations=activations[0], gradients=gradients[0], dim=1, parameters=parameters, eps=eps)
    up_importance = get_adjusted_weight_importance_2_with_gradients(up_weight, activations=activations[1], gradients=gradients[1], dim=1, parameters=parameters, eps=eps)
    down_importance = get_adjusted_weight_importance_2_with_gradients(down_weight_t, activations=activations[2], gradients=gradients[2], dim=0, parameters=parameters, eps=eps)

    # Make sure all tensors have the same distribution
    gate_importance = gate_importance / (gate_importance.max() + eps)
    up_importance = up_importance / (up_importance.max() + eps)
    down_importance = down_importance / (down_importance.max() + eps)

    # Normalize the importance scores
    weights_sum = sum(weights)
    weights = [x / weights_sum for x in weights]

    # Apply weights
    gate_importance = gate_importance * weights[0]
    up_importance = up_importance * weights[1]
    down_importance = down_importance * weights[2]

    return (torch.sqrt(
        gate_importance.pow(2).float()
        + up_importance.pow(2).float()
        + down_importance.pow(2).float()
    ) + get_importance(gate_weight, up_weight, down_weight_t.t(), weights=weights)) / 2


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


def get_adjusted_weight_importance_2(weight: torch.Tensor, parameters: Optional[MethodConfig] = None) -> torch.Tensor:
    """
    Calculate importance scores focusing on activation patterns and stability.

    Args:
        weight: Weight matrix from the layer (shape [N, M])

    Returns:
        adjusted_importance: Stable importance scores (shape [N])
    """
    # Get absolute values and statistics
    
    abs_weight = torch.sqrt(weight.pow(2))
        
    weight_mean = abs_weight.mean(dim=1)  # Size [N]
    weight_std = abs_weight.std(dim=1)  # Size [N]

    # Compute L2 norm of each neuron's weights
    weight_norm = torch.norm(abs_weight, p=2, dim=1)  # Size [N]

    # Activation stability score
    activation_score = torch.zeros_like(weight_mean)
    for i in range(abs_weight.size(0)):
        # Calculate activation pattern stability
        row = abs_weight[i]
        patterns = (row > weight_mean[i]).to(weight.dtype)
        stability = patterns.mean()
        activation_score[i] = stability

    # Feature diversity through correlation
    # Normalize weights
    weight_normed = abs_weight / (
        torch.norm(abs_weight, p=2, dim=1, keepdim=True) + 1e-8
    )  # Shape [N, M]
    # Compute cosine similarity matrix
    similarity_matrix = torch.matmul(weight_normed, weight_normed.t())  # Shape [N, N]
    # Average diversity score for each neuron
    diversity_score = 1 - similarity_matrix.mean(dim=1)  # Size [N]

    # Make sure all tensors have the same distribution
    activation_score = activation_score / (activation_score.max() + 1e-8)
    weight_std = weight_std / (weight_std.max() + 1e-8)
    diversity_score = diversity_score / (diversity_score.max() + 1e-8)

    if parameters is not None:
        if len(parameters.weights) != 6:
            raise ValueError(f"Invalid number of weights provided {len(parameters.weights)}, expected 4")
        
        # Use custom weights
        w1, w2, w3, w4 = parameters.weights
    else:
        # Default weights
        w1, w2, w3, w4 = 0.25, 0.25, 0.25, 0.25
    # Combine scores with learned weights

    combined_score = (
        w1 * weight_norm
        + w2 * activation_score
        + w3 * diversity_score
        + w4 * (weight_std / (weight_mean + 1e-8))
    )

    combined_score = combined_score.pow(2)
    # Normalize the combined score
    importance = combined_score / (combined_score.max() + 1e-8)

    return importance


def process_quantized_weight(weight: torch.Tensor, chunk_size: int = 1024) -> torch.Tensor:
    """Handle different quantization formats safely"""
    # Check if weight is quantized
    if weight.dtype in [torch.int8, torch.uint8]:
        # Process in chunks to avoid OOM
        num_rows = weight.size(0)
        output_chunks = []
        
        for i in range(0, num_rows, chunk_size):
            end_idx = min(i + chunk_size, num_rows)
            chunk = weight[i:end_idx]
            
            # Convert to float32
            if hasattr(chunk, 'dequantize'):
                chunk = chunk.dequantize()
            chunk = chunk.to(torch.float32)
            
            # Rescale if int8
            if weight.dtype == torch.int8:
                chunk = chunk / 127.0
            elif weight.dtype == torch.uint8:
                chunk = (chunk - 128) / 127.0
                
            output_chunks.append(chunk)
            torch.cuda.empty_cache()
            
        return torch.cat(output_chunks, dim=0)
    return weight

def get_adjusted_weight_importance_2_with_gradients(
    weight: torch.Tensor,
    activations: List[np.ndarray] = None,
    gradients: List[np.ndarray] = None,
    parameters: Optional[MethodConfig] = None,
    eps: float = 1e-10,
    dim: int = 1,
):
    # Calculate basic weight statistics
    weight_abs = torch.abs(weight)
    weight_abs_mean = weight.mean(dim=dim)  # Size [N]
    weight_std = weight.std(dim=dim)  # Size [N]
    weight_norm = torch.norm(weight_abs, p=2, dim=dim)  # Size [N]
    # weight_norm = torch.sqrt(
    #         torch.max(weight, dim=dim).values.pow(2)
    #         + torch.min(weight, dim=dim).values.pow(2)
    #     )

    # Activation stability score
    activation_score = torch.zeros_like(weight_abs_mean)
    for i in range(weight.size(0)):
        row = weight_abs[i]
        patterns = (row > weight_abs_mean[i]).to(weight.dtype)
        stability = patterns.mean()
        activation_score[i] = stability

    activation_importance, grad_importance = process_activations_and_gradients(
        activations, gradients, device=weight.device
    )

    activation_importance = activation_importance.mean(dim=0).to(weight.dtype)  # [intermediate_dim]
    grad_importance = grad_importance.mean(dim=0).to(weight.dtype)  # [intermediate_dim]


    if dim == 0:
        # activation_importance = (weight.t() * activation_importance.unsqueeze(0)).mean(dim=1)
        # grad_importance = (weight.t() * grad_importance.unsqueeze(0)).mean(dim=1)
        activation_importance = torch.zeros_like(weight_abs_mean)
        grad_importance = torch.zeros_like(weight_abs_mean)
    # else:
    #     activation_importance = (weight * activation_importance.unsqueeze(1)).sum(dim=1)
    #     grad_importance = (weight * grad_importance.unsqueeze(1)).sum(dim=1)

    # Feature diversity through correlation
    weight_normed = weight / (
        torch.norm(weight, p=2, dim=dim, keepdim=True) + 1e-8
    )  # Shape [N, M]

    if dim == 0:
        weight_normed = weight_normed.t()

    # weight_normed = torch.abs(weight_normed)

    similarity_matrix = torch.matmul(weight_normed, weight_normed.t())  # Shape [N, N]
    # Average diversity score for each neuron
    diversity_score = 1 - similarity_matrix.mean(dim=dim)  # Size [N]

    # print(weight_norm.size(), activation_score.size(), weight_std.size(), activation_importance.size(), grad_importance.size(), diversity_score.size())
    logger.info(f"Parameters max: Weight_norm={weight_norm.max()}, Activation_score={activation_score.max()}, Weight_std={weight_std.max()}, Activation_importance={activation_importance.max()}, Grad_importance={grad_importance.max()}, Diversity_score={diversity_score.max()}")

    # Make sure all tensors have the same distribution
    weight_norm /= weight_norm.max() + eps
    activation_score /= activation_score.max() + eps
    weight_std /= weight.mean(dim=dim) + eps
    weight_std /= weight_std.max() + eps
    activation_importance /= torch.max(activation_importance) + eps
    grad_importance /= torch.max(grad_importance) + eps
    diversity_score /= diversity_score.max() + eps
    
    if parameters is not None:
        if len(parameters.weights) != 6:
            raise ValueError(f"Invalid number of weights provided {len(parameters.weights)}, expected 6")
        
        # Use custom weights
        weights = parameters.weights
    else:
        # Default weights
        weights = [0.3, 0.2, 0.1, 0.1, 0.1, 0.5]

    # Normalize weights
    sum_weights = sum(weights)
    weights = [x / sum_weights for x in weights]
    
    if dim == 0:
        weights[3] = 0
        weights[4] = 0

    logger.info(f"Weighted importance: weights={weights}")
    # logger.info(f"Parameters max: Weight_norm={weight_norm.max()}, Activation_score={activation_score.max()}, Weight_std={weight_std.max()}, Activation_importance={activation_importance.max()}, Grad_importance={grad_importance.max()}, Diversity_score={diversity_score.max()}")
    
    
    # Combine scores with adjusted weights
    combined_score = (
         weight_norm * weights[0]
        + activation_score * weights[1]
        + weight_std * weights[2]
        + activation_importance * weights[3]
        + grad_importance * weights[4]
        + diversity_score * weights[5]
    )

    # Normalize the combined score
    combined_score = combined_score.pow(2)
    # importance = (combined_score - combined_score.min()) / (combined_score.max() - combined_score.min() + eps)
    importance = combined_score / (combined_score.sum() + eps)
    # importance = torch.sqrt(combined_score.pow(2) / (combined_score.pow(2).sum() + eps))

    return importance


def process_activations_and_gradients(
    activations: List[np.ndarray], gradients: List[np.ndarray], device: str = "cuda"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Process activations and gradients to get statistics per neuron"""
    if activations is None or gradients is None:
        raise ValueError("Both activations and gradients must be provided")

    activations = torch.tensor(activations).to(device)
    gradients = torch.tensor(gradients).to(device)

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

    # Calculate statistics for each neuron
    
    return act_stats, grad_stats

def calculate_final_importance(weights: List[torch.Tensor], parameters: Optional[MethodConfig] = None) -> torch.Tensor:
    """
    Calculate the final importance score for the weight matrix.

    Args:
    - weights: List of weight matrices from the layers.
    - parameters: Method configuration parameters.

    Returns:
    - importance: Final importance score for the weight matrix.
    """
    
    return torch.sqrt(sum([(weight * parameter).pow(2).float() for weight, parameter in zip(weights, parameters)])
    )