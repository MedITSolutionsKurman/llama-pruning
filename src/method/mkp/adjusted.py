import torch
from typing import Optional
from src.config.method_config import MethodConfig
from src.func.normalize import normalize_user_weights
from src.func.importance import calculate_final_importance
from logging import getLogger

logger = getLogger()

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

    gate_max_abs = calculate_weight_importance(gate_weight, parameters)
    up_max_abs = calculate_weight_importance(up_weight, parameters)
    down_weight_abs = calculate_weight_importance(down_weight_t, parameters)
    
    weights = normalize_user_weights(weights)

    return calculate_final_importance([gate_max_abs, up_max_abs, down_weight_abs], weights)

def calculate_weight_importance(weight: torch.Tensor, parameters: Optional[MethodConfig] = None) -> torch.Tensor:
    """
    Calculate the importance score for the weight matrix.

    Args:
    - weight: Weight matrix.
    - parameters: MethodConfig object.

    Returns:
    - importance: Importance score for the weight matrix.
    """
    MIN_PARAMETERS_LENGTH = 5

    user_scores = parameters.weights if parameters is not None else [1.0, 1.0, 1.0, 1.0, 1.0]

    if len(user_scores) < MIN_PARAMETERS_LENGTH:
        logger.warning(
            f"User scores length is less than {MIN_PARAMETERS_LENGTH}. Padding with zeros."
        )
        user_scores = user_scores + [0.0] * (MIN_PARAMETERS_LENGTH - len(user_scores))

    if len(user_scores) > MIN_PARAMETERS_LENGTH:
        logger.warning(
            f"User scores length is greater than {MIN_PARAMETERS_LENGTH}. Truncating."
        )
        user_scores = user_scores[:MIN_PARAMETERS_LENGTH]

    # Normalize user scores
    user_scores = normalize_user_weights(user_scores)

    return torch.sqrt(
        torch.max(weight, dim=1).values.pow(2).float() * user_scores[0]
        + torch.mean(weight, dim=1).abs().pow(2).float() * user_scores[1]
        + torch.min(weight, dim=1).values.abs().pow(2).float() * user_scores[2]
        + torch.var(weight, dim=1).abs().pow(2).float() * user_scores[3]
        + torch.std(weight, dim=1).abs().pow(2).float() * user_scores[4]
    )
    