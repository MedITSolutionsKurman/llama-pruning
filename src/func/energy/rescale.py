import torch

def rescale_weight(weight: torch.Tensor, target_min: float = -1.0, target_max: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    """
    Rescale weight tensor to target min-max range using min-max normalization.
    
    Args:
        weight: Input tensor
        target_min: Target minimum value
        target_max: Target maximum value
        eps: Small constant for numerical stability
    """
    # Get current range
    current_min = weight.min()
    current_max = weight.max()
    current_range = current_max - current_min + eps
    
    # Normalize to [0,1] range first
    normalized = (weight - current_min) / current_range
    
    # Scale to target range
    target_range = target_max - target_min
    rescaled = normalized * target_range + target_min
    
    return rescaled

def normalize(weight: torch.Tensor) -> torch.Tensor:
    """
    Normalize weight tensor to unit energy.
    
    Args:
        weight: Input tensor
    """
    return weight / (torch.norm(weight, p='fro') + 1e-8)