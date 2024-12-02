import torch
from typing import List


# Normalize Weight:
# Normalize the weight matrix by removing the removed part and adding the source mean and standard deviation.
def normalize_weight(
    weight: torch.Tensor, removed_part: torch.Tensor, source: torch.Tensor
) -> torch.Tensor:
    """
    Normalize the weight matrix by removing the removed part and adding the source mean and standard deviation.

    Args:
    - weight: Weight matrix from the layer.
    - removed_part: Removed part of the weight matrix.
    - source: Source tensor to calculate the mean and standard deviation.

    Returns:
    - weight: Normalized weight matrix.
    """

    max_ = weight.max()
    min_ = weight.min()

    weight = (weight - weight.mean()) / weight.std()

    if removed_part.size(1) < weight.size(1):
        removed_part = torch.cat(
            [
                removed_part,
                torch.zeros(
                    removed_part.size(0), weight.size(1) - removed_part.size(1)
                ),
            ],
            dim=1,
        )

    weight = (
        weight + removed_part.view(*weight.size(), -1).sum(dim=-1)
    ) * source.std() + source.mean()

    weight = torch.clip(weight, min_, max_)

    dim = 0 if weight.size(1) == source.size(1) else 1

    norm_ = weight.norm(dim=dim, keepdim=True)

    weight /= norm_
    weight *= torch.max(norm_, (norm_ + source.norm(dim=dim, keepdim=True)) / 2)

    return weight

def normalize_user_weights(weights: List[float]) -> List[float]:
    """
    Normalize the user weights to sum to 1.

    Args:
    - weights: User weights for each layer.

    Returns:
    - weights: Normalized user weights.
    """
    
    weights_sum = sum(weights)
    weights = [x / weights_sum for x in weights]

    return weights
