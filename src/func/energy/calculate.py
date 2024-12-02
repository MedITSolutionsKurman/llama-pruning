import torch


def calculate_weight_energy(
    weight: torch.Tensor, method: str = "frobenius", normalize: bool = True
) -> torch.Tensor:
    """
    Calculate energy of weight matrix using different methods.

    Args:
        weight: Input weight tensor
        method: Energy calculation method ('frobenius', 'l1', 'l2', 'relative')
        normalize: Whether to normalize the energy

    Returns:
        torch.Tensor: Calculated energy score
    """
    # Handle quantized weights
    if hasattr(weight, "dequantize"):
        weight = weight.dequantize()
    weight = weight.to(torch.float32)

    if method == "frobenius":
        # Frobenius norm (square root of sum of squared elements)
        energy = torch.norm(weight, p="fro")

    elif method == "l1":
        # L1 norm (sum of absolute values)
        energy = torch.norm(weight, p=1)

    elif method == "l2":
        # L2 norm (square root of sum of squares)
        energy = torch.norm(weight, p=2)

    elif method == "relative":
        # Relative energy (normalized by number of elements)
        energy = torch.sum(weight.pow(2)) / weight.numel()

    else:
        raise ValueError(f"Unknown energy method: {method}")

    if normalize:
        # Normalize by matrix size
        energy = energy / (weight.size(0) * weight.size(1))

    return energy
