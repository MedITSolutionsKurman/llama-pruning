from src.func.energy.calculate import calculate_weight_energy
from torch import nn
from typing import Dict
import torch


def analyze_layer_energy(mlp: nn.Module, method: str = 'frobenius') -> Dict[str, torch.Tensor]:
    """

    Calculate energy metrics for MLP layer

    Args:
    - mlp: MLP layer to analyze
    - method: Energy calculation method ('frobenius', 'l1', 'l2', 'relative'). (default: 'frobenius')

    Returns:
    - energies: Energy metrics for the MLP layer
    """
    with torch.no_grad():
        energies = {
            "gate": calculate_weight_energy(mlp.gate_proj.weight, method=method),
            "up": calculate_weight_energy(mlp.up_proj.weight, method=method),
            "down": calculate_weight_energy(mlp.down_proj.weight, method=method),
        }
    return energies
