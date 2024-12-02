import torch
from src.func.energy.rescale import rescale_weight

def align_weight_energies(weight1: torch.Tensor, weight2: torch.Tensor, method: str = 'scale') -> tuple[torch.Tensor, torch.Tensor]:
    """
    Align two weight matrices to have similar energy distributions.
    
    Args:
        weight1, weight2: Input weight tensors
        method: Alignment method ('scale', 'normalize', 'match_distribution')
    
    Returns:
        Tuple of aligned weight tensors
    """
    def get_energy_profile(w):
        return {
            'total': torch.norm(w, p='fro'),
            'row_wise': torch.norm(w, p=2, dim=1),
            'mean': w.abs().mean(),
            'std': w.std()
        }
    
    # Get original energies
    e1 = get_energy_profile(weight1)
    e2 = get_energy_profile(weight2)
    
    if method == 'scale':
        # Scale second matrix to match first's energy
        scale_factor = e1['total'] / (e2['total'] + 1e-8)
        weight2_aligned = weight2 * scale_factor
        return weight1, weight2_aligned
        
    elif method == 'normalize':
        # Normalize both to unit energy
        w1_aligned = weight1 / (e1['total'] + 1e-8)
        w2_aligned = weight2 / (e2['total'] + 1e-8)
        return w1_aligned, w2_aligned
        
    elif method == 'match_distribution':
        # Match mean and std
        w1_std = weight1.std()
        w2_std = weight2.std()
        w1_mean = weight1.mean()
        w2_mean = weight2.mean()
        
        # Standardize and rescale
        w1_aligned = (weight1 - w1_mean) / (w1_std + 1e-8)
        w2_aligned = (weight2 - w2_mean) / (w2_std + 1e-8)
        
        # Match to weight1's distribution
        w2_aligned = w2_aligned * w1_std + w1_mean
        w2_aligned = rescale_weight(w2_aligned, target_min=weight2.min(), target_max=weight2.max())
        return weight1, w2_aligned
    
    raise ValueError(f"Unknown alignment method: {method}")