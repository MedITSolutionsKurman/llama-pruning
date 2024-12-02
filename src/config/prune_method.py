from enum import Enum
from dataclasses import dataclass


# Enum class for the different pruning methods
@dataclass
class PruneMethod:
    MK_PRUNE = "mk_prune"
    MK_PRUNE_ADJUSTED = "mk_prune_adjusted"
    MK_PRUNE_ADJUSTED_2 = "mk_prune_adjusted_2"
    MK_PRUNE_ADJUSTED_2_WITH_GRADIENTS = "mk_prune_adjusted_2_with_gradients"
