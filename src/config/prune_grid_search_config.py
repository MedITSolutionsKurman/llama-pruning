from dataclasses import dataclass, field
from typing import Optional, List

import itertools


# Dataclass to store the prune grid search configuration.
@dataclass
class PruneGridSearchConfig:
    prune_percent_list: List[float] = field(default_factory=lambda: [0.2])
    prune_method_list: List[str] = field(default_factory=lambda: ["mk_prune"])
    use_normalized_weights_list: List[bool] = field(default_factory=lambda: [False])
    use_layer_norm_tweaks_list: List[bool] = field(default_factory=lambda: [False])
    layer_norm_scale_list: List[float] = field(default_factory=lambda: [4.0])
    target_size_list: List[Optional[int]] = field(default_factory=lambda: [None])
    gate_up_down_weight_weights_list: List[List[float]] = field(
        default_factory=lambda: [[1.0, 1.0, 1.0]]
    )

    def __post_init__(self):
        self.generate_param_combinations()

        assert all(
            0.0 <= p <= 1.0 for p in self.prune_percent_list
        ), "Prune percent must be between 0 and 1"
        assert all(
            m in ["mk_prune", "mk_prune_adjusted", "mk", "mka"]
            for m in self.prune_method_list
        ), "Prune method must be 'mk_prune' or 'mk_prune_adjusted' or 'mk' or 'mka'"
        assert all(
            isinstance(u, bool) for u in self.use_normalized_weights_list
        ), "Use normalized weights must be a boolean"
        assert all(
            isinstance(u, bool) for u in self.use_layer_norm_tweaks_list
        ), "Use layer norm tweaks must be a boolean"
        assert all(
            0.0 <= s for s in self.layer_norm_scale_list
        ), "Layer norm scale must be non-negative"
        assert all(
            t is None or t > 0 for t in self.target_size_list
        ), "Target size must be None or a positive integer"
        assert all(
            all(0.0 <= w <= 1.0 for w in weights)
            for weights in self.gate_up_down_weight_weights_list
        ), "Gate up weight weights must be between 0 and 1"

    @classmethod
    def generate_param_combinations(self):
        return itertools.product(
            self.prune_percent_list,
            self.prune_method_list,
            self.use_normalized_weights_list,
            self.use_layer_norm_tweaks_list,
            self.layer_norm_scale_list,
            self.target_size_list,
            self.gate_up_down_weight_weights_list,
        )
