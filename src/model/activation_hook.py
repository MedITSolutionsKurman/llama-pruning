from typing import Dict, List
import torch.nn as nn
import torch

class ActivationGradientHooks:
    def __init__(self, max_stored: int = 1000):
        self.activations = {}
        self.gradients = {}
        self.handles = []
        self.max_stored = max_stored
    
    def hook_fn(self, name):
        def hook(module, input, output):
            if name not in self.activations:
                self.activations[name] = []
            if len(self.activations[name]) >= self.max_stored:
                self.activations[name].pop(0)
            
            # Get activations from gate_proj output (should be before activation)
            if isinstance(output, tuple):
                output = output[0]
            
            # Store the full tensor shape
            self.activations[name].append(output.detach())
        return hook
    
    def backward_hook_fn(self, name):
        def hook(module, grad_input, grad_output):
            if name not in self.gradients:
                self.gradients[name] = []
            if len(self.gradients[name]) >= self.max_stored:
                self.gradients[name].pop(0)
            
            # Get gradient of gate_proj output
            grad = grad_output[0] if isinstance(grad_output, tuple) else grad_output
            
            # Store the full tensor shape
            self.gradients[name].append(grad.detach())
        return hook
    
    def register_hooks(self, model):
        """Register hooks for each MLP layer in the model."""
        for i, layer in enumerate(model.model.layers):
            name = f"layer_{i}"
            # Register hook on gate_proj specifically
            handle_forward = layer.mlp.gate_proj.register_forward_hook(self.hook_fn(name))
            handle_backward = layer.mlp.gate_proj.register_full_backward_hook(self.backward_hook_fn(name))
            self.handles.append(handle_forward)
            self.handles.append(handle_backward)
    
    def get_layer_statistics(self, layer_name: str) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Get activation and gradient statistics for a specific layer."""
        return (
            self.activations.get(layer_name, []),
            self.gradients.get(layer_name, [])
        )
    
    def remove_hooks(self):
        """Remove all hooks and clear stored activations/gradients."""
        """Before removing the hooks, move them to CPU to avoid CUDA errors."""

        # Move activations to CPU
        for layer in self.activations:
            self.activations[layer] = [act.cpu() for act in self.activations[layer]]

        # Move gradients to CPU
        for layer in self.gradients:
            self.gradients[layer] = [grad.cpu() for grad in self.gradients[layer]]

        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.activations.clear()
        self.gradients.clear()