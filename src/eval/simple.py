from dataclasses import dataclass
from src.model.generate import get_output
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional
from logging import getLogger

logger = getLogger()

import torch


@dataclass
class SimpleEvaluator:
    model: AutoModelForCausalLM
    pruned_model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    device: str = "cuda"
    apply_chat_template: bool = False
    max_new_tokens: int = 50
    quiet: bool = False

    original_logits: Optional[torch.Tensor] = None
    pruned_logits: Optional[torch.Tensor] = None
    original_hidden_states: Optional[torch.Tensor] = None
    pruned_hidden_states: Optional[torch.Tensor] = None

    def generate(
        self,
        prompt: str,
        is_pruned: bool = False,
    ) -> None:
        """
        Evaluate the model using the prompt.

        Args:
        - prompt: Prompt to evaluate the model.
        - is_pruned: If True, the pruned model will be used. (default: False)

        Returns:
        - response: Generated response.
        """

        if is_pruned:
            # Get sample output from the pruned model.
            logger.info("Generating output from the pruned model...")

            _, self.pruned_logits, self.pruned_hidden_states = get_output(
                prompt,
                self.pruned_model,
                self.tokenizer,
                device=self.device,
                apply_chat_template=self.apply_chat_template,
                max_new_tokens=self.max_new_tokens,
                quiet=self.quiet,
            )

        else:
            # Get sample output from the original model.
            logger.info("Generating output from the original model...")

            _, self.original_logits, self.original_hidden_states = get_output(
                prompt,
                self.model,
                self.tokenizer,
                device=self.device,
                apply_chat_template=self.apply_chat_template,
                max_new_tokens=self.max_new_tokens,
                quiet=self.quiet,
            )

    def evaluate(self) -> None:
        """
        Evaluate the model using the generated logits.

        Returns:
        - original_logits: Original logits.
        - pruned_logits: Pruned logits.
        """
        if self.original_logits is None:
            raise ValueError(
                "Original logits are missing. Run generate() method first."
            )

        if self.pruned_logits is None:
            raise ValueError("Pruned logits are missing. Run generate() method first.")

        logger.info("Evaluating the model...")

        # Calculate the difference between the logits of the original and pruned model using KL divergence.
        diff = torch.nn.functional.kl_div(
            self.original_logits.log_softmax(dim=-1),
            self.pruned_logits.softmax(dim=-1),
            reduction="batchmean",
        ).item() / self.original_logits.size(1)

        logger.info(f"\n\nKL divergence: {diff:.2f}\n")

        # Calculate token accuracy
        original_preds = self.original_logits.argmax(dim=-1)
        pruned_preds = self.pruned_logits.argmax(dim=-1)

        original_acc = (original_preds == pruned_preds).float().mean().item()

        logger.info(f"\n\nToken accuracy: {original_acc:.2f}\n")

        original_max = self.original_logits.max().item()
        pruned_max = self.pruned_logits.max().item()

        original_min = self.original_logits.min().item()
        pruned_min = self.pruned_logits.min().item()

        original_mean = self.original_logits.mean().item()
        pruned_mean = self.pruned_logits.mean().item()

        logger.info(
            f"\n\nLogits statistics:\nOriginal: Max: {original_max:.2f} | Min: {original_min:.2f} | Mean: {original_mean:.2f}\nPruned: Max: {pruned_max:.2f} | Min: {pruned_min:.2f} | Mean: {pruned_mean:.2f}\n"
        )

        # Calculate the difference between the hidden states of the original and pruned model using KL divergence.
        diff = self.calculate_hidden_states_kl_div()

        logger.info(f"\n\nKL divergence hidden states: {diff:.2f}\n")

    def calculate_hidden_states_kl_div(self, eps=1e-8):
        """
        Calculate KL divergence between original and pruned hidden states, excluding padding.
        
        Args:
            eps: Small constant for numerical stability
            
        Returns:
            float: Average KL divergence across layers and non-padded tokens
        """
        # Get masks for non-padded tokens (assume zeros are padding)
        original_mask = (self.original_hidden_states.abs().sum(-1) > eps)
        pruned_mask = (self.pruned_hidden_states.abs().sum(-1) > eps)
        
        # Combine masks to only include tokens present in both
        min_size = min(original_mask.size(0), pruned_mask.size(0))
        valid_mask = original_mask[:min_size] & pruned_mask[:min_size]
        
        total_kl_div = 0.0
        total_valid_tokens = 0
        
        # Calculate KL div for each layer
        for layer_idx in range(min_size):
            layer_orig = self.original_hidden_states[layer_idx]
            layer_pruned = self.pruned_hidden_states[layer_idx]
            layer_mask = valid_mask[layer_idx]
            
            if layer_mask.any():
                # Only include non-padded tokens
                orig_valid = layer_orig[layer_mask]
                pruned_valid = layer_pruned[layer_mask]
                
                # Calculate KL divergence
                kl_div = torch.nn.functional.kl_div(
                    orig_valid.log_softmax(dim=-1),
                    pruned_valid.softmax(dim=-1),
                    reduction='sum'
                )
                
                total_kl_div += kl_div.item()
                total_valid_tokens += layer_mask.sum().item()
        
        # Average across all valid tokens and layers
        avg_kl_div = total_kl_div / (total_valid_tokens + eps)
        
        return avg_kl_div