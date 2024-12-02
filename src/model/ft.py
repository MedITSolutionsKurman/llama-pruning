from transformers import PreTrainedModel, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from src.config.prune_config import PruneConfig
from src.config.prune_ft_config import PruneFTConfig
from src.config.prune_ft_method import PruneFTMethod
from logging import getLogger
from tqdm import tqdm
from typing import Optional, List

import torch
import bitsandbytes as bnb

logger = getLogger()

def train(
    model: PreTrainedModel,
    dataloader: DataLoader,
    config: PruneFTConfig,
    device: str = "cuda",
    generation_prompt: Optional[List[int]] = None,
) -> PreTrainedModel:
    """
    Train the model using the dataloader.

    Args:
    - model: Model to train.
    - dataloader: Dataloader to use, with tokenized data.
    - config: Prune fine-tuning configuration.
    - device: Device to use. (default: "cuda")

    Returns:
    - model: Trained model.
    """
    # Train the model.
    logger.info("Training the pruned model...")

    LR = float(config.train_learning_rate)
    WEIGHT_DECAY = config.train_weight_decay
    ACCUMULATION_STEPS = config.train_accumulation_steps
    NUM_EPOCHS = config.train_epochs
    NUM_STEPS = len(dataloader) * NUM_EPOCHS
    WARMUP_STEPS = int(0.1 * NUM_STEPS) if config.train_warmup_steps >= NUM_STEPS else config.train_warmup_steps

    logger.info(f"Method: {config.method}, LR: {LR}, Accumulation Steps: {ACCUMULATION_STEPS}, Num Steps: {NUM_STEPS}, Warmup Steps: {WARMUP_STEPS}")

    if config.method == PruneFTMethod.MLP_ONLY:
        model.requires_grad_(False)

        for layer in model.model.layers:
            layer.mlp.requires_grad_(True)
    elif config.method == PruneFTMethod.FULL:
        model.requires_grad_(True)
    else:
        logger.warning(f"{config.method} is not supported yet. Falling back to {PruneFTMethod.MLP_ONLY}.")
        model.requires_grad_(False)

        for layer in model.model.layers:
            layer.mlp.requires_grad_(True)

    optimizer = bnb.optim.AdamW8bit(
        [p for p in model.parameters() if p.requires_grad], 
        lr=LR,
        weight_decay=WEIGHT_DECAY,)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=NUM_STEPS
    )

    # Print number of trainable parameters.
    logger.info(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    model.train()

    for epoch in range(NUM_EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        step_bar = tqdm(dataloader, total=NUM_STEPS // NUM_EPOCHS, desc="Training")

        cumulative_loss = 0.0

        for i, batch in enumerate(step_bar):
            input_ids = batch["input_ids"].squeeze(1).to(device)
            attention_mask = batch["attention_mask"].squeeze(1).to(device)
            labels = input_ids.clone()

            if generation_prompt is not None:
                labels = input_ids.clone()
                for idx, token in enumerate(generation_prompt):
                    labels[labels == token] = -100

                # Get the first occurrence of -100 in each sequence
                for idx in range(labels.size(0)):
                    first_occurrence = (labels[idx] == -100).nonzero(as_tuple=True)[0]
                    if len(first_occurrence) > 0:
                        labels[idx, :first_occurrence[0]] = -100
                

            if device == "cuda":
                with torch.amp.autocast(
                    device_type=device, dtype=model.dtype
                ):  # Enable automatic mixed precision
                    outputs = model(
                        input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        return_dict=True,
                    )
                    loss = outputs.loss
            else:
                outputs = model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_dict=True,
                )
                loss = outputs.loss

            loss = loss / ACCUMULATION_STEPS
            loss.backward()

            cumulative_loss += loss.item()

            if (i + 1) % ACCUMULATION_STEPS == 0 or i == len(dataloader) - 1:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                step_bar.set_postfix(loss=cumulative_loss, lr=scheduler.get_last_lr()[0])
                cumulative_loss = 0.0

                torch.cuda.empty_cache()

            if (
                config.train_save_steps > 0
                and (i + 1) % config.train_save_steps == 0
            ):
                logger.info(f"Saving model at step {i + 1}...")
                model.save_pretrained(f'{config.train_checkpoint_dir}/checkpoint-{(i + 1)}')

            step_bar.update()
            input_ids = input_ids.detach().to("cpu")
            attention_mask = attention_mask.detach().to("cpu")

    logger.info("Model training completed.")

    return model
