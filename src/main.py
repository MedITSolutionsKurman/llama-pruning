import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import logging

from src.model.load import load_model
from src.func.update import update_model
from src.utils.count import count_parameters
from src.model.generate import get_output
from src.func.load import load_config
from src.config.prune_config import PruneConfig
from src.config.prune_ft_config import PruneFTConfig
from src.config.prune_eval_config import PruneEvalConfig
from src.config.prune_ft_method import PruneFTMethod
from src.eval.simple import SimpleEvaluator
from argparse import ArgumentParser
from logging import getLogger
from datetime import datetime
from src.config.prune_method import PruneMethod
from transformers import set_seed


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s -> %(message)s"
)

logger = getLogger()

if __name__ == "__main__":
    # Parse the arguments.
    parser = ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to yaml configuration file. If provided, other arguments will be ignored. You can find an example configuration file at config/prune_config.yaml.",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="meditsolutions/Llama-3.2-SUN-2.5B-chat",
        help="Name of the model to load.",
    )
    parser.add_argument(
        "--prune_percent",
        type=float,
        default=0.2,
        help="Percentage of MLP neurons to prune.",
    )
    parser.add_argument(
        "--dtype", type=str, default="float32", help="Data type to use."
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None, help="Directory to cache the model."
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use.")
    parser.add_argument(
        "--output",
        type=str,
        default="pruned_model",
        help="Directory to save the pruned model.",
    )
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        default=False,
        help="Apply chat template to the model.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What is the capital of France?",
        help="Prompt to generate the output.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=None,
        help="Target size for the MLPs intermediate layer. (prune_percent will be ignored)",
    )
    parser.add_argument(
        "--test_only",
        action="store_true",
        default=False,
        help="Run the test only. Do not save the model.",
    )
    parser.add_argument(
        "--prune_method",
        type=str,
        default="mk_prune",
        help='Method to use for pruning. Currently, only "mk_prune" (alias: "mk"), "mk_prune_adjusted" (alias: "mka"), "mk_prune_adjusted_2" (alias: "mka2") and "mk_prune_adjusted_2_with_gradients" (alias: "mka2g") are supported. (default: mk_prune)',
    )

    parser.add_argument(
        "--use_normalized_weights",
        action="store_true",
        default=False,
        help="Use normalized weights to calculate the final weights.",
    )

    parser.add_argument(
        "--use_layer_norm_tweaks",
        action="store_true",
        default=False,
        help="Apply layer normalization changes to account for the impact of pruned neurons.",
    )

    parser.add_argument(
        "--layer_norm_scale",
        type=float,
        default=4.0,
        help="Layer normalization scale. Only used if use_layer_norm_tweaks is True. (default: 4.0)",
    )

    parser.add_argument(
        "--print_summary",
        action="store_true",
        default=False,
        help="Print the pruned model summary.",
    )

    parser.add_argument(
        "--gate_up_down_weight_weights",
        type=float,
        nargs="+",
        default=[1.0, 1.0, 1.0],
        help="Weights for the gate, up and down weights. (default: [1.0, 1.0, 1.0])",
    )

    parser.add_argument(
        "--train_method",
        type=str,
        default=PruneFTMethod.MLP_ONLY,
        help="Method to use for training. Currently, 'mlp_only', 'full' and 'layer_wise' are supported. (default: mlp_only)",
    )

    parser.add_argument(
        "--train_dataset",
        type=str,
        default=None,
        help="Hugging Face dataset to train the model.",
    )

    parser.add_argument(
        "--train_dataset_size",
        type=int,
        default=20,
        help="Size of the training dataset. (default: 20)",
    )

    parser.add_argument(
        "--train_max_length",
        type=int,
        default=128,
        help="Maximum length of training sequences.",
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size for training.",
    )

    parser.add_argument(
        "--train_epochs",
        type=int,
        default=0,
        help="Number of epochs for training. (default: 0)",
    )

    parser.add_argument(
        "--train_learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate for training. (default: 1e-5)",
    )

    parser.add_argument(
        "--train_accumulation_steps",
        type=int,
        default=1,
        help="Number of accumulation steps for training. (default: 1)",
    )

    parser.add_argument(
        "--train_warmup_steps",
        type=int,
        default=10,
        help="Number of warmup steps for training. (default: 10)",
    )

    parser.add_argument(
        "--train_weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for training. (default: 0.01)",
    )

    parser.add_argument(
        "--eval_dataset",
        type=str,
        default=None,
        help="Hugging Face dataset to evaluate the model.",
    )

    parser.add_argument(
        "--eval_dataset_size",
        type=int,
        default=20,
        help="Size of the evaluation dataset. (default: 20)",
    )

    parser.add_argument(
        "--eval_max_length",
        type=int,
        default=128,
        help="Maximum length of evaluation sequences.",
    )

    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation.",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Do not print logs.",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory to save the logs. (default: logs)",
    )

    parser.add_argument(
        "--stop_logging",
        action="store_true",
        default=False,
        help="Stop logging to the file.",
    )

    parser.add_argument(
        "--use_full_precision",
        action="store_true",
        default=False,
        help="Use full precision for calculations.",
    )

    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        default=False,
        help="Load the model in 4-bit precision.",
    )

    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        default=False,
        help="Load the model in 8-bit precision.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()

    # Load the configuration.
    if args.config is None:
        config_ft = None
        config_eval = None

        # Load the configuration from the arguments. Filter out unexpected arguments.
        if args.train_num_epochs > 0 or args.train_dataset is not None:
            config_ft = PruneFTConfig(
                method=args.train_method,
                apply_chat_template=args.apply_chat_template,
                cache_dir=args.cache,
                train_dataset=args.train_dataset,
                train_dataset_size=args.train_dataset_size,
                train_max_length=args.train_max_length,
                train_batch_size=args.train_batch_size,
                train_epochs=args.train_epochs,
                train_learning_rate=args.train_learning_rate,
                train_accumulation_steps=args.train_accumulation_steps,
                train_warmup_steps=args.train_warmup_steps,
                train_weight_decay=args.train_weight_decay,
            )

        if args.eval_dataset is not None:
            config_eval = PruneEvalConfig(
                apply_chat_template=args.apply_chat_template,
                cache_dir=args.cache,
                eval_dataset=args.eval_dataset,
                eval_dataset_size=args.eval_dataset_size,
                eval_max_length=args.eval_max_length,
                eval_batch_size=args.eval_batch_size,
            )

        config = PruneConfig(
            model_name=args.model_name,
            dtype=args.dtype,
            device=args.device,
            output=args.output,
            apply_chat_template=args.apply_chat_template,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            prune_percent=args.prune_percent,
            prune_method=args.prune_method,
            use_normalized_weights=args.use_normalized_weights,
            use_layer_norm_tweaks=args.use_layer_norm_tweaks,
            layer_norm_scale=args.layer_norm_scale,
            test_only=args.test_only,
            print_summary=args.print_summary,
            cache_dir=args.cache_dir,
            target_size=args.target_size,
            use_full_precision=args.use_full_precision,
            gate_up_down_weight_weights=args.gate_up_down_weight_weights,
            training=config_ft,
            eval=config_eval,
            quiet=args.quiet,
            log_dir=args.log_dir,
            stop_logging=args.stop_logging,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            seed=args.seed,
        )
    else:
        # Load the configuration from the file.
        config = load_config(args.config)

    set_seed(config.seed)
    os.makedirs(config.log_dir, exist_ok=True)

    logger.handlers = []

    if config.stop_logging is False:
        file_handler = logging.FileHandler(
            os.path.join(
                config.log_dir, f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
            )
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)s | %(levelname)s -> %(message)s")
        )
        logger.addHandler(file_handler)

    # Ensure all logs are saved to file regardless of console logging level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR if config.quiet else logging.INFO)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(name)s | %(levelname)s -> %(message)s")
    )
    logger.addHandler(console_handler)

    dtype = getattr(torch, config.dtype)

    if config.device == "cuda" and not torch.cuda.is_available():
        config.device = "cpu"

    logger.debug(f"{config}")
    logger.info(f"Using device: {config.device}")

    # Load the model and tokenizer.
    model, tokenizer = load_model(
        model_name=config.model_name,
        dtype=dtype,
        cache_dir=config.cache_dir,
        device=config.device,
        load_in_4bit=config.load_in_4bit,
        load_in_8bit=config.load_in_8bit,
    )

    simple_evaluator = None

    # if config.eval_dataset is None:
    simple_evaluator = SimpleEvaluator(
        model=model,
        pruned_model=None,
        tokenizer=tokenizer,
        device=config.device,
        apply_chat_template=config.apply_chat_template,
        max_new_tokens=config.max_new_tokens,
        quiet=config.quiet,
    )

    simple_evaluator.generate(config.prompt)

    parameters_before = count_parameters(model)

    # Prune the model.
    if config.target_size is not None:
        logger.info(
            f"Pruning the model to target size: {config.target_size}, ignoring prune_percent."
        )

    pruned_model = update_model(
        model,
        config,
        tokenizer=tokenizer,
    )

    parameters_after = count_parameters(pruned_model)
    reduction = (parameters_before - parameters_after) / parameters_before * 100

    # if config.eval_dataset is None:
    simple_evaluator.pruned_model = pruned_model
    simple_evaluator.generate(config.prompt, is_pruned=True)
    simple_evaluator.evaluate()

    if config.print_summary:
        logger.info(f"Model summary after pruning:\n{pruned_model}")

    logger.info(
        f"Model pruned successfully. Parameters before: {parameters_before}, Parameters after: {parameters_after}, Reduction: {reduction:.2f}%."
    )

    if config.test_only:
        logger.info("Test completed successfully.")
        exit()

    logger.info(f"Saving the pruned model at {config.output}...")
    pruned_model = pruned_model.to(dtype)
    pruned_model.save_pretrained(config.output)

    logger.info(f"Model saved at {config.output}.")
