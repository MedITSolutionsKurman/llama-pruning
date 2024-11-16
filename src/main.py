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
from src.eval.simple import SimpleEvaluator
from argparse import ArgumentParser
from logging import getLogger
from datetime import datetime
from src.config.prune_method import PruneMethod

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s | %(name)s | %(levelname)s -> %(message)s"
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

    args = parser.parse_args()

    # Load the configuration.
    if args.config is None:
        # Load the configuration from the arguments. Filter out unexpected arguments.
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
            eval_dataset=args.eval_dataset,
            eval_dataset_size=args.eval_dataset_size,
            quiet=args.quiet,
            log_dir=args.log_dir,
            stop_logging=args.stop_logging,
        )
    else:
        # Load the configuration from the file.
        config = load_config(args.config)

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
    device = "cuda" if config.device and torch.cuda.is_available() else "cpu"

    logger.debug(f"{config}")
    logger.info(f"Using device: {device}")

    # Load the model and tokenizer.
    model, tokenizer = load_model(
        model_name=config.model_name,
        dtype=dtype,
        cache_dir=config.cache_dir,
        device=config.device,
    )

    simple_evaluator = None

    # if config.eval_dataset is None:
    simple_evaluator = SimpleEvaluator(
        model=model,
        pruned_model=None,
        tokenizer=tokenizer,
        device=device,
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
        config.prune_percent,
        prune_method=config.prune_method,
        use_normalized_weights=config.use_normalized_weights,
        use_layer_norm_tweaks=config.use_layer_norm_tweaks,
        layer_norm_scale=config.layer_norm_scale,
        device=config.device,
        target_size=config.target_size,
        use_full_precision=config.use_full_precision,
        gate_up_down_weight_weights=config.gate_up_down_weight_weights,
        deepcopy_model=config.grid_search is not None,
        tokenizer=(
            tokenizer
            if config.prune_method == PruneMethod.MK_PRUNE_ADJUSTED_2_WITH_GRADIENTS
            else None
        ),
        eval_dataset=config.eval_dataset,
        use_chat_template=config.apply_chat_template,
    )

    parameters_after = count_parameters(pruned_model)
    reduction = (parameters_before - parameters_after) / parameters_before * 100

    logger.info(
        f"Model pruned successfully. Parameters before: {parameters_before}, Parameters after: {parameters_after}, Reduction: {reduction:.2f}%."
    )

    # if config.eval_dataset is None:
    simple_evaluator.pruned_model = pruned_model
    simple_evaluator.generate(config.prompt, is_pruned=True)
    simple_evaluator.evaluate()

    if config.print_summary:
        logger.info(f"Model summary after pruning:\n{pruned_model}")

    if config.test_only:
        logger.info("Test completed successfully.")
        exit()

    logger.info(f"Saving the pruned model at {config.output}...")
    pruned_model = pruned_model.to(dtype)
    pruned_model.save_pretrained(config.output)

    logger.info(f"Model saved at {config.output}.")
