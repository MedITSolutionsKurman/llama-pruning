# Llama-Pruning

![Llama-pruning-image](/assets/llama-pruning.jpg "Llama pruning")

This project provides tools to load and prune large language models using a structured pruning method. The method is based on the work of [Pere Martra](https://github.com/peremartra) with multiple modifications by [Mariusz Kurman](https://github.com/mkurman), including improved adjusted importance calculation, weight normalization, and enhanced layer normalization techniques.

This method is applicable to all models with a Llama-like architecture that includes MLP gating, such as Llama, Phi, Mistral, Qwen, SmolLM, and others.

Blog post: [Model Pruning: A New Approach](https://mkurman.substack.com/p/model-pruning-a-new-approach).

Original work: [Large-Language-Model-Notebooks-Course](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/6-PRUNING/6_3_pruning_structured_llama3.2-1b_OK.ipynb).

Pere Martra's book: [Large Language Models: Apply and Implement Strategies for Large Language Models](https://amzn.to/4eanT1g)

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Arguments](#arguments)
- [AutoML](#automl) *WIP*
- [Example](#example)
- [License](#license)
- [TODO](#todo)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/MedITSolutionsKurman/llama-pruning.git
    cd llama-pruning
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

To load and prune a model, run the `main.py` script with the appropriate arguments:

Provide `config.yaml` configuration:
```sh
python src/main.py --config config.yaml
```

or use CLI:
```sh
python src/main.py --model_name <model_name> --prune_percent <prune_percent> --dtype <dtype> --cache_dir <cache_dir> --device <device> --output <output> --prompt <prompt> --max_new_tokens <max_new_tokens> [--apply_chat_template]
```

You can find example configurations and usage scripts in the `examples` directory.
It is recommended to store your configuration files in the `configs` directory for better organization.

## Arguments

- `--config`: Path to yaml configuration file. If provided, other arguments will be ignored. You can find an example configuration file at config/prune_config.yaml.
- `--model_name`: Name of the model to load (default: `meditsolutions/Llama-3.2-SUN-1B-chat`).
- `--prune_percent`: Percentage of MLP neurons to prune (default: `0.2`).
- `--dtype`: Data type to use (default: `float32`).
- `--cache_dir`: Directory to cache the model (default: `None`).
- `--device`: Device to use (default: `cuda`).
- `--output`: Directory to save the pruned model (default: `results/pruned_model`).
- `--apply_chat_template`: Apply chat template to the model (default: `None`).
- `--prompt`: Prompt to generate the output (default: `What is the capital of France?`).
- `--max_new_tokens`: Maximum number of tokens to generate (default: `50`).
- `--target_size`: Target size for the MLPs intermediate layer. (`prune_percent` will be ignored).
- `--prune_method`: Method to use for pruning. Currently, only "mk_prune" (alias: "mk") and "mk_prune_adjusted" (alias: "mka") are supported. (default: `mk_prune`)
- `--use_normalized_weights`: Use normalized weights to calculate the final weights. (default: `False`)
- `--use_layer_norm_tweaks`: Apply layer normalization changes to account for the impact of pruned neurons. (default: `False`)
- `--layer_norm_scale`: Layer normalization scale. Only used if use_layer_norm_tweaks is True. (default: `4.0`)
- `--gate_up_down_weight_weights`: Weights for the gate, up and down weights. (default: `[1.0, 1.0, 1.0]`)
- `--use_full_precision`: Use full precision for calculations. (default: `False`)
- `--log_dir`: Directory to save the logs. (default: `logs`)
- `--stop_logging`: Stop logging to the file. (default: `False`)
- `--test_only`: Run the test only. Do not save the model (default: `False`).
- `--print_summary`: Print the pruned model summary. (default: `False`)
- `--quiet`: Do not print logs.
- `--eval_dataset`: Hugging Face dataset to evaluate the model. *WIP*
- `--eval_dataset_size`: Size of the evaluation dataset. (default: 20) *WIP*

## AutoML 
*WIP*

The AutoML feature in this project allows for automated grid search over multiple pruning parameters to find the best configuration for pruning the model. The parameters for the grid search can be specified in a YAML configuration file.

The AutoML feature is currently under development and will be available in version **1.1.0**.

### Example YAML Configuration
```yaml
model_name: "meditsolutions/Llama-3.2-SUN-2.5B-chat"
dtype: "float32"
device: "cuda"
output_dir: "pruned_model"
prune_grid:
  prune_percent_list: [0.1, 0.2, 0.3]
  prune_method_list: ["mk_prune", "mk_prune_adjusted"]
  use_normalized_weights_list: [true, false]
  use_layer_norm_tweaks_list: [true, false]
  layer_norm_scale_list: [2.0, 4.0]
  target_size_list: [null, 512]
  gate_up_down_weight_weights_list: [[1.0, 1.0], [0.5, 0.5]]
```

### Running the Grid Search
To run the grid search, use the following command:

```sh
python src/main.py --config examples/grid_search.yaml
```

The script will load the configuration from the YAML file and perform a grid search over the specified parameters to find the best pruned model based on KL divergence loss.

## Example

```sh
python src/main.py --model_name meditsolutions/Llama-3.2-SUN-2.5B-chat --prune_percent 0.2 --dtype float32 --cache_dir ./cache --device cuda --output ./pruned_model --prompt "How to prepare pierogi (famous Polish dish)?" --max_new_tokens 128 --apply_chat_template --test_only
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## TODO

- [ ] AutoML functionality
- [ ] More pruning methods
- [ ] Model merging integration
- [ ] Eval harness integration
- [ ] Support for additional model architectures
- [ ] Improved logging and monitoring
- [ ] Documentation and examples for custom pruning strategies
- [ ] User-friendly WebUI
- [ ] Performance benchmarking and comparison with other pruning techniques
- [ ] Visualization tools for model pruning and evaluation results