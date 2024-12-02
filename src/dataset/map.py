from ast import literal_eval
from typing import List


def map_conversations(examples: List[dict]) -> List[dict]:
    """

    This function maps the conversations to the format required by the model.

    Args:
    - examples: List of examples.

    Returns:
    - List of examples.

    """
    for idx, example in enumerate(examples):
        if "role" not in example:
            example["role"] = example["from"]

            if example["role"] == "human":
                example["role"] = "user"

            if example["role"] == "gpt":
                example["role"] = "assistant"

        if "content" not in example:
            example["content"] = example["value"]

        examples[idx] = {"role": example["role"], "content": example["content"]}

    return examples


def map_iio(examples: List[dict]) -> List[dict]:
    """

    This function maps the input, instruction, and output to the format required by the model.

    Args:
    - examples: Dictionary containing input, instruction, and output.

    Returns:
    - List of examples.

    """

    tmp = []

    if "input" not in examples and "instruction" in examples:
        examples["input"] = examples["instruction"]
        examples["instruction"] = None

    if "instruction" in examples and examples["instruction"] is not None:
        instruction = examples["instruction"]

        if len(instruction.strip()) != 0:
            tmp.append({"role": "system", "content": instruction})

    if "input" in examples and examples["input"] is not None:
        input_text = examples["input"]

        if len(input_text.strip()) == 0:
            if len(tmp) == 0:
                return None

            tmp[0]["role"] = "user"
        else:
            tmp.append({"role": "user", "content": input_text})
    else:
        return None

    if "output" in examples and examples["output"] is not None:
        output_text = examples["output"]

        if len(output_text.strip()) == 0:
            return None

        tmp.append({"role": "assistant", "content": output_text})
    else:
        return None

    return tmp


def map_iio_steps(examples: List[dict]) -> str:
    """

    This function maps the input, instruction, and output to the format required by the model.

    Args:
    - examples: Dictionary containing input, instruction, and output.

    Returns:
    - List of examples.

    """
    output = []
    examples = literal_eval(examples)

    for idx, example in enumerate(examples):
        if idx == 0:
            output.append(
                example["step"]
                .replace("<step>", "<thinking>")
                .replace("<p>", "")
                .replace("</p>", "")
                .replace("</step>", "</thinking>")
            )
            continue

        if len(examples) > 2 and idx > 0 and idx < len(examples) - 1:
            output.append(
                example["step"]
                .replace("<step>", "<reflection>")
                .replace("<p>", "")
                .replace("</p>", "")
                .replace("</step>", "</reflection>")
            )
        else:
            output.append(
                example["step"]
                .replace("<step>", "<output>")
                .replace("<p>", "")
                .replace("</p>", "")
                .replace("</step>", "</output>")
            )

    return "\n".join(output)
