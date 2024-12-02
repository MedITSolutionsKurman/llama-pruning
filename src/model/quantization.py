import torch


def check_model_quantization(model):
    """
    Check if a HuggingFace model is quantized and return quantization details.

    Returns:
        dict: Quantization information
    """
    quantization_info = {
        "is_quantized": False,
        "quantization_type": None,
        "bits": None,
        "details": [],
    }

    # Check for bitsandbytes quantization
    if hasattr(model, "is_loaded_in_4bit"):
        if model.is_loaded_in_4bit:
            quantization_info.update(
                {"is_quantized": True, "quantization_type": "bitsandbytes", "bits": 4}
            )
            return quantization_info

    if hasattr(model, "is_loaded_in_8bit"):
        if model.is_loaded_in_8bit:
            quantization_info.update(
                {"is_quantized": True, "quantization_type": "bitsandbytes", "bits": 8}
            )
            return quantization_info

    # Check for QLoRA adaptations
    for name, module in model.named_modules():
        if hasattr(module, "weight"):
            if hasattr(module.weight, "quant_state"):
                quantization_info["details"].append(
                    f"{name}: bitsandbytes quantization"
                )
                quantization_info["is_quantized"] = True

            if hasattr(module, "qweight"):
                quantization_info["details"].append(f"{name}: QLoRA quantization")
                quantization_info["is_quantized"] = True

            # Check weight dtypes
            if hasattr(module, "weight"):
                dtype = module.weight.dtype
                if dtype in [torch.int8, torch.uint8]:
                    quantization_info["details"].append(f"{name}: 8-bit ({dtype})")
                    quantization_info["is_quantized"] = True
                elif dtype in [torch.quint4x2]:
                    quantization_info["details"].append(f"{name}: 4-bit ({dtype})")
                    quantization_info["is_quantized"] = True

    return quantization_info
