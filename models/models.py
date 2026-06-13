from typing import Any
from .base import BaseLM
from .anthropic import AnthropicLM
from .huggingface import HuggingfaceLM
from .mock import MockLM
from .openai import OpenaiLM, is_openai_model


def get_model(model_name: str, init_args: Any) -> BaseLM:
    if model_name == "mock-lm":
        return MockLM()
    elif "claude" in model_name:
        return AnthropicLM(model_name, init_args.anthropic_api_key)
    elif is_openai_model(model_name):
        return OpenaiLM(model_name, init_args.openai_api_key)
    else:
        # Don't recognise the name, assume it must be a huggingface model
        return HuggingfaceLM(
            model_name,
            compute_type=init_args.compute_type,
            device=init_args.device,
            quantization_type=init_args.quantization_type,
        )
