from typing import Any
from .base import BaseLM


class MockLM(BaseLM):
    """
    This implements the `BaseLM` interface but does not perform any actual
    inference. It is intended for testing purposes only.
    """

    model_name: str = "mock-model"

    def generate(self, prompt: str, model_args: dict[str, Any] = {}) -> str:
        return "mocked response"
