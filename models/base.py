import abc
from typing import Any


class BaseLM(abc.ABC):
    """Base class defining the interface for all language models."""

    model_name: str

    @abc.abstractmethod
    async def generate(
        self,
        prompt: str,
        model_args: dict[str, Any] = {},
    ) -> str:
        """Generate a response from the model given a prompt and model arguments."""
        pass
