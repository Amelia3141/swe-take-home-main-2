from openai import AsyncOpenAI
from typing import Any
from .base import BaseLM
from ._retry import async_retry, RetryConfig


def is_openai_model(model_name: str) -> bool:
    """
    Check if the model name corresponds to an OpenAI model we support
    """
    return any(
        name == model_name
        for name in [
            "gpt-3.5-turbo",
            "gpt-4o",
            "gpt-4.1-nano"
        ]
    )


class OpenaiLM(BaseLM):
    """
    This implements an interface for querying OpenAI models.
    """

    client: AsyncOpenAI

    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.client = AsyncOpenAI(api_key=api_key)

    async def generate(self, prompt: str, model_args: dict[str, Any] = {}) -> str:
        response = await async_retry(
            self.client.chat.completions.create,
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=model_args["temperature"],
            max_tokens=model_args["max_tokens"],
        )

        return response.choices[0].message.content
