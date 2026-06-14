from typing import Any
from anthropic import AsyncAnthropic
from .base import BaseLM
from ._retry import async_retry


class AnthropicLM(BaseLM):
    """
    This implements an interface for querying Anthropic models.
    """

    client: AsyncAnthropic

    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.client = AsyncAnthropic(api_key=api_key)

    async def generate(self, prompt: str, model_args: dict[str, Any] | None = None) -> str:
        model_args = model_args or {}
        response = await async_retry(
            self.client.messages.create,
            model=self.model_name,
            system="You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=model_args["max_tokens"],
            temperature=model_args["temperature"],
            stream=False,
        )
        return response.content[0].text
