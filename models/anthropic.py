from typing import Any
from anthropic import AsyncAnthropic, HUMAN_PROMPT, AI_PROMPT
from .base import BaseLM
from ._retry import async_retry, RetryConfig


class AnthropicLM(BaseLM):
    """
    This implements an interface for querying Anthropic models.
    """

    client: AsyncAnthropic

    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.client = AsyncAnthropic(api_key=api_key)

    async def generate(self, prompt: str, model_args: dict[str, Any] = {}) -> str:
        response = await async_retry(
            self.client.messages.create,
            model=self.model_name,
            messages=[
                {"role": "user", "content": f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}"}
            ],
            max_tokens=model_args["max_tokens"],
            temperature=model_args["temperature"],
            stream=False,
        )
        return response.content[0].text