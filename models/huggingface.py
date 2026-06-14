import asyncio
import torch
from typing import Any
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

from .base import BaseLM


class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        super().__init__()
        self.stop_token_ids = stop_token_ids

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for stop_ids in self.stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids) :], stop_ids).all():
                return True
        return False


def build_quantization_config(
    quantization_type: str, compute_type: str
) -> BitsAndBytesConfig:
    if compute_type == "16fp":
        torch_compute_type = torch.float16
    elif compute_type == "32fp":
        torch_compute_type = torch.float32
    else:
        raise ValueError(f"Unknown compute type {compute_type}")

    if quantization_type == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch_compute_type
        )
    elif quantization_type == "8bit":
        return BitsAndBytesConfig(
            load_in_8bit=True, bnb_8bit_compute_dtype=torch_compute_type
        )
    else:
        raise ValueError(
            f"Unknown quantization type {quantization_type}, must be 4bit or 8bit"
        )


class HuggingfaceLM(BaseLM):
    """
    This implements an interface for querying Huggingface models.

    These can be downloaded from the Huggingface hub, or loaded from a local
    directory.
    """

    device: str  # Use for loading to GPU/cpu as appropriate, default cpu

    def __init__(
        self,
        model_name,
        quantization_type: str = "4bit",
        compute_type: str = "16fp",
        device: str = "cpu",
        tokenizer: Any = None,
    ):
        self.device = device
        quantization_config = build_quantization_config(quantization_type, compute_type)
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, quantization_config=quantization_config
            )
        else:
            self.tokenizer = tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = (
                self.tokenizer.unk_token
            )  # For Llama-type tokenizer, which don't have a pad token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            return_dict=True,
            quantization_config=quantization_config,
            device_map="auto",
        )
        self.model.to(self.device)

    async def generate(self, prompt: str, model_args: dict[str, Any] | None = None) -> str:
        # The HF generate call is blocking/CPU-bound, so run it off the event
        # loop to keep `await`-ing callers (the runner) concurrent.
        return await asyncio.to_thread(self._generate, prompt, model_args or {})

    def _generate(self, prompt: str, model_args: dict[str, Any]) -> str:
        tokenizer_outputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        tokenizer_outputs = tokenizer_outputs.to(self.device)

        input_ids = tokenizer_outputs["input_ids"]
        input_ids = input_ids.to(self.device)

        # `stop` is optional; only build stopping criteria when it's supplied.
        stop_criteria = None
        stop = model_args.get("stop")
        if stop:
            old_bos_token = self.tokenizer.add_bos_token
            # Don't add bos token to the stopping tokens
            self.tokenizer.add_bos_token = False

            stop_token_ids = self.tokenizer.encode(stop, return_tensors="pt").to(
                self.device
            )
            stop_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])

            self.tokenizer.add_bos_token = old_bos_token

        outputs = self.model.generate(
            **tokenizer_outputs,
            temperature=model_args["temperature"],
            max_new_tokens=model_args["max_tokens"],
            do_sample=True,
            stopping_criteria=stop_criteria,
        )
        outputs = [
            output[len(input_tokens) :]
            for output, input_tokens in zip(outputs, tokenizer_outputs["input_ids"])
        ]

        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return decoded[0] if decoded else ""
