import time
import os
from typing import List, Dict
from huggingface_hub import InferenceClient
from .base import BaseAdapter, ModelResponse


class HFAdapter(BaseAdapter):

    def __init__(self, system_prompt: str):
        super().__init__(
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            system_prompt=system_prompt
        )
        self.client = InferenceClient(
            token=os.getenv("HF_TOKEN")
        )

    def chat(self, messages: List[Dict]) -> ModelResponse:
        full_messages = self._build_messages(messages)

        start = time.time()

        response = self.client.chat_completion(
            model=self.model_name,
            messages=full_messages,
            max_tokens=512,
            temperature=0.7,
        )

        latency_ms = (time.time() - start) * 1000

        tokens_input  = response.usage.prompt_tokens
        tokens_output = response.usage.completion_tokens

        return ModelResponse(
            text=response.choices[0].message.content,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            latency_ms=round(latency_ms, 2),
            cost_usd=0.0,
            model_name=self.model_name
        )