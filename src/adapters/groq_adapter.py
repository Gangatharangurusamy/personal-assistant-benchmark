import time
import os
from typing import List, Dict
from groq import Groq
from .base import BaseAdapter, ModelResponse

INPUT_COST_PER_TOKEN  = 0.59 / 1_000_000
OUTPUT_COST_PER_TOKEN = 0.79 / 1_000_000


class GroqAdapter(BaseAdapter):

    def __init__(self, system_prompt: str):
        super().__init__(
            model_name="llama-3.3-70b-versatile",
            system_prompt=system_prompt
        )
        # groq 1.x — no api_key param, reads env directly
        self.client = Groq()

    def chat(self, messages: List[Dict]) -> ModelResponse:
        full_messages = self._build_messages(messages)

        start = time.time()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=full_messages,
            max_tokens=512,
            temperature=0.7,
        )

        latency_ms = (time.time() - start) * 1000

        tokens_input  = response.usage.prompt_tokens
        tokens_output = response.usage.completion_tokens

        cost = (
            tokens_input  * INPUT_COST_PER_TOKEN +
            tokens_output * OUTPUT_COST_PER_TOKEN
        )

        return ModelResponse(
            text=response.choices[0].message.content,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            latency_ms=round(latency_ms, 2),
            cost_usd=round(cost, 8),
            model_name=self.model_name
        )