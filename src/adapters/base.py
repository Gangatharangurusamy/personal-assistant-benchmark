from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class ModelResponse:
    text: str
    tokens_input: int
    tokens_output: int
    latency_ms: float
    cost_usd: float
    model_name: str


class BaseAdapter(ABC):

    def __init__(self, model_name: str, system_prompt: str):
        self.model_name = model_name
        self.system_prompt = system_prompt

    @abstractmethod
    def chat(self, messages: List[Dict]) -> ModelResponse:
        """
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "how are you?"}
        ]
        """
        pass

    def _build_messages(self, messages: List[Dict]) -> List[Dict]:
        """Prepends system prompt to every call"""
        return [
            {"role": "system", "content": self.system_prompt}
        ] + messages