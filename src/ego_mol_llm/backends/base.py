from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 20
    max_new_tokens: int = 1024
    repetition_penalty: float = 1.05


class LLMBackend(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, messages: list[dict[str, str]], config: GenerationConfig | None = None) -> str:
        """Return model text for chat messages."""
