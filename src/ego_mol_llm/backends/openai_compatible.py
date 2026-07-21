"""OpenAI-compatible API backend (vLLM, Ollama, OpenRouter, local servers)."""

from __future__ import annotations

import os

from ego_mol_llm.backends.base import GenerationConfig, LLMBackend


class OpenAICompatibleBackend(LLMBackend):
    name = "openai-compatible"

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "ollama")

    def generate(self, messages: list[dict[str, str]], config: GenerationConfig | None = None) -> str:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("API backend requires: pip install 'ego-mol-llm[api]'") from e

        cfg = config or GenerationConfig()
        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_new_tokens,
        )
        return resp.choices[0].message.content or ""
