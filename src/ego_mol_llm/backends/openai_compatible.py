"""OpenAI-compatible API backend (vLLM, Ollama, OpenRouter, local servers)."""

from __future__ import annotations

import os
import re

from ego_mol_llm.backends.base import GenerationConfig, LLMBackend


def _model_wants_think_off(model: str) -> bool:
    """Qwen3-style thinking often exhausts max_tokens and leaves content empty."""
    m = (model or "").lower()
    return "qwen3" in m or m.startswith("qwen3:") or "/qwen3" in m


def _extract_message_text(message) -> str:
    """Prefer final answer content; fall back to reasoning/thinking fields."""
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content
    # Some OpenAI-compatible servers (Ollama Qwen3) put chain-of-thought here
    for attr in ("reasoning_content", "reasoning", "thinking"):
        val = getattr(message, attr, None)
        if isinstance(val, str) and val.strip():
            return val
    # dict-shaped message (raw JSON)
    if isinstance(message, dict):
        for key in ("content", "reasoning_content", "reasoning", "thinking"):
            val = message.get(key)
            if isinstance(val, str) and val.strip():
                return val
    # list content parts
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("text"):
                parts.append(str(part["text"]))
            else:
                text = getattr(part, "text", None)
                if text:
                    parts.append(str(text))
        joined = "".join(parts).strip()
        if joined:
            return joined
    return content if isinstance(content, str) else ""


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

        # Ollama Qwen3: disable thinking so the answer lands in message.content
        # within max_tokens (thinking alone often yields empty content).
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "max_tokens": cfg.max_new_tokens,
        }
        extra_body: dict = {}
        think_env = os.environ.get("EGO_MOL_THINK", "").strip().lower()
        if think_env in {"0", "false", "off", "no"}:
            extra_body["think"] = False
        elif think_env in {"1", "true", "on", "yes"}:
            extra_body["think"] = True
        elif _model_wants_think_off(self.model):
            extra_body["think"] = False

        # Ego-network prompts are long; Ollama default num_ctx=4096 often truncates.
        num_ctx_env = os.environ.get("EGO_MOL_NUM_CTX", "").strip()
        if num_ctx_env.isdigit():
            extra_body.setdefault("options", {})["num_ctx"] = int(num_ctx_env)
        elif "ollama" in (self.base_url or "").lower() or self.base_url.rstrip("/").endswith(":11434/v1"):
            extra_body.setdefault("options", {})["num_ctx"] = 16384

        if extra_body:
            kwargs["extra_body"] = extra_body

        resp = client.chat.completions.create(**kwargs)
        text = _extract_message_text(resp.choices[0].message)
        # Strip residual think tags if a server still wraps them
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        return text
