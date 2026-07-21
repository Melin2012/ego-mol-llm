from __future__ import annotations

from ego_mol_llm.backends.base import LLMBackend
from ego_mol_llm.backends.dry_run import DryRunBackend


def build_backend(
    backend: str = "dry-run",
    model: str = "chemdfm-8b",
    load_in_4bit: bool = True,
    base_url: str | None = None,
    api_key: str | None = None,
) -> LLMBackend:
    """
    backend:
      - dry-run: offline heuristic (CI / demos)
      - transformers: local HF model (ChemDFM / Qwen)
      - openai / ollama / api: OpenAI-compatible HTTP endpoint
    """
    b = backend.lower().strip()
    if b in {"dry-run", "dryrun", "mock"}:
        return DryRunBackend()
    if b in {"transformers", "local", "hf", "chemdfm", "qwen"}:
        from ego_mol_llm.backends.transformers_backend import TransformersBackend

        return TransformersBackend(model_id=model, load_in_4bit=load_in_4bit)
    if b in {"openai", "ollama", "api", "vllm", "openrouter"}:
        from ego_mol_llm.backends.openai_compatible import OpenAICompatibleBackend

        return OpenAICompatibleBackend(model=model, base_url=base_url, api_key=api_key)
    raise ValueError(f"Unknown backend: {backend}")
