"""Local Hugging Face Transformers backend (ChemDFM / Qwen)."""

from __future__ import annotations

from ego_mol_llm.backends.base import GenerationConfig, LLMBackend

# Recommended open chemistry / Qwen checkpoints
MODEL_PRESETS = {
    # Chemistry post-trained on Qwen2.5-14B (needs ~8-10GB VRAM in 4-bit)
    "chemdfm-14b": "OpenDFM/ChemDFM-v2.0-14B",
    # Reasoning chemistry LLM (Qwen2.5-14B lineage)
    "chemdfm-r-14b": "OpenDFM/ChemDFM-R-14B",
    # Smaller chemistry model (LLaMA-3-8B lineage) — better for 8GB GPUs
    "chemdfm-8b": "OpenDFM/ChemDFM-v1.5-8B",
    # General open Qwen instruct models (not chemistry-specialized)
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen3.5-4b": "Qwen/Qwen3.5-4B",
}


class TransformersBackend(LLMBackend):
    name = "transformers"

    def __init__(
        self,
        model_id: str = "chemdfm-8b",
        load_in_4bit: bool = True,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = True,
    ) -> None:
        self.model_id = MODEL_PRESETS.get(model_id, model_id)
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self._model = None
        self._tokenizer = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Local inference requires optional deps: pip install 'ego-mol-llm[local]'"
            ) from e

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=self.trust_remote_code
        )
        kwargs: dict = {
            "device_map": self.device_map,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            except Exception:
                # Windows often lacks bitsandbytes; fall back to fp16/bf16
                kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        else:
            if self.torch_dtype == "auto":
                kwargs["torch_dtype"] = "auto"
            else:
                kwargs["torch_dtype"] = getattr(torch, self.torch_dtype)

        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        self._model.eval()

    def generate(self, messages: list[dict[str, str]], config: GenerationConfig | None = None) -> str:
        self._lazy_load()
        assert self._model is not None and self._tokenizer is not None
        cfg = config or GenerationConfig()

        tok = self._tokenizer
        if hasattr(tok, "apply_chat_template"):
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            # Fallback plain concatenation
            prompt = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages) + "\nASSISTANT:"

        inputs = tok(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        import torch
        from transformers import GenerationConfig as HFGenerationConfig

        gen_cfg = HFGenerationConfig(
            do_sample=cfg.temperature > 0,
            temperature=max(cfg.temperature, 1e-5),
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            max_new_tokens=cfg.max_new_tokens,
            repetition_penalty=cfg.repetition_penalty,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
        )
        with torch.no_grad():
            out = self._model.generate(**inputs, generation_config=gen_cfg)
        text = tok.batch_decode(out, skip_special_tokens=True)[0]
        # Strip prompt prefix when possible
        decoded_in = tok.decode(inputs["input_ids"][0], skip_special_tokens=True)
        if text.startswith(decoded_in):
            text = text[len(decoded_in) :].strip()
        return text
