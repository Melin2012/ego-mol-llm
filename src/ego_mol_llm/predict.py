"""End-to-end prediction API."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ego_mol_llm.backends.base import GenerationConfig, LLMBackend
from ego_mol_llm.backends.factory import build_backend
from ego_mol_llm.ego import EgoContext, build_ego
from ego_mol_llm.graphml import MolecularNetwork, load_graphml
from ego_mol_llm.prompts import build_messages
from ego_mol_llm.validate import ParsedPrediction, parse_model_output


@dataclass
class PredictionResult:
    prediction: ParsedPrediction
    ego: EgoContext
    model_raw: str
    backend: str
    model_id: str | None = None
    messages: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        p = self.prediction
        return {
            "smiles": p.canonical_smiles or p.smiles,
            "raw_smiles": p.smiles,
            "smiles_valid": p.smiles_valid,
            "name": p.name,
            "formula": p.formula,
            "adduct": p.adduct,
            "confidence": p.confidence,
            "rationale": p.rationale,
            "alternatives": p.alternatives,
            "exact_mass": p.exact_mass,
            "mass_error_da": p.mass_error_da,
            "mass_ok": p.mass_ok,
            "parse_errors": p.parse_errors,
            "seed_mz": self.ego.seed_mz,
            "seed_id": self.ego.seed.id,
            "true_seed_name": self.ego.meta.get("true_seed_name"),
            "n_neighbors": len(self.ego.neighbors),
            "class_hints": self.ego.class_hints(),
            "backend": self.backend,
            "model_id": self.model_id,
        }


def predict_ego(
    network: MolecularNetwork,
    backend: LLMBackend | None = None,
    seed_id: str | None = None,
    seed_name_contains: str | None = None,
    hide_seed_name: bool = True,
    max_neighbors: int = 25,
    include_two_hop: bool = True,
    gen_config: GenerationConfig | None = None,
    mass_tol_da: float = 0.05,
    extra_instructions: str | None = None,
) -> PredictionResult:
    ego = build_ego(
        network,
        seed_id=seed_id,
        seed_name_contains=seed_name_contains,
        hide_seed_name=hide_seed_name,
        max_neighbors=max_neighbors,
        include_two_hop=include_two_hop,
    )
    be = backend or build_backend("dry-run")
    messages = build_messages(ego, extra_instructions=extra_instructions)
    raw = be.generate(messages, config=gen_config)
    parsed = parse_model_output(raw, precursor_mz=ego.seed_mz, mass_tol_da=mass_tol_da)
    model_id = getattr(be, "model_id", None) or getattr(be, "model", None)
    return PredictionResult(
        prediction=parsed,
        ego=ego,
        model_raw=raw,
        backend=getattr(be, "name", type(be).__name__),
        model_id=model_id,
        messages=messages,
    )


def predict_from_graphml(
    graphml_path: str | Path,
    backend: str = "dry-run",
    model: str = "chemdfm-8b",
    seed_id: str | None = None,
    seed_name_contains: str | None = None,
    hide_seed_name: bool = True,
    max_neighbors: int = 25,
    include_two_hop: bool = True,
    load_in_4bit: bool = True,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.2,
    max_new_tokens: int = 1024,
    mass_tol_da: float = 0.05,
    extra_instructions: str | None = None,
) -> PredictionResult:
    network = load_graphml(graphml_path)
    be = build_backend(
        backend=backend,
        model=model,
        load_in_4bit=load_in_4bit,
        base_url=base_url,
        api_key=api_key,
    )
    return predict_ego(
        network,
        backend=be,
        seed_id=seed_id,
        seed_name_contains=seed_name_contains,
        hide_seed_name=hide_seed_name,
        max_neighbors=max_neighbors,
        include_two_hop=include_two_hop,
        gen_config=GenerationConfig(
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        ),
        mass_tol_da=mass_tol_da,
        extra_instructions=extra_instructions,
    )
