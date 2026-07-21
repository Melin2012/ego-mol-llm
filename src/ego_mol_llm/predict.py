"""End-to-end prediction API."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ego_mol_llm.backends.base import GenerationConfig, LLMBackend
from ego_mol_llm.backends.factory import build_backend
from ego_mol_llm.ego import EgoContext, build_ego
from ego_mol_llm.graphml import MolecularNetwork, load_graphml
from ego_mol_llm.prompts import build_messages
from ego_mol_llm.validate import ParsedPrediction, parse_model_output, validate_smiles_fields


@dataclass
class PredictionResult:
    prediction: ParsedPrediction
    ego: EgoContext
    model_raw: str
    backend: str
    model_id: str | None = None
    messages: list[dict[str, str]] = field(default_factory=list)
    rescue_notes: list[str] = field(default_factory=list)

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
            "matched_adduct": p.matched_adduct,
            "parse_mode": p.parse_mode,
            "source": p.source,
            "parse_errors": p.parse_errors,
            "rescue_notes": self.rescue_notes,
            "seed_mz": self.ego.seed_mz,
            "seed_id": self.ego.seed.id,
            "true_seed_name": self.ego.meta.get("true_seed_name"),
            "n_neighbors": len(self.ego.neighbors),
            "n_near_isobars": self.ego.meta.get("n_near_isobars"),
            "class_hints": self.ego.class_hints(),
            "backend": self.backend,
            "model_id": self.model_id,
        }


def refine_with_neighborhood(
    pred: ParsedPrediction,
    ego: EgoContext,
    mass_tol_da: float = 0.05,
) -> tuple[ParsedPrediction, list[str]]:
    """
    Post-process like expert ego annotation:
    - Keep model SMILES only if mass-consistent.
    - Else promote best near-mass annotated neighbor SMILES.
    - Always attach neighbor hypotheses as alternatives.
    """
    notes: list[str] = []
    # Include multimer/half-mass library SMILES (e.g. monomer @ 407 when seed is [2M+H]+ @ 813)
    hyps = ego.neighbor_structure_hypotheses(
        mass_tol_da=mass_tol_da,
        dmz_max=2.0,
        half_dmz_max=2.0,
        limit=15,
        scan_all_with_smiles=True,
    )

    # Always surface neighbor hypotheses as alternatives (dedup later)
    existing_alts = list(pred.alternatives or [])
    for h in hyps:
        existing_alts.append(
            {
                "smiles": h["smiles"],
                "confidence": h["confidence"],
                "note": h.get("note"),
                "name": h.get("name"),
                "cosine": h.get("cosine"),
                "delta_mz": h.get("delta_mz"),
            }
        )
    pred.alternatives = existing_alts

    model_ok = bool(pred.smiles_valid and pred.mass_ok is True and pred.canonical_smiles)
    if model_ok:
        pred.source = "model"
        notes.append("Accepted model SMILES (mass-consistent).")
        # If a near-isobar neighbor matches same scaffold mass-wise, bump confidence slightly
        if hyps and pred.confidence is not None:
            top = hyps[0]
            if top.get("delta_mz") is not None and top["delta_mz"] <= 0.05 and top["cosine"] >= 0.85:
                pred.confidence = min(0.95, max(pred.confidence, 0.75))
                notes.append("Boosted confidence: strong near-isobar library support.")
        return pred, notes

    if pred.smiles and pred.mass_ok is False:
        notes.append(
            f"Rejected model SMILES on mass gate "
            f"(error_Da={pred.mass_error_da}, exact_mass={pred.exact_mass})."
        )
        # Keep rejected model as alternative for transparency
        pred.alternatives.insert(
            0,
            {
                "smiles": pred.canonical_smiles or pred.smiles,
                "confidence": pred.confidence or 0.1,
                "note": "rejected: mass-inconsistent model output",
            },
        )

    if hyps:
        best = hyps[0]
        pred.smiles = best["smiles"]
        pred.name = best.get("name") or pred.name
        pred.adduct = best.get("adduct") or pred.adduct
        pred.confidence = float(best.get("confidence") or 0.7)
        # Drop model formula/mass fields that belong to the rejected structure
        pred.formula = None
        pred.exact_mass = best.get("exact_mass")
        pred.mass_error_da = best.get("mass_error_da")
        pred.mass_ok = best.get("mass_ok")
        pred.matched_adduct = best.get("adduct")
        pred.smiles_valid = True
        pred.canonical_smiles = best["smiles"]
        pred.source = "neighbor_rescue"
        pred.rationale = (
            (pred.rationale + " | " if pred.rationale else "")
            + (
                f"Neighborhood rescue: used mass-consistent near-isobar/annotated neighbor "
                f"SMILES (cos={best.get('cosine')}, |Δm/z|={best.get('delta_mz')}, "
                f"name={best.get('name')})."
            )
        )
        adduct_note = best.get("adduct") or ""
        notes.append(
            f"Rescued structure from neighbor SMILES={best['smiles']} "
            f"(cos={best.get('cosine')}, dmz={best.get('delta_mz')}, "
            f"half_dmz={best.get('half_mass_delta')}, adduct={adduct_note})."
        )
        if adduct_note and ("2M" in str(adduct_note) or "3M" in str(adduct_note)):
            notes.append(
                f"Multimer mass match: precursor treated as {adduct_note} of monomer SMILES."
            )
        pred = validate_smiles_fields(pred, ego.seed_mz, mass_tol_da)
        pred.source = "neighbor_rescue"
        # Clear mass-failure errors that referred to the old SMILES
        pred.parse_errors = [
            e
            for e in pred.parse_errors
            if "Mass inconsistent" not in e and "Invalid SMILES" not in e
        ]
        # Preserve multimer adduct from hypothesis if validation didn't set one
        if best.get("adduct") and (
            not pred.adduct or pred.mass_ok is not True
        ):
            pred.adduct = best.get("adduct")
            pred.matched_adduct = best.get("adduct")
        if pred.mass_ok is True:
            pass
        elif pred.mass_ok is None and (
            (best.get("delta_mz") is not None and best["delta_mz"] <= 0.5)
            or (
                best.get("half_mass_delta") is not None
                and best["half_mass_delta"] <= 1.0
            )
            or best.get("mass_ok") is True
        ):
            pred.mass_ok = True
            pred.mass_error_da = best.get("mass_error_da") or best.get("half_mass_delta") or best.get("delta_mz")
            pred.adduct = best.get("adduct") or pred.adduct
            pred.matched_adduct = pred.adduct
            if pred.confidence is not None and pred.confidence < 0.65:
                pred.confidence = 0.75
        return pred, notes

    notes.append("No mass-consistent neighbor SMILES available for rescue.")
    if pred.smiles and pred.mass_ok is False:
        # Leave rejected SMILES but mark low confidence
        pred.source = "model"
    return pred, notes


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
    use_neighbor_rescue: bool = True,
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

    notes: list[str] = []
    if use_neighbor_rescue:
        parsed, notes = refine_with_neighborhood(parsed, ego, mass_tol_da=mass_tol_da)

    model_id = getattr(be, "model_id", None) or getattr(be, "model", None)
    return PredictionResult(
        prediction=parsed,
        ego=ego,
        model_raw=raw,
        backend=getattr(be, "name", type(be).__name__),
        model_id=model_id,
        messages=messages,
        rescue_notes=notes,
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
    use_neighbor_rescue: bool = True,
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
        use_neighbor_rescue=use_neighbor_rescue,
    )
