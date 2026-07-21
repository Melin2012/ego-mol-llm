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
            "msms_used": bool(
                getattr(self.ego, "spectral", None)
                and self.ego.spectral
                and self.ego.spectral.seed
            ),
            "spectral": (
                self.ego.spectral.to_dict()
                if getattr(self.ego, "spectral", None) is not None
                else None
            ),
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
    - Boost candidates with high MS/MS cosine when spectra are available.
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

    # Attach MS/MS cosine to hyps when spectral context present
    msms_map = {}
    if getattr(ego, "spectral", None) is not None and ego.spectral:
        msms_map = ego.spectral.neighbor_msms_cosine or {}
        if ego.spectral.seed and ego.spectral.seed.peaks:
            notes.append(
                f"MS/MS context: seed peaks={len(ego.spectral.seed.peaks)}, "
                f"neighbor spectra matched={len(msms_map)}, "
                f"diagnostics={list(ego.spectral.seed_diagnostics.keys())}"
            )
        # Boost confidence / rescue_ok when neighbor SMILES comes from high msms_cos node
        id_by_smiles: dict[str, list[str]] = {}
        for ev in ego.neighbors:
            if ev.node.smiles:
                id_by_smiles.setdefault(ev.node.smiles, []).append(ev.node.id)
                # also try after no canonicalize
        for h in hyps:
            smi = h.get("smiles") or ""
            best_ms = 0.0
            for ev in ego.neighbors:
                if not ev.node.smiles:
                    continue
                # loose match: same string or shared id score
                mc = msms_map.get(ev.node.id, 0.0)
                if ev.node.smiles == smi or (
                    h.get("name") and ev.node.name and h.get("name") == ev.node.name
                ):
                    best_ms = max(best_ms, mc)
            h["msms_cosine"] = best_ms if best_ms > 0 else None
            if best_ms >= 0.7:
                h["confidence"] = min(0.97, float(h.get("confidence") or 0.5) + 0.08)
                h["rescue_ok"] = True if best_ms >= 0.75 and h.get("mass_ok") is not False else h.get("rescue_ok")
                h["note"] = (h.get("note") or "") + f" | high MS/MS cos={best_ms:.2f}"
        # Re-sort hyps: prefer high msms
        hyps.sort(
            key=lambda h: (
                0 if h.get("rescue_ok") else 1,
                0 if h.get("mass_ok") is True else 1,
                -(h.get("msms_cosine") or 0),
                -(h.get("cosine") or 0),
                h.get("mass_error_da") if h.get("mass_error_da") is not None else 99,
            )
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
                "msms_cosine": h.get("msms_cosine"),
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

    # Only rescue with quality-gated hypotheses (not weak cosine / radical junk)
    eligible = [h for h in hyps if h.get("rescue_ok")]
    if not eligible:
        notes.append(
            "No high-quality neighbor rescue candidate "
            "(need strong cosine + tight mass; avoided weak isobar / [M]+ false hits)."
        )
        # Drop unvalidated / mass-fail model SMILES rather than show a false structure
        if pred.smiles and pred.mass_ok is not True:
            pred.alternatives.insert(
                0,
                {
                    "smiles": pred.canonical_smiles or pred.smiles,
                    "confidence": pred.confidence or 0.2,
                    "note": "model/heuristic SMILES withheld (mass not validated)",
                    "name": pred.name,
                },
            )
            pred.smiles = None
            pred.canonical_smiles = None
            pred.smiles_valid = None
            pred.name = None
            pred.mass_ok = None
            notes.append("Withheld unvalidated SMILES (prefer abstain over false hit).")
        if not pred.smiles:
            pred.confidence = 0.15
            pred.rationale = (
                (pred.rationale + " | " if pred.rationale else "")
                + "Model empty/invalid or neighborhood too noisy for safe automatic rescue. "
                "Consider [M+H-H2O]+ for phenols/alcohols if formula ~+18 from observed m/z. "
                "Inspect alternatives; do not trust high-confidence labels without mass fit."
            )
            pred.source = "abstain"
        else:
            pred.source = "model"
        return pred, notes

    best = eligible[0]
    pred.smiles = best["smiles"]
    pred.name = best.get("name") or pred.name
    pred.adduct = best.get("adduct") or pred.adduct
    pred.confidence = float(best.get("confidence") or 0.7)
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
            f"Neighborhood rescue: quality-gated library SMILES "
            f"(cos={best.get('cosine')}, |Δm/z|={best.get('delta_mz')}, "
            f"adduct={best.get('adduct')}, name={best.get('name')})."
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
    if adduct_note and "H2O" in str(adduct_note):
        notes.append(f"Water-loss adduct match: {adduct_note}.")
    pred = validate_smiles_fields(pred, ego.seed_mz, mass_tol_da)
    pred.source = "neighbor_rescue"
    pred.parse_errors = [
        e
        for e in pred.parse_errors
        if "Mass inconsistent" not in e and "Invalid SMILES" not in e
    ]
    if best.get("adduct") and (not pred.adduct or pred.mass_ok is not True):
        pred.adduct = best.get("adduct")
        pred.matched_adduct = best.get("adduct")
    if pred.mass_ok is True:
        pass
    elif best.get("mass_ok") is True:
        pred.mass_ok = True
        pred.mass_error_da = best.get("mass_error_da")
        pred.adduct = best.get("adduct") or pred.adduct
        pred.matched_adduct = pred.adduct
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
    mgf_paths: list[str | Path] | None = None,
    seed_mgf: str | Path | None = None,
) -> PredictionResult:
    from ego_mol_llm.mgf import build_spectral_context

    ego = build_ego(
        network,
        seed_id=seed_id,
        seed_name_contains=seed_name_contains,
        hide_seed_name=hide_seed_name,
        max_neighbors=max_neighbors,
        include_two_hop=include_two_hop,
    )

    # Attach MS/MS when MGF files provided
    if mgf_paths or seed_mgf:
        neighbor_ids = [ev.node.id for ev in ego.neighbors]
        ego.spectral = build_spectral_context(
            seed_id=ego.seed.id,
            seed_mz=ego.seed_mz,
            neighbor_ids=neighbor_ids,
            mgf_paths=list(mgf_paths or []),
            seed_mgf=seed_mgf,
        )
        if ego.spectral.seed:
            ego.meta["msms_seed_peaks"] = len(ego.spectral.seed.peaks)
            ego.meta["msms_neighbor_matches"] = len(ego.spectral.neighbor_msms_cosine)

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
    mgf_paths: list[str | Path] | None = None,
    seed_mgf: str | Path | None = None,
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
        mgf_paths=mgf_paths,
        seed_mgf=seed_mgf,
    )
