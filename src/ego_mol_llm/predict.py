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
from ego_mol_llm.validate import (
    HYP_DMZ_MAX,
    HYP_HALF_DMZ_MAX,
    HYP_LIMIT,
    ParsedPrediction,
    canonicalize_smiles,
    parse_model_output,
    validate_smiles_fields,
)


@dataclass
class PredictionResult:
    prediction: ParsedPrediction
    ego: EgoContext
    model_raw: str
    backend: str
    model_id: str | None = None
    messages: list[dict[str, str]] = field(default_factory=list)
    rescue_notes: list[str] = field(default_factory=list)
    # v0.2/v0.3 product path
    library_hits: list[dict[str, Any]] = field(default_factory=list)
    hybrid_candidates: list[dict[str, Any]] = field(default_factory=list)
    method_card: dict[str, Any] | None = None
    sirius_hits: list[dict[str, Any]] = field(default_factory=list)
    product_version: str = "0.3"

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
            "library_hits": self.library_hits,
            "sirius_hits": self.sirius_hits,
            "hybrid_candidates": self.hybrid_candidates,
            "method_card": self.method_card,
            "product_version": self.product_version,
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
    # Same hypothesis set as the prompt MASS-CONSISTENT LIBRARY SMILES block
    hyps = ego.neighbor_structure_hypotheses(
        mass_tol_da=mass_tol_da,
        dmz_max=HYP_DMZ_MAX,
        half_dmz_max=HYP_HALF_DMZ_MAX,
        limit=HYP_LIMIT,
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
        # Match hyp SMILES to nodes via RDKit canonical form (raw GraphML SMILES vary)
        for h in hyps:
            smi = h.get("smiles") or ""
            can_h = canonicalize_smiles(smi) or smi
            best_ms = 0.0
            for ev in ego.neighbors:
                if not ev.node.smiles:
                    continue
                mc = msms_map.get(ev.node.id, 0.0)
                if mc <= 0:
                    continue
                can_n = canonicalize_smiles(ev.node.smiles) or ev.node.smiles
                name_match = bool(
                    h.get("name")
                    and ev.node.name
                    and str(h.get("name")).strip().lower()
                    == str(ev.node.name).strip().lower()
                )
                if can_n == can_h or name_match:
                    best_ms = max(best_ms, mc)
            h["msms_cosine"] = best_ms if best_ms > 0 else None
            if best_ms >= 0.7:
                h["confidence"] = min(0.97, float(h.get("confidence") or 0.5) + 0.08)
                h["rescue_ok"] = (
                    True
                    if best_ms >= 0.75 and h.get("mass_ok") is not False
                    else h.get("rescue_ok")
                )
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
    # v0.2 product path: library + method + hybrid fusion
    use_library_search: bool = True,
    library_index_path: str | Path | None = None,
    library_top_k: int = 8,
    method_card: Any | None = None,
    use_hybrid_ranker: bool = True,
    # SIRIUS / CSI:FingerID (optional; needs CLI + academic login for structure)
    use_sirius: bool = False,
    sirius_bin: str | Path | None = None,
    sirius_work_dir: str | Path | None = None,
    sirius_top_k: int = 8,
    sirius_hits: list | None = None,
    sirius_parse_existing_only: bool = False,
    sirius_timeout_s: float = 600.0,
) -> PredictionResult:
    """
    End-to-end ego prediction (v0.2+ annotation-propagation product path).

    Pipeline:
      1. Build blind ego neighborhood
      2. Attach MS/MS + RT/metadata from MGF
      3. Optional NIST/library reverse search
      4. Optional SIRIUS + CSI:FingerID
      5. LLM proposal over network + library + SIRIUS + experimental context
      6. Hybrid fusion (NIST ∪ SIRIUS ∪ neighbors ∪ model) and/or classic rescue
    """
    from ego_mol_llm.candidates import select_product_annotation
    from ego_mol_llm.library_search import (
        default_nist_paths,
        load_library_index,
    )
    from ego_mol_llm.method_card import MethodCard
    from ego_mol_llm.mgf import build_spectral_context

    ego = build_ego(
        network,
        seed_id=seed_id,
        seed_name_contains=seed_name_contains,
        hide_seed_name=hide_seed_name,
        max_neighbors=max_neighbors,
        include_two_hop=include_two_hop,
    )

    method = method_card if isinstance(method_card, MethodCard) else MethodCard.from_dict(
        method_card if isinstance(method_card, dict) else None
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
            ego.meta["seed_rt"] = ego.spectral.seed_rt
            ego.meta["seed_ion_mode"] = ego.spectral.seed_ion_mode
            # Spectrum IONMODE wins over study-level method-card defaults.
            from ego_mol_llm.method_card import resolve_method_polarity

            method.polarity = resolve_method_polarity(
                seed_ion_mode=ego.spectral.seed_ion_mode,
                method=method,
            )

    # --- NIST / spectral library reverse search (v0.2) ---
    library_hits = []
    lib_hit_dicts: list[dict] = []
    if use_library_search:
        idx_path = library_index_path
        if idx_path is None:
            _mgf, idx_path = default_nist_paths()
        idx_path = Path(idx_path) if idx_path else None
        if idx_path and idx_path.is_file() and ego.spectral and ego.spectral.seed and ego.spectral.seed.peaks:
            try:
                lib = load_library_index(idx_path)
                search_mode = (
                    (ego.spectral.seed_ion_mode if ego.spectral else None)
                    or method.polarity
                )
                if search_mode == "both":
                    search_mode = None  # do not polarity-filter library
                library_hits = lib.search(
                    ego.spectral.seed.peaks,
                    ego.seed_mz or ego.spectral.seed.pepmass,
                    top_k=library_top_k,
                    precursor_tol_da=max(mass_tol_da, 0.02),
                    ion_mode=search_mode,
                )
                lib_hit_dicts = [h.to_dict() for h in library_hits]
                ego.meta["library_hits"] = lib_hit_dicts
                ego.meta["library_index"] = str(idx_path)
                ego.meta["library_n_records"] = lib.n_records
            except Exception as e:
                ego.meta["library_search_error"] = f"{type(e).__name__}: {e}"

    # --- SIRIUS / CSI:FingerID ---
    sirius_hit_objs: list = list(sirius_hits or [])
    sirius_hit_dicts: list[dict] = []
    if use_sirius or sirius_hit_objs:
        from ego_mol_llm.sirius import SiriusHit, identify_spectrum

        if not sirius_hit_objs and ego.spectral and ego.spectral.seed and ego.spectral.seed.peaks:
            sid = str(
                (ego.meta or {}).get("spectrum_id")
                or getattr(ego.seed, "id", None)
                or "query"
            )
            work = Path(sirius_work_dir) if sirius_work_dir else Path("outputs") / "sirius" / sid
            try:
                sirius_hit_objs, smeta = identify_spectrum(
                    compound_id=sid,
                    precursor_mz=float(ego.seed_mz or ego.spectral.seed.pepmass or 0),
                    peaks=list(ego.spectral.seed.peaks),
                    work_dir=work,
                    ion_mode=ego.spectral.seed_ion_mode or method.polarity,
                    sirius_bin=sirius_bin,
                    top_k=sirius_top_k,
                    timeout_s=sirius_timeout_s,
                    parse_existing_only=sirius_parse_existing_only,
                )
                ego.meta["sirius_run"] = smeta
            except Exception as e:
                ego.meta["sirius_status"] = f"error: {type(e).__name__}: {e}"
                sirius_hit_objs = []
        # normalize to SiriusHit + dicts
        normed: list = []
        for h in sirius_hit_objs:
            if isinstance(h, SiriusHit):
                normed.append(h)
            elif isinstance(h, dict):
                fields = {f.name for f in SiriusHit.__dataclass_fields__.values()}  # type: ignore
                normed.append(SiriusHit(**{k: v for k, v in h.items() if k in fields}))
            else:
                normed.append(h)
        sirius_hit_objs = normed
        sirius_hit_dicts = [
            h.to_dict() if hasattr(h, "to_dict") else dict(h) for h in sirius_hit_objs
        ]
        ego.meta["sirius_hits"] = sirius_hit_dicts
        ego.meta["sirius_status"] = (
            f"n_hits={len(sirius_hit_dicts)}" if sirius_hit_dicts else "no_hits"
        )

    ego.meta["method_card"] = method.to_dict()

    be = backend or build_backend("dry-run")
    messages = build_messages(ego, extra_instructions=extra_instructions)
    raw = be.generate(messages, config=gen_config)
    parsed = parse_model_output(raw, precursor_mz=ego.seed_mz, mass_tol_da=mass_tol_da)

    notes: list[str] = []
    hybrid_dicts: list[dict] = []
    seed_rt = None
    neighbor_rts: list = []
    if ego.spectral:
        seed_rt = ego.spectral.seed_rt
        neighbor_rts = list(ego.spectral.neighbor_rt.values())

    if use_hybrid_ranker and use_neighbor_rescue:
        parsed, hnotes, cands = select_product_annotation(
            ego,
            parsed,
            library_hits=library_hits,
            sirius_hits=sirius_hit_objs,
            method=method,
            mass_tol_da=mass_tol_da,
            seed_rt=seed_rt,
            neighbor_rts=neighbor_rts,
        )
        notes.extend(hnotes)
        hybrid_dicts = [c.to_dict() for c in cands]
        # If hybrid deferred, classic neighbor rescue
        if parsed.source in {"model"} and getattr(parsed, "mass_ok", None) is not True:
            parsed, rnotes = refine_with_neighborhood(parsed, ego, mass_tol_da=mass_tol_da)
            notes.extend(rnotes)
        elif not hybrid_dicts:
            parsed, rnotes = refine_with_neighborhood(parsed, ego, mass_tol_da=mass_tol_da)
            notes.extend(rnotes)
    elif use_neighbor_rescue:
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
        library_hits=lib_hit_dicts,
        hybrid_candidates=hybrid_dicts,
        method_card=method.to_dict(),
        sirius_hits=sirius_hit_dicts,
        product_version="0.3",
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
    use_library_search: bool = True,
    library_index_path: str | Path | None = None,
    library_top_k: int = 8,
    method_card: dict | None = None,
    use_hybrid_ranker: bool = True,
    use_sirius: bool = False,
    sirius_bin: str | Path | None = None,
    sirius_work_dir: str | Path | None = None,
    sirius_top_k: int = 8,
    sirius_hits: list | None = None,
    sirius_parse_existing_only: bool = False,
    sirius_timeout_s: float = 600.0,
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
        use_library_search=use_library_search,
        library_index_path=library_index_path,
        library_top_k=library_top_k,
        method_card=method_card,
        use_hybrid_ranker=use_hybrid_ranker,
        mgf_paths=mgf_paths,
        seed_mgf=seed_mgf,
        use_sirius=use_sirius,
        sirius_bin=sirius_bin,
        sirius_work_dir=sirius_work_dir,
        sirius_top_k=sirius_top_k,
        sirius_hits=sirius_hits,
        sirius_parse_existing_only=sirius_parse_existing_only,
        sirius_timeout_s=sirius_timeout_s,
    )
