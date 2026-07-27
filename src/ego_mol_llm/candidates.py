"""
Hybrid annotation candidates: NIST library ∪ network neighbors ∪ model SMILES.

Fusion ranking is the product-path heart of annotation propagation (v0.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ego_mol_llm.ego import EgoContext
from ego_mol_llm.library_search import LibraryHit
from ego_mol_llm.method_card import MethodCard, rt_compatibility
from ego_mol_llm.validate import (
    canonicalize_smiles,
    check_mass,
    is_multimer_adduct,
)


@dataclass
class AnnotationCandidate:
    smiles: str | None
    name: str | None = None
    source: str = "unknown"  # nist | neighbor | model | hybrid
    fusion_score: float = 0.0
    mass_ok: bool | None = None
    mass_error_da: float | None = None
    adduct: str | None = None
    edge_cosine: float | None = None
    msms_cosine: float | None = None
    lib_match: float | None = None
    rt_compat: float | None = None
    inchikey: str | None = None
    formula: str | None = None
    note: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "smiles": self.smiles,
            "name": self.name,
            "source": self.source,
            "fusion_score": self.fusion_score,
            "mass_ok": self.mass_ok,
            "mass_error_da": self.mass_error_da,
            "adduct": self.adduct,
            "edge_cosine": self.edge_cosine,
            "msms_cosine": self.msms_cosine,
            "lib_match": self.lib_match,
            "rt_compat": self.rt_compat,
            "inchikey": self.inchikey,
            "formula": self.formula,
            "note": self.note,
            "meta": self.meta,
        }


def _mass_fields(
    smiles: str | None,
    seed_mz: float | None,
    mass_tol: float,
    adduct: str | None = None,
) -> tuple[bool | None, float | None, str | None, float | None]:
    if not smiles or seed_mz is None:
        return None, None, None, None
    ok, em, err, matched = check_mass(
        smiles,
        seed_mz,
        adduct,
        tol_da=mass_tol,
        include_multimer=True,
    )
    return ok, err, matched, em


def build_candidates(
    ego: EgoContext,
    *,
    library_hits: list[LibraryHit] | None = None,
    model_smiles: str | None = None,
    model_name: str | None = None,
    model_adduct: str | None = None,
    method: MethodCard | None = None,
    mass_tol_da: float = 0.05,
    seed_rt: float | None = None,
    neighbor_rts: list[float | None] | None = None,
    limit: int = 20,
) -> list[AnnotationCandidate]:
    """Assemble and fusion-rank candidates for prompt + product selection."""
    method = method or MethodCard()
    neighbor_rts = neighbor_rts or []
    cands: list[AnnotationCandidate] = []
    seen: set[str] = set()

    # --- NIST / library ---
    for h in library_hits or []:
        rec = h.record
        smi = canonicalize_smiles(rec.smiles) if rec.smiles else None
        key = smi or (rec.inchikey or rec.name or id(h))
        key_s = str(key)
        if key_s in seen and smi:
            continue
        if smi:
            seen.add(smi)
        ok, err, matched, _em = _mass_fields(
            smi, ego.seed_mz, mass_tol_da, rec.precursor_type
        )
        adduct = matched or rec.precursor_type
        if not method.adduct_prior_ok(adduct):
            continue
        rt_c = rt_compatibility(
            seed_rt, neighbor_rts, card=method, name_hint=rec.name
        )
        # fusion: library match dominates when high
        fusion = (
            0.45 * float(h.match_score)
            + 0.25 * (1.0 if ok is True else 0.15 if ok is None else 0.0)
            + 0.15 * (1.0 - min(1.0, (err or 0.5) / 0.05) if err is not None else 0.2)
            + 0.15 * rt_c
        )
        cands.append(
            AnnotationCandidate(
                smiles=smi,
                name=rec.name,
                source="nist",
                fusion_score=fusion,
                mass_ok=ok,
                mass_error_da=err,
                adduct=adduct,
                lib_match=h.match_score,
                rt_compat=rt_c,
                inchikey=rec.inchikey,
                formula=rec.formula,
                note=f"NIST/lib match score={h.match_score:.3f} Δm/z={h.precursor_error_da:.4f}",
                meta={
                    "spectrum_id": rec.spectrum_id,
                    "instrument": rec.instrument,
                    "source_library": rec.source_library,
                    "lib_precursor_mz": rec.pepmass,
                },
            )
        )

    # --- Network neighbors (mass-consistent hyps) ---
    hyps = ego.neighbor_structure_hypotheses(
        mass_tol_da=mass_tol_da, limit=15, scan_all_with_smiles=True
    )
    for h in hyps:
        smi = h.get("smiles")
        if not smi:
            continue
        can = canonicalize_smiles(smi) or smi
        if can in seen:
            continue
        seen.add(can)
        edge_cos = float(h.get("cosine") or 0.0)
        msms = h.get("msms_cosine")
        if msms is None:
            # try name match later; leave None
            msms_f = 0.0
        else:
            msms_f = float(msms)
        rt_c = rt_compatibility(
            seed_rt, neighbor_rts, card=method, name_hint=h.get("name")
        )
        ok = h.get("mass_ok")
        err = h.get("mass_error_da")
        adduct = h.get("adduct")
        if not method.adduct_prior_ok(adduct):
            continue
        multi_bonus = 0.05 if is_multimer_adduct(adduct) else 0.0
        fusion = (
            0.25 * edge_cos
            + 0.25 * msms_f
            + 0.25 * (1.0 if ok is True else 0.1)
            + 0.15 * rt_c
            + 0.10 * float(h.get("confidence") or 0.5)
            + multi_bonus
        )
        cands.append(
            AnnotationCandidate(
                smiles=can,
                name=h.get("name"),
                source="neighbor",
                fusion_score=fusion,
                mass_ok=ok if isinstance(ok, bool) else bool(ok),
                mass_error_da=err,
                adduct=adduct,
                edge_cosine=edge_cos,
                msms_cosine=msms if msms is not None else None,
                rt_compat=rt_c,
                note=str(h.get("note") or "network neighbor"),
                meta={"rescue_ok": h.get("rescue_ok"), "delta_mz": h.get("delta_mz")},
            )
        )

    # --- Model proposal ---
    if model_smiles:
        can = canonicalize_smiles(model_smiles) or model_smiles
        ok, err, matched, _em = _mass_fields(can, ego.seed_mz, mass_tol_da, model_adduct)
        adduct = matched or model_adduct
        rt_c = rt_compatibility(
            seed_rt, neighbor_rts, card=method, name_hint=model_name
        )
        if method.adduct_prior_ok(adduct):
            fusion = (
                0.20 * 0.7  # base model prior
                + 0.35 * (1.0 if ok is True else 0.0)
                + 0.20 * (1.0 - min(1.0, (err or 0.5) / 0.05) if err is not None else 0.0)
                + 0.15 * rt_c
                + 0.10  # room for later conf
            )
            if can not in seen:
                seen.add(can)
                cands.append(
                    AnnotationCandidate(
                        smiles=can,
                        name=model_name,
                        source="model",
                        fusion_score=fusion,
                        mass_ok=ok,
                        mass_error_da=err,
                        adduct=adduct,
                        rt_compat=rt_c,
                        note="LLM proposal",
                    )
                )

    cands.sort(
        key=lambda c: (
            0 if c.mass_ok is True else 1,
            -(c.fusion_score or 0),
            c.mass_error_da if c.mass_error_da is not None else 99,
        )
    )
    return cands[:limit]


def select_product_annotation(
    ego: EgoContext,
    model_pred: Any,
    *,
    library_hits: list[LibraryHit] | None = None,
    method: MethodCard | None = None,
    mass_tol_da: float = 0.05,
    seed_rt: float | None = None,
    neighbor_rts: list[float | None] | None = None,
    min_fusion_accept: float = 0.45,
) -> tuple[Any, list[str], list[AnnotationCandidate]]:
    """
    Product-path selection: fuse model + NIST + neighbors.

    Returns (updated ParsedPrediction-like, notes, candidates).
    """
    from ego_mol_llm.validate import ParsedPrediction, validate_smiles_fields

    notes: list[str] = []
    model_smi = getattr(model_pred, "canonical_smiles", None) or getattr(
        model_pred, "smiles", None
    )
    cands = build_candidates(
        ego,
        library_hits=library_hits,
        model_smiles=model_smi,
        model_name=getattr(model_pred, "name", None),
        model_adduct=getattr(model_pred, "adduct", None),
        method=method,
        mass_tol_da=mass_tol_da,
        seed_rt=seed_rt,
        neighbor_rts=neighbor_rts,
        limit=25,
    )
    notes.append(f"Hybrid candidates ranked: n={len(cands)}")
    if library_hits:
        notes.append(f"NIST/library hits considered: {len(library_hits)}")

    # Prefer mass-OK candidate with best fusion
    eligible = [c for c in cands if c.mass_ok is True and c.smiles]
    if not eligible:
        eligible = [c for c in cands if c.smiles and c.fusion_score >= min_fusion_accept]

    if not eligible:
        # fall back to classic refine path caller
        notes.append("No hybrid candidate passed gates; defer to classic rescue/abstain.")
        return model_pred, notes, cands

    best = eligible[0]
    if best.fusion_score < min_fusion_accept and best.source != "model":
        notes.append(
            f"Top hybrid score {best.fusion_score:.3f} < {min_fusion_accept}; cautious accept."
        )

    # --- Product selection for annotation propagation ---
    # Network + NIST already enter the *prompt*; post-hoc override should only
    # fire when the model has no mass-consistent SMILES (classic rescue gate).
    # Overriding mass-OK strong models with mid-quality neighbors hurts IK.
    model_ok = getattr(model_pred, "mass_ok", None) is True and bool(model_smi)

    if model_ok:
        model_pred.source = "model"
        notes.append(
            "Kept mass-consistent model SMILES (network/NIST already in prompt; "
            f"top external hybrid={best.source}/{best.fusion_score:.3f} listed as alternative)."
        )
        alts = list(getattr(model_pred, "alternatives", None) or [])
        for c in cands[:8]:
            if c.smiles and c.smiles != model_smi:
                alts.append(
                    {
                        "smiles": c.smiles,
                        "confidence": min(0.95, c.fusion_score),
                        "note": f"hybrid:{c.source} {c.note}",
                        "name": c.name,
                    }
                )
        model_pred.alternatives = alts
        return model_pred, notes, cands

    # Model empty / mass-fail → take best hybrid (NIST / neighbor)
    # Apply best hybrid candidate
    model_pred.smiles = best.smiles
    model_pred.canonical_smiles = best.smiles
    model_pred.name = best.name or getattr(model_pred, "name", None)
    model_pred.adduct = best.adduct or getattr(model_pred, "adduct", None)
    model_pred.formula = best.formula or getattr(model_pred, "formula", None)
    model_pred.mass_ok = best.mass_ok
    model_pred.mass_error_da = best.mass_error_da
    model_pred.matched_adduct = best.adduct
    model_pred.smiles_valid = True if best.smiles else None
    model_pred.confidence = min(0.97, max(0.2, float(best.fusion_score)))
    if best.source == "nist":
        model_pred.source = "nist_library"
    elif best.source == "neighbor":
        model_pred.source = "neighbor_rescue"
    elif best.source == "model":
        model_pred.source = "model"
    else:
        model_pred.source = "hybrid"
    model_pred.rationale = (
        (getattr(model_pred, "rationale", None) or "")
        + f" | Product hybrid pick: source={best.source}, fusion={best.fusion_score:.3f}, "
        f"{best.note}"
    ).strip(" |")
    notes.append(
        f"Selected {best.source} SMILES fusion={best.fusion_score:.3f} "
        f"mass_ok={best.mass_ok} err={best.mass_error_da}"
    )
    alts = []
    for c in cands[:10]:
        if c.smiles and c.smiles != best.smiles:
            alts.append(
                {
                    "smiles": c.smiles,
                    "confidence": min(0.95, c.fusion_score),
                    "note": f"hybrid:{c.source} {c.note}",
                    "name": c.name,
                    "lib_match": c.lib_match,
                    "msms_cosine": c.msms_cosine,
                }
            )
    model_pred.alternatives = alts
    if isinstance(model_pred, ParsedPrediction):
        model_pred = validate_smiles_fields(model_pred, ego.seed_mz, mass_tol_da)
        # restore source after validate
        if best.source == "nist":
            model_pred.source = "nist_library"
        elif best.source == "neighbor":
            model_pred.source = "neighbor_rescue"
        elif best.source == "model":
            model_pred.source = "model"
        else:
            model_pred.source = "hybrid"
    return model_pred, notes, cands
