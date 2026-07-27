"""Prompt templates for chemistry LLMs (ChemDFM / Qwen family)."""

from __future__ import annotations

from typing import Any

from ego_mol_llm.ego import EgoContext, NeighborEvidence
from ego_mol_llm.validate import monomer_mass_targets


SYSTEM_PROMPT = """You are an expert mass spectrometry and natural-product chemist.
You assign structure (SMILES) for an UNKNOWN precursor using **ego-network annotation
propagation**: spectral neighbors, optional spectral library hits (e.g. NIST),
optional SIRIUS/CSI:FingerID ranks, retention time / method context, and raw MS/MS.

This is network-assisted annotation (not pure de novo without evidence).

Critical rules (follow in order):
1. The query identity is hidden. Do NOT invent peak lists you were not given.
2. MASS FIRST: any proposed SMILES must fit the query precursor m/z under a common
   adduct, INCLUDING MULTIMERS:
   - monomers: [M-H]-, [M+H]+, [M+Na]+, [M+NH4]+, [M+H-H2O]+, …
   - dimers: [2M+H]+, [2M-H]-, [2M+Na]+, …
   - trimers: [3M+H]+, [3M-H]-, …
   If the precursor is high (e.g. >600) and many neighbors sit near half-mass,
   strongly consider [2M+H]+ / [2M-H]- of a monomer library structure.
3. When MS/MS peaks are provided:
   - Use diagnostic fragments (e.g. Phe immonium 120, Phe-related 166, BA water losses)
     to choose among mass-consistent candidates.
   - Prefer neighbors with BOTH high network cosine AND high MS/MS cosine to the query.
   - Strong spectral library (NIST) hits are independent evidence — weigh match score + mass.
   - Use the offline MS/MS EXPLANATION block (labeled losses, diagnostics, shared peaks
     vs neighbors). Prefer structures consistent with those fragment clues.
   - When CFM-ID FRAGMENT EXPLANATION is present: use transferred fragment SMILES as
     substructure evidence for the unknown seed; they come from annotated neighbors.
   - SIRIUS/CSI:FingerID ranks are OPTIONAL (slow; only when provided) — never required.
4. NEAR-ISOBAR PRIORITY: neighbors with |Δm/z| ≤ 0.5 Da and high cosine are strongest
   for monomer self-matches — but reject annotations whose formula cannot fit m/z.
5. MULTIMER PRIORITY: self-consistent 2M/3M relationships only when structure mass fits
   (not m/z coincidence alone).
6. Use retention time and method (RP-C18, ESI+/−) as soft chemistry filters when present.
7. Prefer same-study / same-method neighbor annotations when metadata is available.
8. Distant high-cosine edges can share fragmentation only — require mass consistency.
9. Output MUST include a JSON block with:
{
  "smiles": "<canonical SMILES of the *neutral monomer* structure>",
  "iupac_or_common_name": "<string or null>",
  "formula": "<Hill formula or null>",
  "adduct": "<e.g. [M-H]- or [2M+H]+ for the observed precursor>",
  "confidence": <float 0-1>,
  "rationale": "<2-5 sentences citing m/z, network, library, SIRIUS/CSI, RT/method, MS/MS>",
  "alternatives": [{"smiles": "...", "confidence": 0.0, "note": "..."}]
}
"""


def _fmt_neighbor(
    i: int,
    ev: NeighborEvidence,
    seed_mz: float | None,
    msms_cos: float | None = None,
    spectral: Any = None,
) -> str:
    n = ev.node
    name = n.name if n.is_annotated else "NO_MATCH"
    smiles = n.smiles or ""
    mz = f"{n.mz:.4f}" if n.mz is not None else "?"
    cos = f"{ev.cosine:.3f}"
    dmz = ev.resolved_delta_mz(seed_mz)
    hdmz = ev.half_mass_delta(seed_mz)
    dmz_s = f"{dmz:.4f}" if dmz is not None else "?"
    hdmz_s = f"{hdmz:.4f}" if hdmz is not None else "?"
    score = f"{ev.evidence_score(seed_mz):.3f}"
    smi_part = f" | SMILES={smiles}" if smiles else ""
    msms_part = f" | msms_cos={msms_cos:.3f}" if msms_cos is not None else ""
    tag = ""
    if dmz is not None and dmz <= 0.05:
        tag = " | ★ NEAR-ISOBAR"
    elif dmz is not None and dmz <= 0.5:
        tag = " | near-isobar"
    elif hdmz is not None and hdmz <= 0.10:
        tag = " | ★ MULTIMER-CONSISTENT (self-consistent 2M/3M)"
    elif hdmz is not None and hdmz <= 0.5:
        tag = " | multimer residual <0.5 Da"
    if msms_cos is not None and msms_cos >= 0.7:
        tag += " | ★ MS/MS-SIMILAR"
    meta_part = ""
    if spectral is not None:
        meta = (getattr(spectral, "neighbor_meta", None) or {}).get(str(ev.node.id)) or {}
        bits = []
        if meta.get("rt") is not None:
            bits.append(f"RT={float(meta['rt']):.1f}s")
        if meta.get("msv_lib"):
            bits.append(f"MSV={meta['msv_lib']}")
        if meta.get("clustersize"):
            bits.append(f"n={meta['clustersize']}")
        if bits:
            meta_part = " | " + " ".join(bits)
    return (
        f"{i:02d}. m/z={mz} | edge_cos={cos}{msms_part} | |Δm/z|={dmz_s} | |Δhalf|={hdmz_s} "
        f"| score={score} | name={name}{smi_part}{tag}{meta_part}"
    )


def _format_spectral_section(ctx: EgoContext) -> list[str]:
    spec = getattr(ctx, "spectral", None)
    if spec is None or spec.seed is None or not spec.seed.peaks:
        return [
            "",
            "=== QUERY MS/MS ===",
            "(no MGF / MS/MS provided — network-only mode)",
        ]
    from ego_mol_llm.mgf import format_peaks_for_prompt

    lines = [
        "",
        "=== QUERY MS/MS (use these peaks; do not invent others) ===",
        f"  precursor m/z (PEPMASS) = {spec.seed.pepmass or ctx.seed_mz}",
        f"  n_peaks = {len(spec.seed.peaks)}",
        f"  top peaks (mz, rel%): {format_peaks_for_prompt(spec.seed.peaks, 15)}",
    ]
    if getattr(spec, "seed_rt", None) is not None:
        lines.append(f"  retention time (s) = {spec.seed_rt:.3f}")
    if getattr(spec, "seed_ion_mode", None):
        lines.append(f"  ion_mode = {spec.seed_ion_mode}")
    meta = getattr(spec, "seed_meta", None) or {}
    if meta.get("MSV_LIB") or meta.get("FILENAME"):
        lines.append(
            f"  origin: MSV_LIB={meta.get('MSV_LIB', '?')} FILENAME={meta.get('FILENAME', '?')}"
        )
    if meta.get("CLUSTERSIZE"):
        lines.append(f"  clustersize = {meta.get('CLUSTERSIZE')}")
    # Prefer rich offline explanation when available (msms_explain)
    exp = getattr(spec, "explanation", None)
    if exp:
        from ego_mol_llm.msms_explain import (
            DiagnosticHit,
            LabeledLoss,
            MsmsExplanation,
            NeighborPeakDiff,
            format_explanation_for_prompt,
        )

        # Enrich neighbor names from ego when available
        name_by_id = {}
        for ev in getattr(ctx, "neighbors", None) or []:
            name_by_id[str(ev.node.id)] = ev.node.name
        diffs = []
        for d in exp.get("neighbor_diffs") or []:
            if isinstance(d, dict):
                nid = str(d.get("neighbor_id") or "")
                if not d.get("neighbor_name") and nid in name_by_id:
                    d = dict(d)
                    d["neighbor_name"] = name_by_id[nid]
                diffs.append(NeighborPeakDiff(**{
                    k: d.get(k) for k in NeighborPeakDiff.__dataclass_fields__ if k in d
                }))
        losses = [
            LabeledLoss(**{k: x.get(k) for k in LabeledLoss.__dataclass_fields__ if k in x})
            for x in (exp.get("labeled_losses") or [])
            if isinstance(x, dict)
        ]
        diags = [
            DiagnosticHit(**{k: x.get(k) for k in DiagnosticHit.__dataclass_fields__ if k in x})
            for x in (exp.get("diagnostics") or [])
            if isinstance(x, dict)
        ]
        obj = MsmsExplanation(
            labeled_losses=losses,
            diagnostics=diags,
            neighbor_diffs=diffs,
            chemistry_hints=list(exp.get("chemistry_hints") or []),
        )
        lines.extend(format_explanation_for_prompt(obj))
        # Optional CFM-ID–first network fragment block (precomputed into meta)
        cfm = (ctx.meta or {}).get("cfm_explain")
        if cfm:
            from ego_mol_llm.cfm_network_explain import format_cfm_explanation_for_prompt

            lines.extend(format_cfm_explanation_for_prompt(cfm))
        return lines

    # Legacy fallback
    if spec.seed_diagnostics:
        base = max(spec.seed_diagnostics.values()) or 1.0
        diag = ", ".join(
            f"{k}={v:.0f} ({100*v/base:.0f}% of strongest diag)"
            for k, v in sorted(spec.seed_diagnostics.items(), key=lambda x: -x[1])
        )
        lines.append(f"  diagnostic ions: {diag}")
        lines.append(
            "  hint: Phe_immonium~120 + Phe_related~166 → phenylalanine moiety; "
            "Leu/Ile~86; Val~72; Tyr~136; strong H2O losses → alcohols/phenols/BA OH"
        )
    if spec.seed_losses:
        loss_s = "; ".join(
            f"{mz:.2f} (−{loss:.2f}{(' ' + lab) if lab else ''})"
            for mz, _i, loss, lab in spec.seed_losses[:10]
        )
        lines.append(f"  notable fragments vs precursor: {loss_s}")
    return lines


def _format_method_and_library(ctx: EgoContext) -> list[str]:
    lines: list[str] = []
    method = (ctx.meta or {}).get("method_card") or {}
    seed_ion = None
    spec = getattr(ctx, "spectral", None)
    if spec is not None:
        seed_ion = getattr(spec, "seed_ion_mode", None)
    if seed_ion is None:
        seed_ion = (ctx.meta or {}).get("seed_ion_mode")
    if method:
        lines += [
            "",
            "=== EXPERIMENTAL METHOD (soft chemistry prior) ===",
            f"  chromatography = {method.get('chromatography')}",
            f"  polarity = {method.get('polarity')}",
            f"  ionization = {method.get('ionization')}",
            f"  gradient = {method.get('gradient')}",
        ]
        if method.get("study_id"):
            lines.append(f"  study_id = {method.get('study_id')}")
        if seed_ion:
            lines.append(f"  seed_spectrum_ion_mode = {seed_ion} (from MGF; authoritative)")
            mpol = (method.get("polarity") or "").lower()
            if mpol in {"positive", "negative"} and seed_ion in {"positive", "negative"} and mpol != seed_ion:
                lines.append(
                    "  ⚠ POLARITY CONFLICT: method card ≠ seed MGF IONMODE — "
                    "trust seed_spectrum_ion_mode and matching-mode library adducts."
                )
            lines.append(
                "  Prefer adducts consistent with seed ion mode "
                f"({'[M+H]+ / [M+Na]+ / …' if seed_ion == 'positive' else '[M-H]- / …'})."
            )

    hits = (ctx.meta or {}).get("library_hits") or []
    lines += [
        "",
        "=== SPECTRAL LIBRARY SEARCH (e.g. NIST; independent of network names) ===",
    ]
    if seed_ion in {"positive", "negative"}:
        lines.append(
            f"  (hits filtered to {seed_ion}-mode library spectra / adducts when metadata allows)"
        )
    if not hits:
        lines.append(
            "(no library index / no hits — network + MS/MS only; "
            "build index with scripts/build_library_index.py)"
        )
        return lines
    for h in hits[:10]:
        smi = h.get("smiles") or ""
        smi_p = f" SMILES={smi}" if smi else ""
        lines.append(
            f"  #{h.get('rank')}: score={h.get('match_score'):.3f} "
            f"lib_mz={h.get('pepmass')} Δm/z={h.get('precursor_error_da'):.4f} "
            f"adduct={h.get('precursor_type')} name={h.get('name')}"
            f"{smi_p} inchikey={h.get('inchikey')}"
        )
    lines.append(
        "  Use high-scoring library hits as strong structure candidates when mass-consistent "
        "and adduct polarity matches the seed ion mode."
    )

    # SIRIUS / CSI:FingerID (precomputed into ego.meta by pack refresh or predict)
    sirius_hits = (ctx.meta or {}).get("sirius_hits") or []
    if sirius_hits:
        from ego_mol_llm.sirius import SiriusHit, hits_to_prompt_block

        objs: list[Any] = []
        fields = set(SiriusHit.__dataclass_fields__)  # type: ignore[attr-defined]
        for h in sirius_hits:
            if isinstance(h, SiriusHit):
                objs.append(h)
            elif isinstance(h, dict):
                objs.append(SiriusHit(**{k: v for k, v in h.items() if k in fields}))
            else:
                objs.append(h)
        lines.extend(hits_to_prompt_block(objs, max_n=8))
    elif (ctx.meta or {}).get("sirius_status"):
        lines += [
            "",
            "=== SIRIUS / CSI:FingerID ===",
            f"  status: {(ctx.meta or {}).get('sirius_status')}",
        ]
    return lines


def build_user_prompt(ctx: EgoContext, extra_instructions: str | None = None) -> str:
    mz = f"{ctx.seed_mz:.6f}" if ctx.seed_mz is not None else "unknown"
    from ego_mol_llm.validate import (
        DEFAULT_HALF_DMZ_MAX,
        HYP_DMZ_MAX,
        HYP_HALF_DMZ_MAX,
        HYP_LIMIT,
    )

    ranked = ctx.top_neighbors
    isobars = ctx.near_isobars(0.5)
    halfs = ctx.half_mass_neighbors(DEFAULT_HALF_DMZ_MAX)
    # Same hypothesis list as refine_with_neighborhood (rescue)
    hyps = ctx.neighbor_structure_hypotheses(
        mass_tol_da=0.05,
        dmz_max=HYP_DMZ_MAX,
        half_dmz_max=HYP_HALF_DMZ_MAX,
        limit=HYP_LIMIT,
        scan_all_with_smiles=True,
    )
    msms_map = {}
    spectral = getattr(ctx, "spectral", None)
    if spectral is not None:
        msms_map = spectral.neighbor_msms_cosine or {}

    lines = [
        "TASK: Assign structure of the UNKNOWN center via ego-network annotation propagation.",
        "",
        "QUERY (unknown structure):",
        f"  precursor m/z = {mz}",
        f"  node_id = {ctx.seed.id}",
        f"  degree = {ctx.meta.get('degree', len(ctx.neighbors))}",
        f"  near-isobar neighbors (|Δm/z|≤0.5) = {len(isobars)}",
        f"  multimer-consistent neighbors (|Δmultimer|≤{DEFAULT_HALF_DMZ_MAX}) = "
        f"{ctx.meta.get('n_half_mass_neighbors', len(halfs))}",
        f"  MS/MS available = {bool(spectral and spectral.seed)}",
        f"  seed RT (s) = {getattr(spectral, 'seed_rt', None)}",
    ]
    lines.extend(_format_spectral_section(ctx))
    lines.extend(_format_method_and_library(ctx))
    if ctx.seed_mz is not None:
        lines.append("  multimer-implied monomer mass targets (approx):")
        shown = set()
        for label, t in monomer_mass_targets(float(ctx.seed_mz)):
            if "via [2M" in label or label.startswith("monomer"):
                key = round(t, 2)
                if key in shown:
                    continue
                shown.add(key)
                lines.append(f"    - {t:.4f}  ({label})")
                if len(shown) >= 6:
                    break

    lines.append("")
    lines.append("=== HIGHEST-PRIORITY: NEAR-ISOBAR NEIGHBORS (|Δm/z| ≤ 0.5) ===")
    if isobars:
        for i, ev in enumerate(isobars, start=1):
            lines.append(
                _fmt_neighbor(
                    i, ev, ctx.seed_mz, msms_map.get(ev.node.id), spectral=spectral
                )
            )
    else:
        lines.append("(none — consider multimer / half-mass logic below)")

    lines.append("")
    lines.append(
        f"=== MULTIMER-CONSISTENT NEIGHBORS (self-consistent residual ≤ {DEFAULT_HALF_DMZ_MAX} Da) ==="
    )
    lines.append(
        "Neighbor ion + seed precursor form a consistent [2M…]/[3M…] relationship "
        "(not merely within 2 Da of any of many mass targets)."
    )
    if halfs:
        for i, ev in enumerate(halfs[:15], start=1):
            lines.append(
                _fmt_neighbor(
                    i, ev, ctx.seed_mz, msms_map.get(ev.node.id), spectral=spectral
                )
            )
    else:
        lines.append("(none)")

    if hyps:
        lines.append("")
        lines.append(
            "=== MASS-CONSISTENT LIBRARY SMILES (monomer OR multimer adduct vs seed m/z) ==="
        )
        for i, h in enumerate(hyps, start=1):
            # try match msms by scanning neighbors with same smiles is hard; skip
            lines.append(
                f"{i:02d}. SMILES={h['smiles']} | edge_cos={h['cosine']:.3f} | "
                f"|Δm/z|={h['delta_mz']} | |Δhalf|={h.get('half_mass_delta')} | "
                f"adduct~{h.get('adduct')} | err_Da={h.get('mass_error_da')} | "
                f"name={h.get('name')}"
            )

    # Neighbors ranked by combined edge + msms when available
    def _rank_key(ev: NeighborEvidence):
        mc = msms_map.get(ev.node.id)
        ms = mc if mc is not None else 0.0
        return (-(0.55 * ev.cosine + 0.45 * ms), -ev.evidence_score(ctx.seed_mz))

    if msms_map:
        lines.append("")
        lines.append(
            "=== DIRECT NEIGHBORS ranked by 0.55*edge_cos + 0.45*msms_cos (MS/MS-aware) ==="
        )
        ordered = sorted(ranked, key=_rank_key)
    else:
        lines.append("")
        lines.append(
            "=== ALL DIRECT NEIGHBORS (ranked by mass-aware evidence score) ==="
        )
        ordered = ranked
    for i, ev in enumerate(ordered, start=1):
        lines.append(
            _fmt_neighbor(
                i, ev, ctx.seed_mz, msms_map.get(ev.node.id), spectral=spectral
            )
        )

    if ctx.two_hop_named:
        lines.append("")
        lines.append("SELECTED 2-HOP ANNOTATED NODES (context only; still respect mass):")
        for j, n in enumerate(ctx.two_hop_named, start=1):
            mz2 = f"{n.mz:.4f}" if n.mz is not None else "?"
            smi = f" | SMILES={n.smiles}" if n.smiles else ""
            lines.append(f"{j:02d}. m/z={mz2} | name={n.name}{smi}")

    lines.extend(
        [
            "",
            "INSTRUCTIONS:",
            f"1) Propose a SMILES whose mass fits precursor m/z={mz} as monomer OR multimer adduct.",
            "2) If MS/MS diagnostics are present, use them to choose among mass-consistent candidates "
            "(e.g. Phe 120/166 → phenylalanine conjugate).",
            "3) Prefer neighbors with high edge_cos AND high msms_cos when both are available.",
            "4) If half-mass neighbors dominate with a shared scaffold, report monomer SMILES "
            "with adduct [2M+H]+ / [2M-H]- as appropriate.",
            "5) Prefer ★ NEAR-ISOBAR / ★ MS/MS-SIMILAR / mass-consistent library SMILES.",
            "6) End with the JSON block.",
        ]
    )
    if extra_instructions:
        lines.extend(["", "ADDITIONAL INSTRUCTIONS:", extra_instructions])
    return "\n".join(lines)


def build_messages(ctx: EgoContext, extra_instructions: str | None = None) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(ctx, extra_instructions)},
    ]
