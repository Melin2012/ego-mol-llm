"""Prompt templates for chemistry LLMs (ChemDFM / Qwen family)."""

from __future__ import annotations

from ego_mol_llm.ego import EgoContext, NeighborEvidence
from ego_mol_llm.validate import monomer_mass_targets


SYSTEM_PROMPT = """You are an expert mass spectrometry and natural-product chemist.
You predict molecular structure (SMILES) for an UNKNOWN precursor from its
MS/MS molecular-network ego neighborhood AND, when provided, raw MS/MS peak lists.

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
   - Edge cosine < 0.5 is weak; MS/MS cosine < 0.5 is weak spectral support.
4. NEAR-ISOBAR PRIORITY: neighbors with |Δm/z| ≤ 0.5 Da and high cosine are strongest
   for monomer self-matches — but reject annotations whose formula cannot fit m/z.
5. HALF-MASS PRIORITY: when |Δm/z| to seed is large but neighbors cluster near m/2
   with shared scaffold annotations, consider multimer ions of the monomer.
6. Distant high-cosine edges can share fragmentation only — require mass consistency.
7. Output MUST include a JSON block with:
{
  "smiles": "<canonical SMILES of the *neutral monomer* structure>",
  "iupac_or_common_name": "<string or null>",
  "formula": "<Hill formula or null>",
  "adduct": "<e.g. [M-H]- or [2M+H]+ for the observed precursor>",
  "confidence": <float 0-1>,
  "rationale": "<2-5 sentences citing m/z, edges, MS/MS diagnostics if present>",
  "alternatives": [{"smiles": "...", "confidence": 0.0, "note": "..."}]
}
"""


def _fmt_neighbor(
    i: int,
    ev: NeighborEvidence,
    seed_mz: float | None,
    msms_cos: float | None = None,
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
    elif hdmz is not None and hdmz <= 0.5:
        tag = " | ★ HALF-MASS (multimer monomer?)"
    elif hdmz is not None and hdmz <= 2.0:
        tag = " | half-mass region"
    if msms_cos is not None and msms_cos >= 0.7:
        tag += " | ★ MS/MS-SIMILAR"
    return (
        f"{i:02d}. m/z={mz} | edge_cos={cos}{msms_part} | |Δm/z|={dmz_s} | |Δhalf|={hdmz_s} "
        f"| score={score} | name={name}{smi_part}{tag}"
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


def build_user_prompt(ctx: EgoContext, extra_instructions: str | None = None) -> str:
    mz = f"{ctx.seed_mz:.6f}" if ctx.seed_mz is not None else "unknown"
    ranked = ctx.top_neighbors
    isobars = ctx.near_isobars(0.5)
    halfs = ctx.half_mass_neighbors(1.0)
    hyps = ctx.neighbor_structure_hypotheses(
        mass_tol_da=0.05, dmz_max=2.0, half_dmz_max=2.0, limit=10
    )
    msms_map = {}
    if getattr(ctx, "spectral", None) is not None:
        msms_map = ctx.spectral.neighbor_msms_cosine or {}

    lines = [
        "TASK: Predict the structure of the UNKNOWN center node of this MS/MS ego network.",
        "",
        "QUERY (unknown structure):",
        f"  precursor m/z = {mz}",
        f"  node_id = {ctx.seed.id}",
        f"  degree = {ctx.meta.get('degree', len(ctx.neighbors))}",
        f"  near-isobar neighbors (|Δm/z|≤0.5) = {len(isobars)}",
        f"  half-mass neighbors (|Δhalf|≤1.0) = {ctx.meta.get('n_half_mass_neighbors', len(halfs))}",
        f"  MS/MS available = {bool(getattr(ctx, 'spectral', None) and ctx.spectral and ctx.spectral.seed)}",
    ]
    lines.extend(_format_spectral_section(ctx))
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
                _fmt_neighbor(i, ev, ctx.seed_mz, msms_map.get(ev.node.id))
            )
    else:
        lines.append("(none — consider multimer / half-mass logic below)")

    lines.append("")
    lines.append(
        "=== HALF-MASS / MULTIMER-MONOMER NEIGHBORS (|Δhalf| ≤ 1.0) ==="
    )
    lines.append(
        "If seed m/z ≈ 2× these ions, the unknown may be [2M+H]+/[2M-H]- of this scaffold."
    )
    if halfs:
        for i, ev in enumerate(halfs[:15], start=1):
            lines.append(
                _fmt_neighbor(i, ev, ctx.seed_mz, msms_map.get(ev.node.id))
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
            _fmt_neighbor(i, ev, ctx.seed_mz, msms_map.get(ev.node.id))
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
