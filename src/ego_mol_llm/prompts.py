"""Prompt templates for chemistry LLMs (ChemDFM / Qwen family)."""

from __future__ import annotations

from ego_mol_llm.ego import EgoContext, NeighborEvidence


SYSTEM_PROMPT = """You are an expert mass spectrometry and natural-product chemist.
You predict molecular structure (SMILES) for an UNKNOWN precursor using only its
MS/MS molecular-network neighborhood (spectral similarity, precursor m/z differences,
and library annotations of neighbors).

Critical rules (follow in order):
1. The query identity is hidden. Do NOT invent peak lists you were not given.
2. MASS FIRST: any proposed SMILES must be consistent with the query precursor m/z
   under a common adduct ([M-H]-, [M+H]+, [M+Na]+, [M+NH4]+, [M+H-H2O]+, etc.).
   Reject peptides/lipids whose neutral mass is far from the precursor.
3. NEAR-ISOBAR PRIORITY: neighbors with |Δm/z| ≤ 0.5 Da and high cosine are the
   strongest evidence (possible library self-match or close analog). Prefer their
   scaffolds/SMILES over distant high-cosine nodes (|Δm/z| >> 50).
4. Distant high-cosine edges often share fragmentation motifs only — do not copy
   their full structure if mass does not fit.
5. Use annotated SMILES on near-mass neighbors as primary structure hypotheses.
6. Output MUST include a JSON block (and may also use key: value lines) with:
{
  "smiles": "<canonical SMILES or null>",
  "iupac_or_common_name": "<string or null>",
  "formula": "<Hill formula or null>",
  "adduct": "<e.g. [M-H]- or [M+H]+>",
  "confidence": <float 0-1>,
  "rationale": "<2-5 sentences citing neighbor m/z, cosine, Δm/z>",
  "alternatives": [{"smiles": "...", "confidence": 0.0, "note": "..."}]
}
"""


def _fmt_neighbor(i: int, ev: NeighborEvidence, seed_mz: float | None) -> str:
    n = ev.node
    name = n.name if n.is_annotated else "NO_MATCH"
    smiles = n.smiles or ""
    mz = f"{n.mz:.4f}" if n.mz is not None else "?"
    cos = f"{ev.cosine:.3f}"
    dmz = ev.resolved_delta_mz(seed_mz)
    dmz_s = f"{dmz:.4f}" if dmz is not None else "?"
    score = f"{ev.evidence_score(seed_mz):.3f}"
    smi_part = f" | SMILES={smiles}" if smiles else ""
    tag = ""
    if dmz is not None and dmz <= 0.05:
        tag = " | ★ NEAR-ISOBAR"
    elif dmz is not None and dmz <= 0.5:
        tag = " | near-isobar"
    return (
        f"{i:02d}. m/z={mz} | cosine={cos} | |Δm/z|={dmz_s} | score={score} "
        f"| name={name}{smi_part}{tag}"
    )


def build_user_prompt(ctx: EgoContext, extra_instructions: str | None = None) -> str:
    mz = f"{ctx.seed_mz:.6f}" if ctx.seed_mz is not None else "unknown"
    ranked = ctx.top_neighbors
    isobars = ctx.near_isobars(0.5)
    hyps = ctx.neighbor_structure_hypotheses(mass_tol_da=0.05, dmz_max=2.0, limit=8)

    lines = [
        "TASK: Predict the structure of the UNKNOWN center node of this MS/MS ego network.",
        "",
        "QUERY (unknown structure):",
        f"  precursor m/z = {mz}",
        f"  node_id = {ctx.seed.id}",
        f"  degree = {ctx.meta.get('degree', len(ctx.neighbors))}",
        f"  near-isobar neighbors (|Δm/z|≤0.5) = {len(isobars)}",
        "",
        "=== HIGHEST-PRIORITY EVIDENCE: NEAR-ISOBAR NEIGHBORS (|Δm/z| ≤ 0.5) ===",
        "These dominate the structure call when cosine is high.",
    ]
    if isobars:
        for i, ev in enumerate(isobars, start=1):
            lines.append(_fmt_neighbor(i, ev, ctx.seed_mz))
    else:
        lines.append("(none with |Δm/z|≤0.5)")

    if hyps:
        lines.append("")
        lines.append("=== MASS-CONSISTENT LIBRARY SMILES FROM NEIGHBORS (pre-filtered) ===")
        for i, h in enumerate(hyps, start=1):
            lines.append(
                f"{i:02d}. SMILES={h['smiles']} | cos={h['cosine']:.3f} | "
                f"|Δm/z|={h['delta_mz']} | adduct~{h.get('adduct')} | "
                f"name={h.get('name')}"
            )

    lines.append("")
    lines.append(
        "=== ALL DIRECT NEIGHBORS (ranked by mass-aware evidence score, not cosine alone) ==="
    )
    for i, ev in enumerate(ranked, start=1):
        lines.append(_fmt_neighbor(i, ev, ctx.seed_mz))

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
            f"1) Propose a SMILES whose neutral mass fits precursor m/z={mz} with a common adduct.",
            "2) Prefer scaffolds from ★ NEAR-ISOBAR / mass-consistent library SMILES above.",
            "3) Do NOT propose large peptides or distant lipids unless mass matches.",
            "4) End with the JSON block (confidence calibrated; lower if mass evidence is weak).",
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
