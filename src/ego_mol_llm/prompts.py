"""Prompt templates for chemistry LLMs (ChemDFM / Qwen family)."""

from __future__ import annotations

from ego_mol_llm.ego import EgoContext, NeighborEvidence
from ego_mol_llm.validate import monomer_mass_targets


SYSTEM_PROMPT = """You are an expert mass spectrometry and natural-product chemist.
You predict molecular structure (SMILES) for an UNKNOWN precursor using only its
MS/MS molecular-network neighborhood (spectral similarity, precursor m/z differences,
and library annotations of neighbors).

Critical rules (follow in order):
1. The query identity is hidden. Do NOT invent peak lists you were not given.
2. MASS FIRST: any proposed SMILES must fit the query precursor m/z under a common
   adduct, INCLUDING MULTIMERS:
   - monomers: [M-H]-, [M+H]+, [M+Na]+, [M+NH4]+, [M+H-H2O]+, …
   - dimers: [2M+H]+, [2M-H]-, [2M+Na]+, …
   - trimers: [3M+H]+, [3M-H]-, …
   If the precursor is high (e.g. >600) and many neighbors sit near half-mass,
   strongly consider [2M+H]+ / [2M-H]- of a monomer library structure.
3. NEAR-ISOBAR PRIORITY: neighbors with |Δm/z| ≤ 0.5 Da and high cosine are strongest
   for monomer self-matches.
4. HALF-MASS PRIORITY: when |Δm/z| to seed is large but neighbors cluster near m/2
   (or other multimer-implied monomer ion masses) with shared scaffold annotations
   (e.g. bile acids at ~407 when seed is ~813), the monomer structure is the answer
   and the seed ion is likely [2M+H]+ (or similar).
5. Distant high-cosine edges can share fragmentation only — require mass consistency.
6. Output MUST include a JSON block with:
{
  "smiles": "<canonical SMILES of the *neutral monomer* structure>",
  "iupac_or_common_name": "<string or null>",
  "formula": "<Hill formula or null>",
  "adduct": "<e.g. [M-H]- or [2M+H]+ for the observed precursor>",
  "confidence": <float 0-1>,
  "rationale": "<2-5 sentences citing neighbor m/z, cosine, Δm/z, multimer logic>",
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
    hdmz = ev.half_mass_delta(seed_mz)
    dmz_s = f"{dmz:.4f}" if dmz is not None else "?"
    hdmz_s = f"{hdmz:.4f}" if hdmz is not None else "?"
    score = f"{ev.evidence_score(seed_mz):.3f}"
    smi_part = f" | SMILES={smiles}" if smiles else ""
    tag = ""
    if dmz is not None and dmz <= 0.05:
        tag = " | ★ NEAR-ISOBAR"
    elif dmz is not None and dmz <= 0.5:
        tag = " | near-isobar"
    elif hdmz is not None and hdmz <= 0.5:
        tag = " | ★ HALF-MASS (multimer monomer?)"
    elif hdmz is not None and hdmz <= 2.0:
        tag = " | half-mass region"
    return (
        f"{i:02d}. m/z={mz} | cosine={cos} | |Δm/z|={dmz_s} | |Δhalf|={hdmz_s} "
        f"| score={score} | name={name}{smi_part}{tag}"
    )


def build_user_prompt(ctx: EgoContext, extra_instructions: str | None = None) -> str:
    mz = f"{ctx.seed_mz:.6f}" if ctx.seed_mz is not None else "unknown"
    ranked = ctx.top_neighbors
    isobars = ctx.near_isobars(0.5)
    halfs = ctx.half_mass_neighbors(1.0)
    hyps = ctx.neighbor_structure_hypotheses(
        mass_tol_da=0.05, dmz_max=2.0, half_dmz_max=2.0, limit=10
    )

    lines = [
        "TASK: Predict the structure of the UNKNOWN center node of this MS/MS ego network.",
        "",
        "QUERY (unknown structure):",
        f"  precursor m/z = {mz}",
        f"  node_id = {ctx.seed.id}",
        f"  degree = {ctx.meta.get('degree', len(ctx.neighbors))}",
        f"  near-isobar neighbors (|Δm/z|≤0.5) = {len(isobars)}",
        f"  half-mass neighbors (|Δhalf|≤1.0) = {ctx.meta.get('n_half_mass_neighbors', len(halfs))}",
    ]
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
            lines.append(_fmt_neighbor(i, ev, ctx.seed_mz))
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
            lines.append(_fmt_neighbor(i, ev, ctx.seed_mz))
    else:
        lines.append("(none)")

    if hyps:
        lines.append("")
        lines.append(
            "=== MASS-CONSISTENT LIBRARY SMILES (monomer OR multimer adduct vs seed m/z) ==="
        )
        for i, h in enumerate(hyps, start=1):
            lines.append(
                f"{i:02d}. SMILES={h['smiles']} | cos={h['cosine']:.3f} | "
                f"|Δm/z|={h['delta_mz']} | |Δhalf|={h.get('half_mass_delta')} | "
                f"adduct~{h.get('adduct')} | err_Da={h.get('mass_error_da')} | "
                f"name={h.get('name')}"
            )

    lines.append("")
    lines.append(
        "=== ALL DIRECT NEIGHBORS (ranked by mass-aware evidence score) ==="
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
            f"1) Propose a SMILES whose mass fits precursor m/z={mz} as monomer OR multimer adduct.",
            "2) If half-mass neighbors dominate with a shared scaffold, report that monomer SMILES "
            "and set adduct to [2M+H]+ or [2M-H]- as appropriate.",
            "3) Prefer ★ NEAR-ISOBAR / ★ HALF-MASS / mass-consistent library SMILES.",
            "4) End with the JSON block.",
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
