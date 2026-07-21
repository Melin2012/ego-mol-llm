"""Prompt templates for chemistry LLMs (ChemDFM / Qwen family)."""

from __future__ import annotations

from ego_mol_llm.ego import EgoContext, NeighborEvidence


SYSTEM_PROMPT = """You are an expert mass spectrometry and natural-product chemist.
You predict molecular structure (SMILES) for an UNKNOWN precursor using only its
MS/MS molecular-network neighborhood (spectral similarity, precursor m/z differences,
and library annotations of neighbors).

Rules:
1. The query node identity is hidden. Do NOT invent experimental peak lists you were not given.
2. Prefer structures consistent with precursor m/z (assume [M-H]- if m/z looks like neutral-1,
   or [M+H]+ if neutral+1; state the adduct).
3. Use high cosine neighbors and explicit library structures as strongest evidence.
4. Distrust weak cosine isobars (e.g. peptides near the same m/z) unless spectral evidence is strong.
5. Output MUST end with a JSON block in exactly this schema:
{
  "smiles": "<canonical SMILES or null>",
  "iupac_or_common_name": "<string or null>",
  "formula": "<Hill formula or null>",
  "adduct": "<e.g. [M-H]- or [M+H]+>",
  "confidence": <float 0-1>,
  "rationale": "<2-5 sentences>",
  "alternatives": [{"smiles": "...", "confidence": 0.0, "note": "..."}]
}
6. SMILES must be chemically valid when possible. If unsure, lower confidence and list alternatives.
"""


def _fmt_neighbor(i: int, ev: NeighborEvidence) -> str:
    n = ev.node
    name = n.name if n.is_annotated else "NO_MATCH"
    smiles = n.smiles or ""
    mz = f"{n.mz:.4f}" if n.mz is not None else "?"
    cos = f"{ev.cosine:.3f}"
    dmz = f"{ev.edge.abs_diff_mz:.4f}" if ev.edge.abs_diff_mz is not None else "?"
    smi_part = f" | SMILES={smiles}" if smiles else ""
    return (
        f"{i:02d}. m/z={mz} | cosine={cos} | |Δm/z|={dmz} | name={name}{smi_part}"
    )


def build_user_prompt(ctx: EgoContext, extra_instructions: str | None = None) -> str:
    mz = f"{ctx.seed_mz:.6f}" if ctx.seed_mz is not None else "unknown"
    lines = [
        "TASK: Predict the structure of the UNKNOWN center node of this MS/MS ego network.",
        "",
        "QUERY (unknown structure):",
        f"  precursor m/z = {mz}",
        f"  node_id = {ctx.seed.id}",
        f"  degree = {ctx.meta.get('degree', len(ctx.neighbors))}",
        "",
        "DIRECT SPECTRAL NEIGHBORS (sorted by cosine similarity):",
    ]
    for i, ev in enumerate(ctx.top_neighbors, start=1):
        lines.append(_fmt_neighbor(i, ev))

    if ctx.two_hop_named:
        lines.append("")
        lines.append("SELECTED 2-HOP ANNOTATED NODES (chemical context, not direct edges):")
        for j, n in enumerate(ctx.two_hop_named, start=1):
            mz2 = f"{n.mz:.4f}" if n.mz is not None else "?"
            smi = f" | SMILES={n.smiles}" if n.smiles else ""
            lines.append(f"{j:02d}. m/z={mz2} | name={n.name}{smi}")

    lines.extend(
        [
            "",
            "Please reason carefully about scaffold family, formula, and adduct,",
            "then give your best SMILES prediction with a calibrated confidence.",
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
