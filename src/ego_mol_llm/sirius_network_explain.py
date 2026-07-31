"""
SIRIUS-first fragment explanation for ego networks (offline CLI).

Analogous to ``cfm_network_explain`` (CFM-ID–first), but uses **SIRIUS
mass decomposition / fragmentation-tree chemistry**:

1. Neighbors with known SMILES → parent formula (RDKit) → ``sirius decomp
   --parent FORMULA`` on experimental peaks → peak → subformula map
2. Blind seed → transfer labels when seed peaks match annotated neighbor m/z;
   also decomp seed peaks under top SIRIUS/CSI formula (if available)
3. LLM prompt gets structured peak → formula / loss chemistry **before**
   proposing SMILES

Does not require CSI login for ``decomp`` (local). Full ``formulas`` tree
jobs are optional and slower.
"""

from __future__ import annotations

import os
import re
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ego_mol_llm.sirius import find_sirius_binary, ionization_for_mode


def _decomp_work_dir() -> Path:
    """Stable long-path work dir (Windows short TEMP paths break some SIRIUS writes)."""
    base = Path(os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP") or ".")
    d = base / "ego_mol_sirius_decomp"
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class PeakFormulaAnnotation:
    mz: float
    intensity: float
    fragment_formula: str | None = None
    parent_formula: str | None = None
    adduct: str | None = None
    alt_formulas: list[str] = field(default_factory=list)
    source: str = "sirius_decomp"  # sirius_decomp | transferred | seed_formula | unexplained
    note: str = ""
    neighbor_id: str | None = None
    neighbor_name: str | None = None
    neighbor_smiles: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NodeSiriusFragmentExplanation:
    node_id: str
    smiles: str | None
    name: str | None
    ion_mode: str | None
    adduct: str | None
    role: str  # seed | neighbor
    parent_formula: str | None = None
    n_peaks: int = 0
    n_annotated: int = 0
    peak_annotations: list[PeakFormulaAnnotation] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["peak_annotations"] = [p.to_dict() for p in self.peak_annotations]
        return d


@dataclass
class EgoSiriusFragmentExplanation:
    spectrum_id: str
    seed: NodeSiriusFragmentExplanation | None = None
    neighbors: list[NodeSiriusFragmentExplanation] = field(default_factory=list)
    seed_peak_map: list[PeakFormulaAnnotation] = field(default_factory=list)
    chemistry_summary: list[str] = field(default_factory=list)
    backend: str = "sirius_decomp"
    product_version: str = "0.4-sirius-first"
    seed_sirius_formula: str | None = None
    seed_sirius_smiles: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "spectrum_id": self.spectrum_id,
            "seed": self.seed.to_dict() if self.seed else None,
            "neighbors": [n.to_dict() for n in self.neighbors],
            "seed_peak_map": [p.to_dict() for p in self.seed_peak_map],
            "chemistry_summary": self.chemistry_summary,
            "backend": self.backend,
            "product_version": self.product_version,
            "seed_sirius_formula": self.seed_sirius_formula,
            "seed_sirius_smiles": self.seed_sirius_smiles,
        }


def formula_from_smiles(smiles: str) -> str | None:
    try:
        from rdkit import Chem
        from rdkit.Chem.rdMolDescriptors import CalcMolFormula

        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None
        return CalcMolFormula(m)
    except Exception:
        return None


def _normalize_adduct(ion_mode: str | None, adduct: str | None = None) -> str:
    if adduct and str(adduct).strip():
        a = str(adduct).strip()
        # sirius decomp accepts [M+H]+ style
        return a.replace(" ", "")
    return ionization_for_mode(ion_mode).replace(" ", "")


def run_sirius_decomp(
    masses: list[float],
    *,
    parent_formula: str,
    ion_mode: str | None = "positive",
    adduct: str | None = None,
    sirius_bin: str | Path | None = None,
    ppm: float = 10.0,
    abs_tol: float = 0.02,
    max_decomps: int = 5,
    timeout_s: float = 120.0,
) -> dict[float, list[str]]:
    """
    Map peak m/z → list of subformulas of ``parent_formula`` via SIRIUS decomp.

    Returns dict keyed by rounded m/z (4 decimals) → formula strings.
    """
    bin_path = find_sirius_binary(sirius_bin)
    if bin_path is None or not masses or not parent_formula:
        return {}

    ion = _normalize_adduct(ion_mode, adduct)
    uniq: list[float] = []
    seen: set[float] = set()
    for m in masses:
        key = round(float(m), 5)
        if key not in seen:
            seen.add(key)
            uniq.append(float(m))

    # SIRIUS decomp rejects float-looking option values like "10.0" / "0.020"
    # on some Windows builds — use compact formatting.
    def _num(x: float | int) -> str:
        try:
            xf = float(x)
        except (TypeError, ValueError):
            return str(x)
        if xf == int(xf):
            return str(int(xf))
        return f"{xf:g}"

    # Prefer writing TSV via -o (stdout is mostly JVM noise on Windows).
    out_file = (_decomp_work_dir() / f"decomp_{uuid.uuid4().hex}.tsv").resolve()
    cmd = [
        str(bin_path),
        "decomp",
        "--parent",
        parent_formula,
        "-i",
        ion,
        "-p",
        _num(ppm),
        "-a",
        _num(abs_tol),
        "-d",
        _num(max_decomps),
        "-o",
        str(out_file),
        "-m",
        *[f"{m:.6f}" for m in uniq],
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
    except Exception:
        return {}

    text = ""
    if out_file.is_file():
        try:
            text = out_file.read_text(encoding="utf-8", errors="replace")
        finally:
            try:
                out_file.unlink(missing_ok=True)
            except OSError:
                pass
    parsed = _parse_decomp_output(text)
    if parsed:
        return parsed
    # Fallback: stdout TSV (no -o)
    cmd2 = [
        str(bin_path),
        "decomp",
        "--parent",
        parent_formula,
        "-i",
        ion,
        "-p",
        _num(ppm),
        "-a",
        _num(abs_tol),
        "-d",
        _num(max_decomps),
        "-m",
        *[f"{m:.6f}" for m in uniq],
    ]
    try:
        proc2 = subprocess.run(
            cmd2,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
        return _parse_decomp_output(proc2.stdout or "")
    except Exception:
        return {}


def _parse_decomp_output(text: str) -> dict[float, list[str]]:
    """
    Parse SIRIUS decomp stdout table (TSV preferred)::

        parentFormula\\tadduct\\tm/z\\tdecompositions
        C5H4N4O2\\t[M + H]+\\t136.014\\tC5HN3O2
        C5H4N4O2\\t[M + H]+\\t110.035\\tC4H3N3O,C5H3NO2,C3HN4O
    """
    out: dict[float, list[str]] = {}
    formula_tok = re.compile(r"^[A-Z][A-Za-z0-9]*$")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.lower().startswith("parentformula"):
            continue
        # skip JVM / log noise
        if line.startswith("[") or "INFO" in line[:20] or "SEVERE" in line[:20]:
            continue
        if "http" in line.lower() or "cite if" in line.lower():
            continue

        # Prefer real TSV (adduct contains spaces: "[M + H]+")
        if "\t" in raw:
            cols = [c.strip() for c in raw.split("\t")]
            if len(cols) < 4:
                continue
            try:
                mz_val = float(cols[2])
            except ValueError:
                continue
            decomp_field = cols[3]
        else:
            # Fallback: find m/z float then next token(s) as formulas
            parts = line.split()
            mz_val = None
            decomp_field = None
            for i, p in enumerate(parts):
                try:
                    cand = float(p)
                except ValueError:
                    continue
                if 10.0 < cand < 5000.0:
                    mz_val = cand
                    if i + 1 < len(parts):
                        decomp_field = parts[i + 1]
                    break
            if mz_val is None or not decomp_field:
                continue

        formulas = [f.strip() for f in decomp_field.split(",") if f.strip()]
        formulas = [
            f for f in formulas if formula_tok.match(f) and any(c.isdigit() for c in f)
        ]
        if formulas and mz_val is not None:
            out[round(float(mz_val), 4)] = formulas
    return out


def annotate_peaks_with_parent(
    peaks: list[tuple[float, float]],
    *,
    parent_formula: str,
    ion_mode: str | None,
    adduct: str | None = None,
    sirius_bin: str | Path | None = None,
    top_n: int = 30,
    ppm: float = 10.0,
    abs_tol: float = 0.02,
) -> list[PeakFormulaAnnotation]:
    tops = sorted(peaks, key=lambda x: -x[1])[:top_n]
    if not tops or not parent_formula:
        return [
            PeakFormulaAnnotation(mz=float(m), intensity=float(i), source="unexplained")
            for m, i in tops
        ]
    mass_map = run_sirius_decomp(
        [m for m, _ in tops],
        parent_formula=parent_formula,
        ion_mode=ion_mode,
        adduct=adduct,
        sirius_bin=sirius_bin,
        ppm=ppm,
        abs_tol=abs_tol,
    )
    ion = _normalize_adduct(ion_mode, adduct)
    anns: list[PeakFormulaAnnotation] = []
    for mz, inten in tops:
        forms = mass_map.get(round(float(mz), 4)) or mass_map.get(round(float(mz), 3))
        if not forms:
            # nearest key within abs_tol
            forms = None
            for k, v in mass_map.items():
                if abs(k - float(mz)) <= abs_tol:
                    forms = v
                    break
        if forms:
            anns.append(
                PeakFormulaAnnotation(
                    mz=float(mz),
                    intensity=float(inten),
                    fragment_formula=forms[0],
                    parent_formula=parent_formula,
                    adduct=ion,
                    alt_formulas=forms[1:],
                    source="sirius_decomp",
                    note=f"subformula of {parent_formula}",
                )
            )
        else:
            anns.append(
                PeakFormulaAnnotation(
                    mz=float(mz),
                    intensity=float(inten),
                    parent_formula=parent_formula,
                    adduct=ion,
                    source="unexplained",
                    note="no subformula within tolerance",
                )
            )
    return anns


def transfer_seed_formula_annotations(
    seed_peaks: list[tuple[float, float]],
    neighbor_expls: list[NodeSiriusFragmentExplanation],
    *,
    tol: float = 0.02,
    top_neighbors: int = 6,
) -> list[PeakFormulaAnnotation]:
    ranked = sorted(neighbor_expls, key=lambda n: -n.n_annotated)[:top_neighbors]

    def bin_mz(mz: float) -> int:
        return int(round(mz / tol))

    bins: dict[int, list[tuple[NodeSiriusFragmentExplanation, PeakFormulaAnnotation]]] = {}
    for ne in ranked:
        for pa in ne.peak_annotations:
            if not pa.fragment_formula:
                continue
            bins.setdefault(bin_mz(pa.mz), []).append((ne, pa))

    out: list[PeakFormulaAnnotation] = []
    for mz, inten in sorted(seed_peaks, key=lambda x: -x[1])[:40]:
        hits = bins.get(bin_mz(mz), [])
        if not hits:
            for d in (-1, 1):
                hits = bins.get(bin_mz(mz) + d, [])
                if hits:
                    break
        if not hits:
            out.append(
                PeakFormulaAnnotation(
                    mz=float(mz),
                    intensity=float(inten),
                    source="unexplained",
                    note="no SIRIUS transfer from top neighbors",
                )
            )
            continue
        ne, pa = hits[0]
        out.append(
            PeakFormulaAnnotation(
                mz=float(mz),
                intensity=float(inten),
                fragment_formula=pa.fragment_formula,
                parent_formula=pa.parent_formula or ne.parent_formula,
                adduct=pa.adduct or ne.adduct,
                alt_formulas=list(pa.alt_formulas or []),
                source="transferred",
                note=(
                    f"shared with neighbor {ne.name or ne.node_id} "
                    f"(parent={ne.parent_formula}; SMILES={ne.smiles})"
                ),
                neighbor_id=ne.node_id,
                neighbor_name=ne.name,
                neighbor_smiles=ne.smiles,
            )
        )
    return out


def merge_seed_maps(
    transferred: list[PeakFormulaAnnotation],
    seed_direct: list[PeakFormulaAnnotation],
    *,
    tol: float = 0.02,
) -> list[PeakFormulaAnnotation]:
    """Prefer transferred labels; fill gaps with seed-parent decomp."""
    by_bin: dict[int, PeakFormulaAnnotation] = {}

    def b(mz: float) -> int:
        return int(round(mz / tol))

    for p in seed_direct:
        by_bin[b(p.mz)] = p
    for p in transferred:
        if p.fragment_formula:
            by_bin[b(p.mz)] = p
        elif b(p.mz) not in by_bin:
            by_bin[b(p.mz)] = p
    # intensity order from transferred list first (same ranking as seed peaks)
    order = [b(p.mz) for p in transferred] if transferred else list(by_bin.keys())
    seen: set[int] = set()
    out: list[PeakFormulaAnnotation] = []
    for k in order:
        if k in seen or k not in by_bin:
            continue
        seen.add(k)
        out.append(by_bin[k])
    for k, p in by_bin.items():
        if k not in seen:
            out.append(p)
    return out


def build_ego_sirius_fragment_explanation(
    *,
    spectrum_id: str,
    seed_peaks: list[tuple[float, float]],
    seed_ion_mode: str | None,
    neighbors: list[dict[str, Any]],
    max_neighbors: int = 6,
    sirius_bin: str | Path | None = None,
    seed_formula: str | None = None,
    seed_sirius_smiles: str | None = None,
    top_peaks: int = 25,
    ppm: float = 10.0,
    abs_tol: float = 0.02,
) -> EgoSiriusFragmentExplanation:
    """
    neighbors: dicts with id, smiles, name, peaks, mz, msms_cosine, ion_mode
    """

    def nkey(n: dict) -> tuple:
        return (
            0 if n.get("smiles") else 1,
            -(float(n.get("msms_cosine") or 0.0)),
        )

    ranked = sorted(neighbors, key=nkey)[:max_neighbors]
    neigh_expls: list[NodeSiriusFragmentExplanation] = []

    for n in ranked:
        smi = n.get("smiles")
        peaks = n.get("peaks") or []
        if not smi or not peaks:
            continue
        parent = formula_from_smiles(smi)
        ion = n.get("ion_mode") or seed_ion_mode
        adduct = _normalize_adduct(ion)
        ne = NodeSiriusFragmentExplanation(
            node_id=str(n.get("id")),
            smiles=smi,
            name=n.get("name"),
            ion_mode=ion,
            adduct=adduct,
            role="neighbor",
            parent_formula=parent,
            n_peaks=len(peaks),
        )
        if not parent:
            ne.meta["error"] = "no_formula_from_smiles"
            neigh_expls.append(ne)
            continue
        anns = annotate_peaks_with_parent(
            peaks,
            parent_formula=parent,
            ion_mode=ion,
            adduct=adduct,
            sirius_bin=sirius_bin,
            top_n=top_peaks,
            ppm=ppm,
            abs_tol=abs_tol,
        )
        for a in anns:
            a.neighbor_id = ne.node_id
            a.neighbor_name = ne.name
            a.neighbor_smiles = smi
        ne.peak_annotations = anns
        ne.n_annotated = sum(1 for a in anns if a.fragment_formula)
        neigh_expls.append(ne)

    transferred = transfer_seed_formula_annotations(seed_peaks, neigh_expls)
    seed_direct: list[PeakFormulaAnnotation] = []
    if seed_formula:
        seed_direct = annotate_peaks_with_parent(
            seed_peaks,
            parent_formula=seed_formula,
            ion_mode=seed_ion_mode,
            sirius_bin=sirius_bin,
            top_n=top_peaks,
            ppm=ppm,
            abs_tol=abs_tol,
        )
        for a in seed_direct:
            if a.fragment_formula:
                a.source = "seed_formula"
                a.note = f"decomp under seed SIRIUS formula {seed_formula}"

    seed_map = merge_seed_maps(transferred, seed_direct)
    n_trans = sum(1 for p in seed_map if p.source == "transferred" and p.fragment_formula)
    n_seed = sum(1 for p in seed_map if p.source == "seed_formula" and p.fragment_formula)
    n_any = sum(1 for p in seed_map if p.fragment_formula)

    summary: list[str] = [
        f"Seed peaks with fragment formulas: {n_any}/{len(seed_map)} "
        f"(transferred={n_trans}, seed_SIRIUS_formula={n_seed})",
    ]
    if seed_formula:
        summary.append(f"Seed parent formula (SIRIUS/CSI or prior): {seed_formula}")
    if seed_sirius_smiles:
        summary.append(f"Seed CSI top SMILES (optional clue): {seed_sirius_smiles}")
    for ne in neigh_expls[:5]:
        summary.append(
            f"Neighbor {ne.name or ne.node_id}: parent={ne.parent_formula} "
            f"annotated={ne.n_annotated}/{min(ne.n_peaks, top_peaks)}"
        )

    seed_node = NodeSiriusFragmentExplanation(
        node_id="0",
        smiles=None,
        name="SEED_UNKNOWN",
        ion_mode=seed_ion_mode,
        adduct=_normalize_adduct(seed_ion_mode),
        role="seed",
        parent_formula=seed_formula,
        n_peaks=len(seed_peaks),
        n_annotated=n_any,
        peak_annotations=seed_map,
        meta={"note": "structure hidden; formulas from network transfer + seed SIRIUS formula"},
    )

    return EgoSiriusFragmentExplanation(
        spectrum_id=spectrum_id,
        seed=seed_node,
        neighbors=neigh_expls,
        seed_peak_map=seed_map,
        chemistry_summary=summary,
        backend="sirius_decomp",
        seed_sirius_formula=seed_formula,
        seed_sirius_smiles=seed_sirius_smiles,
    )


def format_sirius_fragment_explanation_for_prompt(
    expl: EgoSiriusFragmentExplanation | dict[str, Any],
    *,
    max_seed_peaks: int = 15,
    max_neighbors: int = 4,
) -> list[str]:
    if not isinstance(expl, dict):
        expl = expl.to_dict()

    lines = [
        "",
        "=== SIRIUS FRAGMENT EXPLANATION (precomputed; network-first) ===",
        f"  backend = {expl.get('backend')}",
        "  Peak chemistry from SIRIUS subformula decomp (not CFM-ID SMILES fragments).",
    ]
    for s in (expl.get("chemistry_summary") or [])[:8]:
        lines.append(f"  • {s}")

    if expl.get("seed_sirius_formula") or expl.get("seed_sirius_smiles"):
        lines.append("  Seed-level SIRIUS/CSI prior (structure still unknown to you):")
        if expl.get("seed_sirius_formula"):
            lines.append(f"    parent_formula ≈ {expl.get('seed_sirius_formula')}")
        if expl.get("seed_sirius_smiles"):
            lines.append(
                f"    CSI top SMILES clue (may be wrong): {expl.get('seed_sirius_smiles')}"
            )

    lines.append("  SEED experimental peaks → fragment FORMULAS (from network / SIRIUS):")
    for p in (expl.get("seed_peak_map") or [])[:max_seed_peaks]:
        if not isinstance(p, dict):
            continue
        ff = p.get("fragment_formula")
        if not ff:
            continue
        alts = p.get("alt_formulas") or []
        alt_s = f" alts={','.join(alts[:3])}" if alts else ""
        lines.append(
            f"    m/z {float(p.get('mz') or 0):.4f} → {ff} "
            f"[{p.get('source')}] parent={p.get('parent_formula')}{alt_s} "
            f"{(p.get('note') or '')[:90]}"
        )

    lines.append("  Neighbor structure → SIRIUS peak formulas:")
    for n in (expl.get("neighbors") or [])[:max_neighbors]:
        if not isinstance(n, dict):
            continue
        lines.append(
            f"    • {n.get('name') or n.get('node_id')}: SMILES={n.get('smiles')} "
            f"parent={n.get('parent_formula')} "
            f"annotated={n.get('n_annotated')}/{n.get('n_peaks')}"
        )
        shown = 0
        for pa in n.get("peak_annotations") or []:
            if not isinstance(pa, dict) or not pa.get("fragment_formula"):
                continue
            lines.append(
                f"        peak {float(pa.get('mz') or 0):.4f} → {pa.get('fragment_formula')}"
            )
            shown += 1
            if shown >= 5:
                break

    lines.append(
        "  Use fragment FORMULAS + shared peaks as substructure / elemental clues for the "
        "unknown seed; prefer SMILES whose losses match annotated formulas "
        "(e.g. −NH3 → drop N+3H, −H2O → drop H2O)."
    )
    return lines
