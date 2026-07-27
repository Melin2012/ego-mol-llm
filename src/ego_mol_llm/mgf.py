"""MGF spectrum I/O and MS/MS similarity for annotation prompts."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class Spectrum:
    peaks: list[tuple[float, float]]
    pepmass: float | None = None
    charge: int | None = None
    node_id: str | None = None
    meta: dict[str, str] = field(default_factory=dict)

    def top_peaks(self, n: int = 15) -> list[tuple[float, float]]:
        return sorted(self.peaks, key=lambda x: -x[1])[:n]

    def base_peak(self) -> tuple[float, float] | None:
        if not self.peaks:
            return None
        return max(self.peaks, key=lambda x: x[1])


def parse_mgf(path: str | Path) -> list[Spectrum]:
    """Parse an MGF file into Spectrum records (tab- or space-separated peaks)."""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"(?i)BEGIN IONS", text)
    out: list[Spectrum] = []
    for b in blocks[1:]:
        body = re.split(r"(?i)END IONS", b)[0]
        meta: dict[str, str] = {}
        peaks: list[tuple[float, float]] = []
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            if "=" in line and not (line[0].isdigit() or line.startswith(".")):
                k, v = line.split("=", 1)
                meta[k.strip().upper()] = v.strip()
            else:
                parts = re.split(r"[\s\t]+", line)
                if len(parts) >= 2:
                    try:
                        peaks.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        continue
        pepmass = None
        if "PEPMASS" in meta:
            try:
                pepmass = float(meta["PEPMASS"].split()[0])
            except ValueError:
                pepmass = None
        charge = None
        if "CHARGE" in meta:
            try:
                charge = int(re.sub(r"[^0-9\-]", "", meta["CHARGE"]) or "0") or None
            except ValueError:
                charge = None
        node_id = (
            meta.get("NETWORK_NODE_ID")
            or meta.get("SCANS")
            or meta.get("FEATURE_ID")
            or meta.get("CLUSTERINDEX")
        )
        out.append(
            Spectrum(
                peaks=peaks,
                pepmass=pepmass,
                charge=charge,
                node_id=str(node_id) if node_id else None,
                meta=meta,
            )
        )
    return out


def index_spectra(spectra: Iterable[Spectrum]) -> dict[str, Spectrum]:
    """Index by node id and rounded pepmass keys."""
    idx: dict[str, Spectrum] = {}
    for sp in spectra:
        if sp.node_id:
            idx[str(sp.node_id)] = sp
            idx[f"id:{sp.node_id}"] = sp
        if sp.pepmass is not None:
            # first wins for mz keys; still useful for seed-only files
            key = f"mz:{sp.pepmass:.3f}"
            idx.setdefault(key, sp)
            key2 = f"mz:{sp.pepmass:.2f}"
            idx.setdefault(key2, sp)
    return idx


def cosine_peaks(
    peaks_a: list[tuple[float, float]],
    peaks_b: list[tuple[float, float]],
    tol: float = 0.02,
    sqrt_intensity: bool = True,
) -> float:
    """
    Peak-list cosine similarity with m/z tolerance matching.

    By default applies √intensity weighting (GNPS-style) before L2 normalization
    so a single base peak cannot force cosine ≈ 1.0.
    """
    if not peaks_a or not peaks_b:
        return 0.0

    def dens(peaks: list[tuple[float, float]]) -> dict[float, float]:
        d: dict[float, float] = defaultdict(float)
        for mz, inten in peaks:
            if inten <= 0:
                continue
            w = math.sqrt(inten) if sqrt_intensity else float(inten)
            d[round(mz / tol) * tol] += w
        norm = math.sqrt(sum(v * v for v in d.values())) or 1.0
        return {k: v / norm for k, v in d.items()}

    a, b = dens(peaks_a), dens(peaks_b)
    score = 0.0
    used: set[float] = set()
    for ka, va in a.items():
        best = None
        best_d = 1e9
        for kb in b:
            if kb in used:
                continue
            dd = abs(ka - kb)
            if dd <= tol and dd < best_d:
                best_d = dd
                best = kb
        if best is not None:
            score += va * b[best]
            used.add(best)
    return max(0.0, min(1.0, score))


def diagnostic_ions(
    peaks: list[tuple[float, float]],
    tol: float = 0.02,
) -> dict[str, float]:
    """Return diagnostic ion labels → intensity (offline table in msms_explain)."""
    from ego_mol_llm.msms_explain import explain_diagnostics

    return {d.label: d.intensity for d in explain_diagnostics(peaks, tol=tol)}


def neutral_losses(
    peaks: list[tuple[float, float]],
    precursor_mz: float | None,
    top_n: int = 12,
) -> list[tuple[float, float, float, str]]:
    """
    Top peaks with neutral loss from precursor.
    Returns (frag_mz, intensity, loss, label).
    Labels come from the expanded offline loss table (msms_explain).
    """
    from ego_mol_llm.msms_explain import explain_neutral_losses

    return [
        (L.frag_mz, L.intensity, L.loss_da, L.label)
        for L in explain_neutral_losses(peaks, precursor_mz, top_n=top_n)
    ]


def format_peaks_for_prompt(peaks: list[tuple[float, float]], n: int = 15) -> str:
    tops = sorted(peaks, key=lambda x: -x[1])[:n]
    if not tops:
        return "(no peaks)"
    base = tops[0][1] or 1.0
    lines = []
    for mz, inten in tops:
        lines.append(f"{mz:.4f} ({100.0 * inten / base:.1f}%)")
    return "; ".join(lines)


@dataclass
class SpectralContext:
    """MS/MS attached to an ego prediction (v0.2+: RT/meta + offline explanation)."""

    seed: Spectrum | None = None
    neighbor_msms_cosine: dict[str, float] = field(default_factory=dict)
    seed_diagnostics: dict[str, float] = field(default_factory=dict)
    seed_losses: list[tuple[float, float, float, str]] = field(default_factory=list)
    n_spectra_indexed: int = 0
    sources: list[str] = field(default_factory=list)
    # v0.2 experimental context
    seed_rt: float | None = None
    seed_ion_mode: str | None = None
    seed_meta: dict[str, str] = field(default_factory=dict)
    neighbor_meta: dict[str, dict[str, Any]] = field(default_factory=dict)
    neighbor_rt: dict[str, float | None] = field(default_factory=dict)
    # neighbor peak lists for offline explanation (id → peaks)
    neighbor_peaks: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    # filled by attach_msms_explanation / build path
    explanation: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed_pepmass": self.seed.pepmass if self.seed else None,
            "seed_n_peaks": len(self.seed.peaks) if self.seed else 0,
            "seed_top_peaks": self.seed.top_peaks(15) if self.seed else [],
            "seed_diagnostics": self.seed_diagnostics,
            "seed_losses": [
                {"mz": m, "inten": i, "loss": lo, "label": lab}
                for m, i, lo, lab in self.seed_losses
            ],
            "neighbor_msms_cosine": self.neighbor_msms_cosine,
            "n_spectra_indexed": self.n_spectra_indexed,
            "sources": self.sources,
            "seed_rt": self.seed_rt,
            "seed_ion_mode": self.seed_ion_mode,
            "seed_meta": self.seed_meta,
            "neighbor_rt": self.neighbor_rt,
            "neighbor_meta": self.neighbor_meta,
            "explanation": self.explanation,
        }


def build_spectral_context(
    seed_id: str,
    seed_mz: float | None,
    neighbor_ids: list[str],
    mgf_paths: list[str | Path] | None = None,
    seed_mgf: str | Path | None = None,
    peak_tol: float = 0.02,
) -> SpectralContext:
    """
    Load MGF file(s), resolve seed + neighbor spectra, compute MS/MS cosines.

    Parameters
    ----------
    mgf_paths:
        One or more MGF libraries (e.g. subgraph with NETWORK_NODE_ID).
    seed_mgf:
        Optional dedicated MGF for the query spectrum (e.g. Ego_MSMS.mgf).
    """
    ctx = SpectralContext()
    spectra: list[Spectrum] = []
    if seed_mgf:
        p = Path(seed_mgf)
        if p.exists():
            spectra.extend(parse_mgf(p))
            ctx.sources.append(str(p))
    for mp in mgf_paths or []:
        p = Path(mp)
        if p.exists():
            spectra.extend(parse_mgf(p))
            ctx.sources.append(str(p))
    if not spectra:
        return ctx

    idx = index_spectra(spectra)
    ctx.n_spectra_indexed = len(spectra)

    # Resolve seed spectrum: explicit seed mgf first (often single spectrum),
    # then node id, then pepmass.
    seed_sp: Spectrum | None = None
    if seed_mgf and Path(seed_mgf).exists():
        seed_only = parse_mgf(seed_mgf)
        if len(seed_only) == 1:
            seed_sp = seed_only[0]
        elif seed_only:
            # pick closest pepmass to seed_mz
            if seed_mz is not None:
                seed_sp = min(
                    seed_only,
                    key=lambda s: abs((s.pepmass or 0) - seed_mz)
                    if s.pepmass
                    else 1e9,
                )
            else:
                seed_sp = seed_only[0]
    if seed_sp is None:
        seed_sp = idx.get(str(seed_id)) or idx.get(f"id:{seed_id}")
    if seed_sp is None and seed_mz is not None:
        seed_sp = idx.get(f"mz:{seed_mz:.3f}") or idx.get(f"mz:{seed_mz:.2f}")
        if seed_sp is None:
            # nearest pepmass among all
            cands = [s for s in spectra if s.pepmass is not None]
            if cands:
                seed_sp = min(cands, key=lambda s: abs((s.pepmass or 0) - seed_mz))
                if abs((seed_sp.pepmass or 0) - seed_mz) > 0.05:
                    seed_sp = None

    ctx.seed = seed_sp
    if seed_sp:
        from ego_mol_llm.method_card import parse_ion_mode, parse_rt_seconds

        ctx.seed_meta = dict(seed_sp.meta or {})
        ctx.seed_rt = parse_rt_seconds(seed_sp.meta)
        ctx.seed_ion_mode = parse_ion_mode(seed_sp.meta)
    if seed_sp and seed_sp.peaks:
        from ego_mol_llm.method_card import parse_rt_seconds

        ctx.seed_diagnostics = diagnostic_ions(seed_sp.peaks, tol=peak_tol)
        ctx.seed_losses = neutral_losses(
            seed_sp.peaks, seed_sp.pepmass or seed_mz, top_n=12
        )
        for nid in neighbor_ids:
            nsp = idx.get(str(nid))
            if nsp is None:
                continue
            # Neighbor experimental metadata (study, RT, file, cluster size)
            meta = dict(nsp.meta or {})
            ctx.neighbor_meta[str(nid)] = {
                "rt": parse_rt_seconds(meta),
                "msv_lib": meta.get("MSV_LIB") or meta.get("LIBRARY"),
                "filename": meta.get("FILENAME"),
                "clustersize": meta.get("CLUSTERSIZE"),
                "ion_mode": meta.get("IONMODE") or meta.get("ION_MODE"),
                "pepmass": nsp.pepmass,
            }
            ctx.neighbor_rt[str(nid)] = parse_rt_seconds(meta)
            if nsp.peaks and seed_sp.peaks:
                ctx.neighbor_msms_cosine[str(nid)] = cosine_peaks(
                    seed_sp.peaks, nsp.peaks, tol=peak_tol
                )
                # keep peaks for offline neighbor differential (top-N only later)
                ctx.neighbor_peaks[str(nid)] = list(nsp.peaks)

        # Offline MS/MS explanation (always-on, no SIRIUS)
        from ego_mol_llm.msms_explain import build_msms_explanation

        top_nbs = sorted(
            ctx.neighbor_msms_cosine.items(), key=lambda x: -x[1]
        )[:3]
        neighbor_specs = []
        for nid, cos in top_nbs:
            meta = ctx.neighbor_meta.get(str(nid)) or {}
            neighbor_specs.append(
                {
                    "id": nid,
                    "name": None,
                    "peaks": ctx.neighbor_peaks.get(str(nid)) or [],
                    "mz": meta.get("pepmass"),
                    "msms_cosine": cos,
                }
            )
        exp = build_msms_explanation(
            seed_sp.peaks,
            seed_sp.pepmass or seed_mz,
            neighbor_spectra=neighbor_specs,
            max_neighbors=3,
        )
        ctx.explanation = exp.to_dict()
    return ctx
