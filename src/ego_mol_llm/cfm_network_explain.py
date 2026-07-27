"""
CFM-ID–first fragment explanation for ego networks (offline Docker).

Workflow (recommended for LLM prompts)
--------------------------------------
1. For each **annotated neighbor** with SMILES + experimental MS/MS:
   - ``cfm-annotate``: map experimental peaks → fragment ions (SMILES / mass)
   - ``cfm-predict``: predicted spectrum for structure↔spectrum cosine
2. For the **blind seed** (no structure):
   - transfer annotations when seed peaks match neighbor experimental peaks
   - merge with offline rule-based ``msms_explain``
3. LLM sees a structured block of peak chemistry **before** proposing SMILES.

This is independent of SIRIUS and does not require knowing the seed structure.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ego_mol_llm.mgf import cosine_peaks
from ego_mol_llm.spectrum_predict import (
    CfmIdDockerPredictor,
    RuleBasedPredictor,
    _default_adduct,
    canonicalize_smiles,
)


@dataclass
class PeakAnnotation:
    mz: float
    intensity: float
    fragment_id: int | None = None
    fragment_smiles: str | None = None
    fragment_mass: float | None = None
    score: float | None = None
    source: str = "cfmid_annotate"  # cfmid_annotate | transferred | rule | predict_match
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NodeCfmExplanation:
    node_id: str
    smiles: str | None
    name: str | None
    ion_mode: str | None
    adduct: str | None
    role: str  # seed | neighbor
    n_peaks: int = 0
    n_annotated: int = 0
    predict_cosine: float | None = None
    peak_annotations: list[PeakAnnotation] = field(default_factory=list)
    fragment_table: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["peak_annotations"] = [p.to_dict() for p in self.peak_annotations]
        return d


@dataclass
class EgoCfmExplanation:
    spectrum_id: str
    seed: NodeCfmExplanation | None = None
    neighbors: list[NodeCfmExplanation] = field(default_factory=list)
    seed_peak_map: list[PeakAnnotation] = field(default_factory=list)
    chemistry_summary: list[str] = field(default_factory=list)
    backend: str = "cfmid"
    product_version: str = "0.4-cfm-first"

    def to_dict(self) -> dict[str, Any]:
        return {
            "spectrum_id": self.spectrum_id,
            "seed": self.seed.to_dict() if self.seed else None,
            "neighbors": [n.to_dict() for n in self.neighbors],
            "seed_peak_map": [p.to_dict() for p in self.seed_peak_map],
            "chemistry_summary": self.chemistry_summary,
            "backend": self.backend,
            "product_version": self.product_version,
        }


def _docker_bin() -> str:
    return shutil.which("docker") or "docker"


def docker_available() -> bool:
    try:
        r = subprocess.run(
            [_docker_bin(), "info"], capture_output=True, text=True, timeout=15
        )
        return r.returncode == 0
    except Exception:
        return False


def _win_mount(path: Path) -> str:
    p = str(path.resolve()).replace("\\", "/")
    if len(p) > 1 and p[1] == ":":
        return f"/{p[0].lower()}{p[2:]}"
    return p


def write_cfm_energy_spectrum(
    path: Path,
    peaks: list[tuple[float, float]],
    *,
    top_n: int = 40,
) -> Path:
    """Write 3 energy levels (same peaks) for CFM-ID models that expect CE0/1/2."""
    tops = sorted(peaks, key=lambda x: -x[1])[:top_n]
    lines: list[str] = []
    for eng in ("energy0", "energy1", "energy2"):
        lines.append(eng)
        for mz, inten in tops:
            lines.append(f"{float(mz):.6f} {float(max(inten, 0.0)):.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def _parse_cfm_annotate(text: str) -> tuple[list[PeakAnnotation], list[dict[str, Any]]]:
    """
    Parse cfm-annotate output: energy blocks with optional fragment ids,
    then fragment table ``id mass smiles``.
    """
    frag_table: dict[int, dict[str, Any]] = {}
    # Fragment lines like: 2 76.0393048521 [NH2+]=CC(O)O
    frag_re = re.compile(
        r"^(\d+)\s+([0-9.]+)\s+(\S.*)$"
    )
    peak_re = re.compile(
        r"^([0-9.]+)\s+([0-9.]+)(?:\s+(\d+)(?:\s+\(([0-9.]+)\))?)?"
    )

    sections = re.split(r"(?im)^(energy\d+)\s*$", text)
    # sections: [preamble, energy0, body0, energy1, body1, ...]
    peak_anns: list[PeakAnnotation] = []
    current_energy = None
    for i, part in enumerate(sections):
        if re.match(r"(?i)^energy\d+$", part.strip()):
            current_energy = part.strip().lower()
            continue
        if current_energy is None:
            # may contain fragment table after peaks
            for line in part.splitlines():
                line = line.strip()
                m = frag_re.match(line)
                if m and not line.lower().startswith("target"):
                    fid = int(m.group(1))
                    # heuristic: fragment table has integer first and mass then smiles
                    # avoid counting peak lines
                    if " " in m.group(3) or any(c.isalpha() for c in m.group(3)):
                        try:
                            fmass = float(m.group(2))
                            # peak lines have only 2 floats sometimes; fragment has SMILES
                            smi = m.group(3).strip()
                            if re.search(r"[A-Za-z\[\]=#]", smi):
                                frag_table[fid] = {
                                    "id": fid,
                                    "mass": fmass,
                                    "smiles": smi,
                                }
                        except ValueError:
                            pass
            continue
        # peak body for this energy — prefer energy1
        if current_energy != "energy1" and peak_anns:
            # still parse fragment table at end outside
            for line in part.splitlines():
                line = line.strip()
                m = frag_re.match(line)
                if m:
                    smi = m.group(3).strip()
                    if re.search(r"[A-Za-z\[\]=#]", smi):
                        try:
                            frag_table[int(m.group(1))] = {
                                "id": int(m.group(1)),
                                "mass": float(m.group(2)),
                                "smiles": smi,
                            }
                        except ValueError:
                            pass
            continue
        for line in part.splitlines():
            line = line.strip()
            if not line or line.lower().startswith("target"):
                continue
            pm = peak_re.match(line)
            if not pm:
                # try fragment table line
                fm = frag_re.match(line)
                if fm and re.search(r"[A-Za-z\[\]=#]", fm.group(3)):
                    try:
                        frag_table[int(fm.group(1))] = {
                            "id": int(fm.group(1)),
                            "mass": float(fm.group(2)),
                            "smiles": fm.group(3).strip(),
                        }
                    except ValueError:
                        pass
                continue
            mz = float(pm.group(1))
            inten = float(pm.group(2))
            fid = int(pm.group(3)) if pm.group(3) else None
            sc = float(pm.group(4)) if pm.group(4) else None
            peak_anns.append(
                PeakAnnotation(
                    mz=mz,
                    intensity=inten,
                    fragment_id=fid,
                    score=sc,
                    source="cfmid_annotate",
                )
            )

    # second pass: full text fragment table (more reliable)
    # After blank line: "0 160.09 SMILES"
    in_table = False
    for line in text.splitlines():
        line = line.strip()
        if not line:
            in_table = True
            continue
        if re.match(r"(?i)^energy\d+$", line):
            in_table = False
            continue
        m = frag_re.match(line)
        if m and re.search(r"[CNOSPFIcnops\[\]()=#@+\\/-]", m.group(3)):
            try:
                fid = int(m.group(1))
                fmass = float(m.group(2))
                smi = m.group(3).strip()
                # skip pure numeric second tokens without chemistry chars already checked
                if fmass > 5:
                    frag_table[fid] = {"id": fid, "mass": fmass, "smiles": smi}
            except ValueError:
                pass

    # attach fragment smiles to peaks
    for p in peak_anns:
        if p.fragment_id is not None and p.fragment_id in frag_table:
            ft = frag_table[p.fragment_id]
            p.fragment_smiles = ft.get("smiles")
            p.fragment_mass = ft.get("mass")
            p.note = f"cfm frag#{p.fragment_id}"

    return peak_anns, list(frag_table.values())


def cfm_annotate_spectrum(
    smiles: str,
    peaks: list[tuple[float, float]],
    *,
    ion_mode: str | None = "positive",
    image: str = "wishartlab/cfmid:latest",
    timeout_s: float = 120.0,
    ppm: float = 10.0,
    abs_tol: float = 0.02,
) -> NodeCfmExplanation:
    """Run cfm-annotate in Docker for one structure + experimental peaks."""
    can = canonicalize_smiles(smiles) or smiles
    adduct = _default_adduct(ion_mode)
    if (ion_mode or "").lower().startswith("neg"):
        model = r"/trained_models_cfmid4.0/[M-H]-"
        adduct = "[M-H]-"
    else:
        model = r"/trained_models_cfmid4.0/[M+H]+"
        adduct = "[M+H]+"

    expl = NodeCfmExplanation(
        node_id="?",
        smiles=can,
        name=None,
        ion_mode=ion_mode,
        adduct=adduct,
        role="neighbor",
        n_peaks=len(peaks),
    )

    if not docker_available() or not peaks:
        expl.meta["error"] = "docker_unavailable_or_no_peaks"
        return expl

    with tempfile.TemporaryDirectory(prefix="cfm_ann_") as tmp:
        tdir = Path(tmp)
        spec = tdir / "spec.txt"
        write_cfm_energy_spectrum(spec, peaks)
        out = tdir / "out.txt"
        smi_q = can.replace("'", r"'\''")
        param = f"{model}/param_output.log"
        conf = f"{model}/param_config.txt"
        inner = (
            f"cfm-annotate '{smi_q}' /work/spec.txt SID {ppm} {abs_tol} "
            f"{param} {conf} /work/out.txt"
        )
        cmd = [
            _docker_bin(),
            "run",
            "--rm",
            "-v",
            f"{_win_mount(tdir)}:/work",
            image,
            "sh",
            "-c",
            inner,
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout_s
            )
        except Exception as e:
            expl.meta["error"] = f"{type(e).__name__}: {e}"
            return expl
        text = ""
        if out.is_file():
            text = out.read_text(encoding="utf-8", errors="replace")
        if not text:
            text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        expl.meta["returncode"] = proc.returncode
        anns, frags = _parse_cfm_annotate(text)
        expl.peak_annotations = anns
        expl.fragment_table = frags
        expl.n_annotated = sum(1 for a in anns if a.fragment_smiles)
    return expl


def explain_neighbor_with_cfm(
    *,
    node_id: str,
    smiles: str,
    name: str | None,
    peaks: list[tuple[float, float]],
    ion_mode: str | None,
    predictor: CfmIdDockerPredictor | None = None,
) -> NodeCfmExplanation:
    """Annotate + predict-cosine for one neighbor."""
    expl = cfm_annotate_spectrum(smiles, peaks, ion_mode=ion_mode)
    expl.node_id = str(node_id)
    expl.name = name
    expl.role = "neighbor"
    pred_eng = predictor or CfmIdDockerPredictor()
    if pred_eng.available() and peaks:
        pred = pred_eng.predict(smiles, ion_mode=ion_mode)
        if pred and pred.peaks:
            expl.predict_cosine = cosine_peaks(
                peaks, pred.peaks, tol=0.02, sqrt_intensity=True
            )
            expl.meta["predict_backend"] = pred.backend
    return expl


def transfer_seed_annotations(
    seed_peaks: list[tuple[float, float]],
    neighbor_expls: list[NodeCfmExplanation],
    *,
    tol: float = 0.02,
    top_neighbors: int = 5,
) -> list[PeakAnnotation]:
    """
    Map seed experimental peaks to fragment chemistry via shared m/z with
    annotated neighbors (structure known → CFM fragments → seed peak labels).
    """
    # rank neighbors by how many annotations they have / predict cosine
    ranked = sorted(
        neighbor_expls,
        key=lambda n: (
            -(n.predict_cosine or 0.0),
            -n.n_annotated,
        ),
    )[:top_neighbors]

    # index neighbor peak annotations by binned mz
    def bin_mz(mz: float) -> int:
        return int(round(mz / tol))

    neigh_bins: dict[int, list[tuple[NodeCfmExplanation, PeakAnnotation]]] = {}
    for ne in ranked:
        for pa in ne.peak_annotations:
            if not pa.fragment_smiles and pa.fragment_id is None:
                continue
            neigh_bins.setdefault(bin_mz(pa.mz), []).append((ne, pa))

    out: list[PeakAnnotation] = []
    base = max((i for _, i in seed_peaks), default=1.0) or 1.0
    for mz, inten in sorted(seed_peaks, key=lambda x: -x[1])[:40]:
        hits = neigh_bins.get(bin_mz(mz), [])
        if not hits:
            # try neighbors
            for d in (-1, 1):
                hits = neigh_bins.get(bin_mz(mz) + d, [])
                if hits:
                    break
        if not hits:
            out.append(
                PeakAnnotation(
                    mz=float(mz),
                    intensity=float(inten),
                    source="unexplained",
                    note="no CFM transfer from top neighbors",
                )
            )
            continue
        # pick best annotated hit
        ne, pa = hits[0]
        for cand_ne, cand_pa in hits:
            if cand_pa.fragment_smiles:
                ne, pa = cand_ne, cand_pa
                break
        out.append(
            PeakAnnotation(
                mz=float(mz),
                intensity=float(inten),
                fragment_id=pa.fragment_id,
                fragment_smiles=pa.fragment_smiles,
                fragment_mass=pa.fragment_mass,
                score=pa.score,
                source="transferred",
                note=(
                    f"shared with neighbor {ne.name or ne.node_id} "
                    f"(SMILES={ne.smiles}); CFM frag of annotated structure"
                ),
            )
        )
    return out


def build_ego_cfm_explanation(
    *,
    spectrum_id: str,
    seed_peaks: list[tuple[float, float]],
    seed_ion_mode: str | None,
    neighbors: list[dict[str, Any]],
    max_neighbors: int = 8,
    use_cfm: bool = True,
) -> EgoCfmExplanation:
    """
    neighbors: list of dicts with keys
      id, smiles, name, peaks, mz, msms_cosine, ion_mode
    """
    # pick top neighbors by msms_cosine then presence of smiles
    def nkey(n: dict) -> tuple:
        return (
            0 if n.get("smiles") else 1,
            -(float(n.get("msms_cosine") or 0.0)),
        )

    ranked = sorted(neighbors, key=nkey)[:max_neighbors]
    pred = CfmIdDockerPredictor() if use_cfm else None
    use_docker = use_cfm and docker_available() and pred is not None and pred.available()

    neigh_expls: list[NodeCfmExplanation] = []
    for n in ranked:
        smi = n.get("smiles")
        peaks = n.get("peaks") or []
        if not smi or not peaks:
            continue
        ion = n.get("ion_mode") or seed_ion_mode
        if use_docker:
            ne = explain_neighbor_with_cfm(
                node_id=str(n.get("id")),
                smiles=smi,
                name=n.get("name"),
                peaks=peaks,
                ion_mode=ion,
                predictor=pred,
            )
        else:
            # rule fallback: predict peaks from structure, match experimental
            rb = RuleBasedPredictor()
            pr = rb.predict(smi, ion_mode=ion)
            ne = NodeCfmExplanation(
                node_id=str(n.get("id")),
                smiles=canonicalize_smiles(smi) or smi,
                name=n.get("name"),
                ion_mode=ion,
                adduct=_default_adduct(ion),
                role="neighbor",
                n_peaks=len(peaks),
            )
            if pr and pr.peaks:
                ne.predict_cosine = cosine_peaks(peaks, pr.peaks, tol=0.02)
                for mz, inten in sorted(peaks, key=lambda x: -x[1])[:20]:
                    matched = any(abs(pmz - mz) <= 0.02 for pmz, _ in pr.peaks)
                    ne.peak_annotations.append(
                        PeakAnnotation(
                            mz=float(mz),
                            intensity=float(inten),
                            source="rule_predict_match" if matched else "unexplained",
                            note="rule-based structure prediction match"
                            if matched
                            else "",
                        )
                    )
                ne.n_annotated = sum(
                    1 for a in ne.peak_annotations if a.source == "rule_predict_match"
                )
            ne.meta["backend"] = "rule_fallback"
        neigh_expls.append(ne)

    seed_map = transfer_seed_annotations(seed_peaks, neigh_expls)
    # also rule-based labels for seed losses
    from ego_mol_llm.msms_explain import build_msms_explanation

    rule = build_msms_explanation(seed_peaks, None)
    summary: list[str] = []
    n_trans = sum(1 for p in seed_map if p.source == "transferred" and p.fragment_smiles)
    summary.append(
        f"Seed peaks with CFM-transferred fragment SMILES: {n_trans}/{len(seed_map)}"
    )
    for ne in neigh_expls[:5]:
        summary.append(
            f"Neighbor {ne.name or ne.node_id}: annotated={ne.n_annotated}/"
            f"{ne.n_peaks}, predict_cos={ne.predict_cosine}"
        )
    if rule.chemistry_hints:
        summary.extend([f"rule: {h}" for h in rule.chemistry_hints[:4]])

    seed_node = NodeCfmExplanation(
        node_id="0",
        smiles=None,
        name="SEED_UNKNOWN",
        ion_mode=seed_ion_mode,
        adduct=None,
        role="seed",
        n_peaks=len(seed_peaks),
        n_annotated=n_trans,
        peak_annotations=seed_map,
        meta={"note": "structure hidden; labels transferred from neighbors"},
    )

    return EgoCfmExplanation(
        spectrum_id=spectrum_id,
        seed=seed_node,
        neighbors=neigh_expls,
        seed_peak_map=seed_map,
        chemistry_summary=summary,
        backend="cfmid" if use_docker else "rule_fallback",
    )


def format_cfm_explanation_for_prompt(
    expl: EgoCfmExplanation | dict[str, Any],
    *,
    max_seed_peaks: int = 15,
    max_neighbors: int = 4,
) -> list[str]:
    if isinstance(expl, dict):
        # lightweight format without rehydrate
        lines = [
            "",
            "=== CFM-ID / IN-SILICO FRAGMENT EXPLANATION (precomputed; offline) ===",
            f"  backend = {expl.get('backend')}",
        ]
        for s in (expl.get("chemistry_summary") or [])[:8]:
            lines.append(f"  • {s}")
        seed_map = expl.get("seed_peak_map") or []
        lines.append("  SEED experimental peaks → fragment chemistry (from network):")
        for p in seed_map[:max_seed_peaks]:
            if not isinstance(p, dict):
                continue
            fs = p.get("fragment_smiles")
            if fs:
                lines.append(
                    f"    m/z {p.get('mz'):.4f} → {fs} "
                    f"[{p.get('source')}] {p.get('note','')[:80]}"
                )
            elif p.get("source") == "unexplained":
                continue  # skip clutter
        lines.append("  Neighbor structure annotations (CFM-ID):")
        for n in (expl.get("neighbors") or [])[:max_neighbors]:
            if not isinstance(n, dict):
                continue
            lines.append(
                f"    • {n.get('name') or n.get('node_id')}: SMILES={n.get('smiles')} "
                f"annotated={n.get('n_annotated')}/{n.get('n_peaks')} "
                f"predict_cos={n.get('predict_cosine')}"
            )
            # top annotated peaks
            shown = 0
            for pa in n.get("peak_annotations") or []:
                if not isinstance(pa, dict) or not pa.get("fragment_smiles"):
                    continue
                lines.append(
                    f"        peak {pa.get('mz'):.4f} → {pa.get('fragment_smiles')}"
                )
                shown += 1
                if shown >= 5:
                    break
        lines.append(
            "  Use transferred fragment SMILES as substructure clues for the unknown seed; "
            "prefer candidates that rationalize unexplained intense peaks."
        )
        return lines

    d = expl.to_dict()
    return format_cfm_explanation_for_prompt(d, max_seed_peaks=max_seed_peaks, max_neighbors=max_neighbors)


def cache_path(cache_dir: Path, spectrum_id: str) -> Path:
    return cache_dir / f"{spectrum_id}.json"
