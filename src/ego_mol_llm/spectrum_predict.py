"""
In-silico MS/MS spectrum prediction and candidate re-ranking (offline).

Backends (optional → always-available fallback):
  1. **cfmid** — Wishart CFM-ID 4 via Docker (``wishartlab/cfmid``)
  2. **iceberg** — Coley lab ms-pred / ICEBERG if importable
  3. **rule**   — RDKit structure-aware fragment/loss model (default, no deps)

Re-ranking compares predicted peaks to the experimental seed spectrum with
``cosine_peaks`` and boosts hybrid fusion scores.

SIRIUS is unrelated: this scores *your* candidate SMILES against the measured
spectrum, which is what annotation propagation needs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ego_mol_llm.mgf import cosine_peaks
from ego_mol_llm.validate import canonicalize_smiles

# ---------------------------------------------------------------------------
# SMARTS → diagnostic fragment m/z (positive-mode biased)
# ---------------------------------------------------------------------------
_SMARTS_FRAGS: list[tuple[str, list[tuple[float, float]], str]] = [
    # (smarts, [(mz, relative_intensity), ...], label)
    ("c1ccc(cc1)C", [(91.054, 0.6), (120.081, 0.4)], "benzyl/Phe-like"),
    ("c1ccc(cc1)O", [(93.034, 0.5), (109.029, 0.3)], "phenol"),
    ("c1ccc(cc1)N", [(94.065, 0.4)], "aniline"),
    ("C(=O)O", [(45.0, 0.2)], "carboxyl"),
    ("NCC(=O)O", [(76.039, 0.9), (30.034, 0.3)], "glycine"),
    ("NC(C)C(=O)O", [(44.05, 0.4)], "alanine-like"),
    ("N1CCCC1", [(70.065, 0.5)], "proline-like"),
    ("c1c[nH]c2ccccc12", [(130.065, 0.4), (159.092, 0.3)], "indole/Trp"),
    ("C[N+](C)(C)C", [(60.081, 0.7), (58.065, 0.3)], "choline/TMA"),
    ("OP(=O)(O)O", [(98.984, 0.4), (80.974, 0.3)], "phosphate"),
    ("S(=O)(=O)O", [(79.957, 0.5), (96.960, 0.3)], "sulfate"),
    ("C1OC(CO)C(O)C(O)C1O", [(163.060, 0.5), (145.050, 0.3)], "hexose"),
]

_COMMON_LOSS_MASSES: list[tuple[str, float, float]] = [
    # label, loss_da, relative intensity weight
    ("H2O", 18.0106, 0.7),
    ("NH3", 17.0265, 0.4),
    ("CO", 27.9949, 0.35),
    ("CO2", 43.9898, 0.55),
    ("HCOOH", 46.0055, 0.45),
    ("CH2O", 30.0106, 0.3),
    ("C2H2O", 42.0106, 0.35),
    ("C2H4", 28.0313, 0.25),
    ("C3H6", 42.0470, 0.25),
    ("C4H8", 56.0626, 0.3),
    ("C2H5NO2", 75.0320, 0.5),  # glycine
    ("SO3", 79.9568, 0.45),
    ("H3PO4", 97.9769, 0.4),
    ("C6H10O5", 162.0528, 0.4),
]


@dataclass
class PredictedSpectrum:
    smiles: str
    adduct: str
    peaks: list[tuple[float, float]]
    backend: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "smiles": self.smiles,
            "adduct": self.adduct,
            "peaks": self.peaks,
            "backend": self.backend,
            "meta": self.meta,
        }


class SpectrumPredictor(ABC):
    name: str = "base"

    @abstractmethod
    def predict(
        self,
        smiles: str,
        *,
        adduct: str | None = None,
        ion_mode: str | None = None,
    ) -> PredictedSpectrum | None:
        ...

    def available(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Rule-based (default, always offline)
# ---------------------------------------------------------------------------
class RuleBasedPredictor(SpectrumPredictor):
    """
    Structure-aware heuristic spectrum:
      - precursor ion from exact mass + adduct
      - common neutral losses (weighted)
      - SMARTS-matched diagnostic fragments
    Fast; weaker than CFM-ID/ICEBERG but useful for re-ranking isomers.
    """

    name = "rule"

    def predict(
        self,
        smiles: str,
        *,
        adduct: str | None = None,
        ion_mode: str | None = None,
    ) -> PredictedSpectrum | None:
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors
        except Exception:
            return None
        can = canonicalize_smiles(smiles) or smiles
        mol = Chem.MolFromSmiles(can)
        if mol is None:
            return None
        exact = float(Descriptors.ExactMolWt(mol))
        adduct = adduct or _default_adduct(ion_mode)
        prec = _precursor_mz(exact, adduct, ion_mode)
        if prec is None or prec <= 0:
            return None

        peaks: dict[float, float] = {round(prec, 4): 100.0}

        # Neutral losses from precursor
        for _lab, loss, w in _COMMON_LOSS_MASSES:
            mz = prec - loss
            if mz > 30:
                peaks[round(mz, 4)] = max(peaks.get(round(mz, 4), 0.0), 100.0 * w)

        # Double water etc.
        for n in (2, 3):
            mz = prec - n * 18.0106
            if mz > 30:
                peaks[round(mz, 4)] = max(peaks.get(round(mz, 4), 0.0), 40.0 / n)

        # SMARTS diagnostics
        for smarts, frags, _lab in _SMARTS_FRAGS:
            patt = Chem.MolFromSmarts(smarts)
            if patt is None:
                continue
            if mol.HasSubstructMatch(patt):
                for fmz, rel in frags:
                    if fmz < prec + 5:
                        peaks[round(fmz, 4)] = max(
                            peaks.get(round(fmz, 4), 0.0), 100.0 * rel
                        )

        # Side-chain / formula-simple: add [M+H-side] style using exact - large bits
        # Keep top peaks only
        items = sorted(peaks.items(), key=lambda x: -x[1])[:40]
        return PredictedSpectrum(
            smiles=can,
            adduct=adduct,
            peaks=items,
            backend=self.name,
            meta={"exact_mass": exact, "precursor_mz": prec},
        )


def _default_adduct(ion_mode: str | None) -> str:
    mode = (ion_mode or "positive").lower()
    if mode.startswith("neg"):
        return "[M-H]-"
    return "[M+H]+"


def _precursor_mz(exact: float, adduct: str | None, ion_mode: str | None) -> float | None:
    a = (adduct or _default_adduct(ion_mode)).replace(" ", "")
    # common ESI adducts (neutral monomer mass → ion m/z)
    table = {
        "[M+H]+": exact + 1.007276,
        "[M-H]-": exact - 1.007276,
        "[M+Na]+": exact + 22.989218,
        "[M+K]+": exact + 38.963158,
        "[M+NH4]+": exact + 18.033823,
        "[M+H-H2O]+": exact + 1.007276 - 18.010565,
        "[M-H2O-H]-": exact - 1.007276 - 18.010565,
        "[2M+H]+": 2 * exact + 1.007276,
        "[2M-H]-": 2 * exact - 1.007276,
        "[M]+": exact - 0.000548,  # odd-electron approx
        "[M]-": exact + 0.000548,
    }
    if a in table:
        return table[a]
    # fuzzy
    al = a.lower()
    if "2m" in al and "+" in a:
        return 2 * exact + 1.007276
    if "2m" in al and "-" in a:
        return 2 * exact - 1.007276
    if "+na" in al:
        return exact + 22.989218
    if "+h" in al or a.endswith("+"):
        return exact + 1.007276
    if "-h" in al or a.endswith("-"):
        return exact - 1.007276
    return exact + 1.007276


# ---------------------------------------------------------------------------
# CFM-ID via Docker
# ---------------------------------------------------------------------------
class CfmIdDockerPredictor(SpectrumPredictor):
    """
    CFM-ID 4 spectrum prediction through Docker image ``wishartlab/cfmid``.

    Requires Docker daemon running. First call may pull ~200MB image.
    """

    name = "cfmid"

    def __init__(
        self,
        *,
        image: str = "wishartlab/cfmid:latest",
        docker_bin: str | None = None,
        timeout_s: float = 120.0,
        cache_dir: str | Path | None = None,
        energy: str = "energy1",  # med CE
    ) -> None:
        self.image = image
        self.docker_bin = docker_bin or shutil.which("docker") or "docker"
        self.timeout_s = timeout_s
        self.energy = energy
        self.cache_dir = Path(cache_dir) if cache_dir else Path(
            os.environ.get("EGO_MOL_CFMID_CACHE", Path.home() / ".cache" / "ego_mol_cfmid")
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._docker_ok: bool | None = None

    def available(self) -> bool:
        if self._docker_ok is not None:
            return self._docker_ok
        try:
            r = subprocess.run(
                [self.docker_bin, "info"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            self._docker_ok = r.returncode == 0
        except Exception:
            self._docker_ok = False
        return self._docker_ok

    def _cache_key(self, smiles: str, adduct: str) -> Path:
        h = hashlib.sha1(f"{smiles}|{adduct}|{self.energy}".encode()).hexdigest()[:16]
        return self.cache_dir / f"{h}.json"

    def predict(
        self,
        smiles: str,
        *,
        adduct: str | None = None,
        ion_mode: str | None = None,
    ) -> PredictedSpectrum | None:
        if not self.available():
            return None
        can = canonicalize_smiles(smiles) or smiles
        adduct = adduct or _default_adduct(ion_mode)
        # map adduct → CFM model folder
        if adduct.endswith("-") or (ion_mode or "").lower().startswith("neg"):
            model = r"/trained_models_cfmid4.0/[M-H]-"
            adduct_use = "[M-H]-"
        else:
            model = r"/trained_models_cfmid4.0/[M+H]+"
            adduct_use = "[M+H]+"

        cpath = self._cache_key(can, adduct_use)
        if cpath.is_file():
            try:
                d = json.loads(cpath.read_text(encoding="utf-8"))
                return PredictedSpectrum(
                    smiles=can,
                    adduct=adduct_use,
                    peaks=[(float(a), float(b)) for a, b in d["peaks"]],
                    backend=self.name,
                    meta={"cache": str(cpath)},
                )
            except Exception:
                pass

        # escape smiles for shell inside container
        smi_q = can.replace("'", r"'\''")
        param = f"{model}/param_output.log"
        conf = f"{model}/param_config.txt"
        # run cfm-predict, capture stdout
        inner = (
            f"cfm-predict '{smi_q}' 0.001 {param} {conf} 0 /dev/stdout 1 1"
        )
        cmd = [
            self.docker_bin,
            "run",
            "--rm",
            self.image,
            "sh",
            "-c",
            inner,
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except Exception as e:
            return None
        if proc.returncode != 0:
            return None
        peaks = _parse_cfm_predict_stdout(proc.stdout or "", energy=self.energy)
        if not peaks:
            # try merge all energies
            peaks = _parse_cfm_predict_stdout(proc.stdout or "", energy=None)
        if not peaks:
            return None
        try:
            cpath.write_text(
                json.dumps({"smiles": can, "adduct": adduct_use, "peaks": peaks}),
                encoding="utf-8",
            )
        except Exception:
            pass
        return PredictedSpectrum(
            smiles=can,
            adduct=adduct_use,
            peaks=peaks,
            backend=self.name,
            meta={"returncode": proc.returncode},
        )


def _parse_cfm_predict_stdout(
    text: str, *, energy: str | None = "energy1"
) -> list[tuple[float, float]]:
    """
    Parse cfm-predict multi-energy output::

        energy0
        15.02 0.03
        ...
        energy1
        ...
    """
    current = None
    buckets: dict[str, list[tuple[float, float]]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.match(r"(?i)^energy\d+$", line) or line.lower() in {"low", "med", "high"}:
            current = line.lower().replace("medium", "med")
            # map low/med/high → energy0/1/2
            if current == "low":
                current = "energy0"
            elif current in {"med", "medium"}:
                current = "energy1"
            elif current == "high":
                current = "energy2"
            buckets.setdefault(current, [])
            continue
        parts = line.split()
        if len(parts) >= 2 and current is not None:
            try:
                mz = float(parts[0])
                inten = float(parts[1])
            except ValueError:
                continue
            if inten > 0 and mz > 0:
                buckets[current].append((mz, inten))
    if energy and energy in buckets and buckets[energy]:
        return buckets[energy]
    # merge all
    merged: dict[float, float] = {}
    for peaks in buckets.values():
        for mz, inten in peaks:
            k = round(mz, 4)
            merged[k] = max(merged.get(k, 0.0), inten)
    return sorted(merged.items(), key=lambda x: -x[1])[:50]


# ---------------------------------------------------------------------------
# ICEBERG / ms-pred (optional import)
# ---------------------------------------------------------------------------
class IcebergPredictor(SpectrumPredictor):
    """
    Placeholder backend for Coley lab ICEBERG / ``ms_pred``.

    Install separately (GPU recommended). If not importable, ``available()`` is False.
    """

    name = "iceberg"

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self._mod = None
        try:
            import ms_pred  # type: ignore  # noqa: F401

            self._mod = True
        except Exception:
            self._mod = None

    def available(self) -> bool:
        return self._mod is not None

    def predict(
        self,
        smiles: str,
        *,
        adduct: str | None = None,
        ion_mode: str | None = None,
    ) -> PredictedSpectrum | None:
        # Full ICEBERG wiring depends on local checkpoint layout; keep explicit.
        # Users can subclass or extend once models are installed.
        if not self.available():
            return None
        return None  # force fallback until checkpoints configured


# ---------------------------------------------------------------------------
# Factory + re-rank
# ---------------------------------------------------------------------------
def get_predictor(
    backend: str = "auto",
    **kwargs: Any,
) -> SpectrumPredictor:
    """
    backend: auto | rule | cfmid | iceberg

    auto prefers cfmid if Docker works, else rule.
    """
    b = (backend or "auto").lower()
    if b == "rule":
        return RuleBasedPredictor()
    if b == "cfmid":
        return CfmIdDockerPredictor(**kwargs)
    if b == "iceberg":
        return IcebergPredictor(**kwargs)
    if b == "auto":
        cfm = CfmIdDockerPredictor(**kwargs)
        if cfm.available():
            return cfm
        ice = IcebergPredictor(**kwargs)
        if ice.available():
            return ice
        return RuleBasedPredictor()
    raise ValueError(f"Unknown spectrum predictor backend: {backend}")


def score_smiles_against_spectrum(
    smiles: str,
    experimental_peaks: list[tuple[float, float]],
    *,
    predictor: SpectrumPredictor | None = None,
    adduct: str | None = None,
    ion_mode: str | None = None,
    peak_tol: float = 0.02,
) -> tuple[float, PredictedSpectrum | None]:
    """Return (cosine 0–1, predicted spectrum)."""
    pred_eng = predictor or RuleBasedPredictor()
    pred = pred_eng.predict(smiles, adduct=adduct, ion_mode=ion_mode)
    if pred is None or not pred.peaks or not experimental_peaks:
        return 0.0, pred
    cos = cosine_peaks(experimental_peaks, pred.peaks, tol=peak_tol, sqrt_intensity=True)
    return float(cos), pred


def rerank_candidates_by_insilico(
    candidates: list[Any],
    experimental_peaks: list[tuple[float, float]],
    *,
    predictor: SpectrumPredictor | None = None,
    ion_mode: str | None = None,
    peak_tol: float = 0.02,
    weight: float = 0.35,
    cache: dict[str, float] | None = None,
) -> list[Any]:
    """
    Boost ``fusion_score`` of AnnotationCandidate-like objects by predicted-spectrum cosine.

    ``fusion_score <- (1-weight)*fusion + weight*insilico_cosine``
    Sets ``insilico_cosine`` attribute / meta when present.
    """
    pred_eng = predictor or get_predictor("auto")
    cache = cache if cache is not None else {}
    out = []
    for c in candidates:
        smi = getattr(c, "smiles", None)
        if not smi:
            out.append(c)
            continue
        key = canonicalize_smiles(smi) or smi
        adduct = getattr(c, "adduct", None)
        if key in cache:
            cos = cache[key]
            pred = None
        else:
            cos, pred = score_smiles_against_spectrum(
                smi,
                experimental_peaks,
                predictor=pred_eng,
                adduct=adduct,
                ion_mode=ion_mode,
                peak_tol=peak_tol,
            )
            cache[key] = cos
        # write fields
        if hasattr(c, "meta") and isinstance(c.meta, dict):
            c.meta["insilico_cosine"] = cos
            c.meta["insilico_backend"] = pred_eng.name
            if pred is not None:
                c.meta["insilico_n_peaks"] = len(pred.peaks)
        # optional typed field
        try:
            c.insilico_cosine = cos  # type: ignore[attr-defined]
        except Exception:
            pass
        old = float(getattr(c, "fusion_score", 0.0) or 0.0)
        c.fusion_score = (1.0 - weight) * old + weight * cos
        note = getattr(c, "note", "") or ""
        if "insilico=" not in note:
            c.note = (note + f" | insilico={cos:.3f}[{pred_eng.name}]").strip(" |")
        out.append(c)

    out.sort(
        key=lambda x: (
            0 if getattr(x, "mass_ok", None) is True else 1,
            -(getattr(x, "fusion_score", 0.0) or 0.0),
        )
    )
    return out
