"""
Spectral library search (NIST / custom MGF) for annotation propagation.

Large libraries (e.g. NIST2023 ~2.5GB) are streamed once into a precursor-binned
on-disk pickle index (top peaks only) for fast reverse search.
"""

from __future__ import annotations

import pickle
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from ego_mol_llm.mgf import cosine_peaks


@dataclass
class LibraryRecord:
    pepmass: float
    name: str | None = None
    smiles: str | None = None
    inchikey: str | None = None
    formula: str | None = None
    precursor_type: str | None = None
    ion_mode: str | None = None
    instrument: str | None = None
    spectrum_id: str | None = None
    top_peaks: list[tuple[float, float]] = field(default_factory=list)
    source_library: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LibraryHit:
    record: LibraryRecord
    match_score: float
    precursor_error_da: float
    rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = self.record.to_dict()
        d.update(
            {
                "match_score": self.match_score,
                "precursor_error_da": self.precursor_error_da,
                "rank": self.rank,
            }
        )
        return d


def _bin_key(mz: float, bin_da: float = 0.01) -> int:
    return int(round(mz / bin_da))


def _top_peaks(peaks: list[tuple[float, float]], n: int = 40) -> list[tuple[float, float]]:
    if not peaks:
        return []
    tops = sorted(peaks, key=lambda x: -x[1])[:n]
    # relative intensity scale optional; keep absolute for cosine_peaks √I path
    return [(float(m), float(i)) for m, i in tops]


def _ion_mode_from_meta(meta: dict[str, str]) -> str | None:
    v = (meta.get("ION_MODE") or meta.get("IONMODE") or "").strip().upper()
    if v in {"P", "POS", "POSITIVE", "+"}:
        return "positive"
    if v in {"N", "NEG", "NEGATIVE", "-"}:
        return "negative"
    return v.lower() or None


def polarity_from_adduct(adduct: str | None) -> str | None:
    """Infer positive/negative from precursor_type / adduct string."""
    if not adduct:
        return None
    a = str(adduct).strip().replace(" ", "")
    if not a:
        return None
    if a.endswith("-"):
        return "negative"
    if a.endswith("+"):
        return "positive"
    return None


def resolve_record_polarity(rec: LibraryRecord) -> str | None:
    """Prefer explicit ion_mode; fall back to adduct charge sign."""
    if rec.ion_mode in {"positive", "negative"}:
        return rec.ion_mode
    return polarity_from_adduct(rec.precursor_type)


def iter_mgf_records(
    path: str | Path,
    *,
    max_spectra: int | None = None,
    top_n_peaks: int = 40,
    source_library: str | None = None,
) -> Iterator[LibraryRecord]:
    """Stream MGF into compact LibraryRecords (does not hold full file)."""
    path = Path(path)
    src = source_library or path.stem
    # Stream by blocks without loading whole file into one string if possible
    # For simplicity and robustness use chunked read of BEGIN/END via parse in slices
    # Full parse_mgf loads entire text — 2.5GB is heavy. Use line iterator.
    meta: dict[str, str] = {}
    peaks: list[tuple[float, float]] = []
    in_ions = False
    n = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            u = line.strip()
            if not u:
                continue
            if re.match(r"(?i)BEGIN IONS", u):
                in_ions = True
                meta = {}
                peaks = []
                continue
            if re.match(r"(?i)END IONS", u):
                if in_ions and "PEPMASS" in meta:
                    try:
                        pep = float(meta["PEPMASS"].split()[0])
                    except ValueError:
                        pep = None
                    if pep is not None and peaks:
                        yield LibraryRecord(
                            pepmass=pep,
                            name=meta.get("NAME") or meta.get("COMPOUND"),
                            smiles=meta.get("SMILES") or meta.get("SMILE"),
                            inchikey=meta.get("INCHIKEY") or meta.get("INCHI_KEY"),
                            formula=meta.get("FORMULA") or meta.get("MOLECULAR_FORMULA"),
                            precursor_type=meta.get("PRECURSOR_TYPE") or meta.get("ADDUCT"),
                            ion_mode=_ion_mode_from_meta(meta),
                            instrument=meta.get("INSTRUMENT") or meta.get("INSTRUMENT_TYPE"),
                            spectrum_id=meta.get("SPECTRUMID") or meta.get("ID"),
                            top_peaks=_top_peaks(peaks, top_n_peaks),
                            source_library=src,
                        )
                        n += 1
                        if max_spectra is not None and n >= max_spectra:
                            return
                in_ions = False
                continue
            if not in_ions:
                continue
            if "=" in u and not (u[0].isdigit() or u.startswith(".")):
                k, v = u.split("=", 1)
                meta[k.strip().upper()] = v.strip()
            else:
                parts = re.split(r"[\s\t]+", u)
                if len(parts) >= 2:
                    try:
                        peaks.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        pass


class SpectralLibraryIndex:
    """Precursor-binned reverse spectral library."""

    def __init__(
        self,
        bins: dict[int, list[LibraryRecord]] | None = None,
        *,
        bin_da: float = 0.01,
        source_path: str | None = None,
        n_records: int = 0,
    ) -> None:
        self.bins = bins or {}
        self.bin_da = bin_da
        self.source_path = source_path
        self.n_records = n_records

    @classmethod
    def build_from_mgf(
        cls,
        mgf_path: str | Path,
        *,
        bin_da: float = 0.01,
        top_n_peaks: int = 40,
        max_spectra: int | None = None,
        progress_every: int = 50_000,
    ) -> "SpectralLibraryIndex":
        mgf_path = Path(mgf_path)
        bins: dict[int, list[LibraryRecord]] = {}
        n = 0
        for rec in iter_mgf_records(
            mgf_path, max_spectra=max_spectra, top_n_peaks=top_n_peaks
        ):
            k = _bin_key(rec.pepmass, bin_da)
            bins.setdefault(k, []).append(rec)
            n += 1
            if progress_every and n % progress_every == 0:
                print(f"[library_index] indexed {n} spectra…", flush=True)
        print(f"[library_index] done n={n} bins={len(bins)} from {mgf_path}", flush=True)
        return cls(bins=bins, bin_da=bin_da, source_path=str(mgf_path), n_records=n)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {
                    "bins": self.bins,
                    "bin_da": self.bin_da,
                    "source_path": self.source_path,
                    "n_records": self.n_records,
                    "version": 2,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SpectralLibraryIndex":
        path = Path(path)
        with path.open("rb") as f:
            data = pickle.load(f)
        return cls(
            bins=data["bins"],
            bin_da=float(data.get("bin_da") or 0.01),
            source_path=data.get("source_path"),
            n_records=int(data.get("n_records") or 0),
        )

    def _candidate_records(
        self, precursor_mz: float, precursor_tol_da: float
    ) -> Iterable[LibraryRecord]:
        lo = precursor_mz - precursor_tol_da
        hi = precursor_mz + precursor_tol_da
        k0 = _bin_key(lo, self.bin_da)
        k1 = _bin_key(hi, self.bin_da)
        for k in range(k0 - 1, k1 + 2):
            for rec in self.bins.get(k, []):
                if lo <= rec.pepmass <= hi:
                    yield rec

    def search(
        self,
        peaks: list[tuple[float, float]],
        precursor_mz: float | None,
        *,
        top_k: int = 10,
        precursor_tol_da: float = 0.02,
        peak_tol: float = 0.02,
        ion_mode: str | None = None,
        min_score: float = 0.35,
    ) -> list[LibraryHit]:
        if not peaks or precursor_mz is None:
            return []
        hits: list[LibraryHit] = []
        want = (ion_mode or "").strip().lower() or None
        if want and want.startswith("pos"):
            want = "positive"
        elif want and want.startswith("neg"):
            want = "negative"
        for rec in self._candidate_records(float(precursor_mz), precursor_tol_da):
            rec_pol = resolve_record_polarity(rec)
            if want in {"positive", "negative"} and rec_pol in {"positive", "negative"}:
                if rec_pol != want:
                    continue
            score = cosine_peaks(peaks, rec.top_peaks, tol=peak_tol, sqrt_intensity=True)
            if score < min_score:
                continue
            err = abs(float(precursor_mz) - rec.pepmass)
            hits.append(
                LibraryHit(
                    record=rec,
                    match_score=score,
                    precursor_error_da=err,
                )
            )
        hits.sort(key=lambda h: (-h.match_score, h.precursor_error_da))
        out = hits[:top_k]
        for i, h in enumerate(out, 1):
            h.rank = i
        return out


# Process-level cache
_INDEX_CACHE: dict[str, SpectralLibraryIndex] = {}


def load_library_index(path: str | Path) -> SpectralLibraryIndex:
    key = str(Path(path).resolve())
    if key not in _INDEX_CACHE:
        _INDEX_CACHE[key] = SpectralLibraryIndex.load(path)
    return _INDEX_CACHE[key]


def default_nist_paths(repo_root: Path | None = None) -> tuple[Path, Path]:
    """Return (mgf_path, index_path) defaults under repo libraries/."""
    root = repo_root or Path(__file__).resolve().parents[2]
    lib = root / "libraries"
    mgf = lib / "LEVEL2_NIST2023MSMS_20240408.mgf"
    idx = lib / "LEVEL2_NIST2023MSMS_20240408.index.pkl"
    return mgf, idx


def ensure_nist_index(
    mgf_path: str | Path | None = None,
    index_path: str | Path | None = None,
    *,
    rebuild: bool = False,
    max_spectra: int | None = None,
) -> Path | None:
    """
    Ensure pickle index exists. Returns index path or None if MGF missing.
    Building full NIST may take many minutes.
    """
    default_mgf, default_idx = default_nist_paths()
    mgf_path = Path(mgf_path) if mgf_path else default_mgf
    index_path = Path(index_path) if index_path else default_idx
    if not mgf_path.is_file():
        print(f"[library] MGF not found: {mgf_path}", flush=True)
        return None
    if index_path.is_file() and not rebuild:
        return index_path
    print(f"[library] building index from {mgf_path} → {index_path}", flush=True)
    idx = SpectralLibraryIndex.build_from_mgf(mgf_path, max_spectra=max_spectra)
    idx.save(index_path)
    return index_path
