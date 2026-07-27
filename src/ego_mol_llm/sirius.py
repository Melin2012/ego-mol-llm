"""
SIRIUS + CSI:FingerID integration for annotation propagation.

Pipeline role
-------------
Independent spectral structure evidence (formula tree + CSI:FingerID ranks)
alongside ego neighbors and NIST reverse search.

Requirements
------------
- Local SIRIUS 5/6 CLI on PATH or via ``SIRIUS_BIN`` / ``--sirius-bin``
- Academic/non-commercial login for CSI:FingerID web services
  (commercial use: Bright Giant license — not bundled here)

CLI pattern (SIRIUS 5/6-style)::

    sirius login
    sirius -i spectrum.ms -o project.sirius \\
        formula -p orbitrap structure \\
        write-summaries --output summaries/

Outputs are parsed from TSV/CSV under the project or summaries folder.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class SiriusHit:
    """One structure (or formula-only) identification from SIRIUS/CSI:FingerID."""

    rank: int = 0
    smiles: str | None = None
    inchikey: str | None = None
    name: str | None = None
    formula: str | None = None
    adduct: str | None = None
    csi_score: float | None = None
    confidence: float | None = None
    zodiac_score: float | None = None
    formula_score: float | None = None
    source: str = "sirius"  # sirius | csi_fingerid | formula
    note: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def find_sirius_binary(explicit: str | Path | None = None) -> Path | None:
    """Resolve SIRIUS executable path."""
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
    env = os.environ.get("SIRIUS_BIN") or os.environ.get("SIRIUS_PATH")
    if env:
        p = Path(env)
        if p.is_file():
            return p
        # directory containing sirius / sirius.bat
        for name in ("sirius.bat", "sirius.exe", "sirius", "sirius.cmd"):
            cand = p / name
            if cand.is_file():
                return cand
    which = shutil.which("sirius") or shutil.which("sirius.bat") or shutil.which("sirius.exe")
    if which:
        return Path(which)
    return None


def ionization_for_mode(ion_mode: str | None, charge: int | None = None) -> str:
    """Default SIRIUS ionization string from polarity."""
    mode = (ion_mode or "").lower()
    if mode.startswith("neg") or (charge is not None and charge < 0):
        return "[M-H]-"
    if mode.startswith("pos") or (charge is not None and charge > 0):
        return "[M+H]+"
    return "[M+H]+"


def write_sirius_ms(
    path: str | Path,
    *,
    compound_id: str,
    precursor_mz: float,
    peaks: list[tuple[float, float]],
    ion_mode: str | None = None,
    ionization: str | None = None,
    collision_energy: str | None = None,
    ms1_peaks: list[tuple[float, float]] | None = None,
) -> Path:
    """
    Write a SIRIUS ``.ms`` spectrum file.

    Format follows the classic SIRIUS compound block (compatible with v4–v6).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ion = ionization or ionization_for_mode(ion_mode)
    lines = [
        f">compound {compound_id}",
        f">parentmass {float(precursor_mz):.6f}",
        f">ionization {ion}",
    ]
    if collision_energy:
        lines.append(f">collision {collision_energy}")
    if ms1_peaks:
        lines.append(">ms1")
        for mz, inten in ms1_peaks:
            lines.append(f"{float(mz):.6f} {float(inten):.4f}")
    lines.append(">ms2")
    # SIRIUS prefers absolute intensities; keep relative if that is all we have
    tops = sorted(peaks, key=lambda x: -x[1])[:200]
    for mz, inten in tops:
        lines.append(f"{float(mz):.6f} {float(max(inten, 0.0)):.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_sirius_ms_from_mgf_spectrum(
    path: str | Path,
    *,
    compound_id: str,
    pepmass: float,
    peaks: list[tuple[float, float]],
    ion_mode: str | None = None,
) -> Path:
    return write_sirius_ms(
        path,
        compound_id=compound_id,
        precursor_mz=pepmass,
        peaks=peaks,
        ion_mode=ion_mode,
    )


def _float_or_none(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _pick(row: dict[str, str], *keys: str) -> str | None:
    lower = {k.lower(): k for k in row}
    for key in keys:
        k = lower.get(key.lower())
        if k is not None and str(row[k]).strip():
            return str(row[k]).strip()
    return None


def parse_structure_tsv(path: str | Path, *, top_k: int = 10) -> list[SiriusHit]:
    """Parse SIRIUS structure_identifications.tsv / .csv."""
    path = Path(path)
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    # dialect sniff
    sample = text[:2048]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,;")
    except csv.Error:
        dialect = csv.excel_tab if "\t" in sample else csv.excel
    reader = csv.DictReader(text.splitlines(), dialect=dialect)
    hits: list[SiriusHit] = []
    for i, row in enumerate(reader, 1):
        if i > top_k * 3:
            break
        smi = _pick(row, "smiles", "SMILES", "opt_smiles")
        ik = _pick(row, "InChIKey", "inchikey", "InChIKey2D", "InChIkey2D")
        name = _pick(row, "name", "Name", "compoundName", "pubchemname")
        formula = _pick(
            row, "molecularFormula", "formula", "MolecularFormula", "precursorFormula"
        )
        adduct = _pick(row, "adduct", "Adduct", "ion", "ionization")
        csi = _float_or_none(
            _pick(row, "CSI:FingerIDScore", "ConfidenceScore", "score", "SiriusScore")
        )
        conf = _float_or_none(
            _pick(row, "ConfidenceScore", "confidence", "Confidence", "ZodiacScore")
        )
        rank_s = _pick(row, "rank", "Rank", "structureRank")
        try:
            rank = int(float(rank_s)) if rank_s else len(hits) + 1
        except ValueError:
            rank = len(hits) + 1
        if not smi and not formula:
            continue
        hits.append(
            SiriusHit(
                rank=rank,
                smiles=smi,
                inchikey=ik,
                name=name,
                formula=formula,
                adduct=adduct,
                csi_score=csi,
                confidence=conf,
                source="csi_fingerid" if smi else "formula",
                note=f"SIRIUS/CSI rank={rank}",
                meta={k: v for k, v in row.items() if v},
            )
        )
        if len(hits) >= top_k:
            break
    return hits


def parse_formula_tsv(path: str | Path, *, top_k: int = 5) -> list[SiriusHit]:
    """Parse formula_identifications when structure step is unavailable."""
    path = Path(path)
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(text[:2048], delimiters="\t,;")
    except csv.Error:
        dialect = csv.excel_tab if "\t" in text[:200] else csv.excel
    reader = csv.DictReader(text.splitlines(), dialect=dialect)
    hits: list[SiriusHit] = []
    for i, row in enumerate(reader, 1):
        if len(hits) >= top_k:
            break
        formula = _pick(row, "molecularFormula", "formula", "MolecularFormula")
        if not formula:
            continue
        adduct = _pick(row, "adduct", "Adduct", "ionization")
        score = _float_or_none(_pick(row, "SiriusScore", "ZodiacScore", "score", "Score"))
        hits.append(
            SiriusHit(
                rank=i,
                formula=formula,
                adduct=adduct,
                formula_score=score,
                source="formula",
                note=f"SIRIUS formula rank={i}",
                meta={k: v for k, v in row.items() if v},
            )
        )
    return hits


def discover_summary_files(root: str | Path) -> dict[str, Path]:
    """Find structure/formula summary tables under a SIRIUS project or summaries dir."""
    root = Path(root)
    found: dict[str, Path] = {}
    if not root.exists():
        return found
    patterns = {
        "structure": (
            "structure_identifications.tsv",
            "structure_identifications.csv",
            "compound_identifications.tsv",
            "compound_identifications.csv",
            "structure_candidates.tsv",
        ),
        "formula": (
            "formula_identifications.tsv",
            "formula_identifications.csv",
            "formula_candidates.tsv",
        ),
    }
    files = list(root.rglob("*")) if root.is_dir() else []
    names = {p.name.lower(): p for p in files if p.is_file()}
    for kind, candidates in patterns.items():
        for name in candidates:
            p = names.get(name.lower())
            if p is not None:
                found[kind] = p
                break
    return found


def parse_sirius_project(project_dir: str | Path, *, top_k: int = 10) -> list[SiriusHit]:
    """Load best available structure (else formula) hits from a SIRIUS output tree."""
    project_dir = Path(project_dir)
    found = discover_summary_files(project_dir)
    if "structure" in found:
        hits = parse_structure_tsv(found["structure"], top_k=top_k)
        if hits:
            return hits
    if "formula" in found:
        return parse_formula_tsv(found["formula"], top_k=min(top_k, 5))
    # Some runs write JSON summaries
    for jp in project_dir.rglob("*.json"):
        if "structure" in jp.name.lower() or "identification" in jp.name.lower():
            try:
                data = json.loads(jp.read_text(encoding="utf-8"))
            except Exception:
                continue
            hits = _parse_structure_json(data, top_k=top_k)
            if hits:
                return hits
    return []


def _parse_structure_json(data: Any, *, top_k: int = 10) -> list[SiriusHit]:
    rows: list[dict] = []
    if isinstance(data, list):
        rows = [x for x in data if isinstance(x, dict)]
    elif isinstance(data, dict):
        for key in ("structures", "candidates", "results", "identifications"):
            if isinstance(data.get(key), list):
                rows = [x for x in data[key] if isinstance(x, dict)]
                break
        if not rows and ("smiles" in data or "SMILES" in data):
            rows = [data]
    hits: list[SiriusHit] = []
    for i, row in enumerate(rows[:top_k], 1):
        smi = row.get("smiles") or row.get("SMILES")
        hits.append(
            SiriusHit(
                rank=int(row.get("rank") or i),
                smiles=smi,
                inchikey=row.get("InChIKey") or row.get("inchikey"),
                name=row.get("name"),
                formula=row.get("molecularFormula") or row.get("formula"),
                adduct=row.get("adduct"),
                csi_score=_float_or_none(row.get("CSI:FingerIDScore") or row.get("score")),
                confidence=_float_or_none(row.get("ConfidenceScore") or row.get("confidence")),
                source="csi_fingerid" if smi else "formula",
                note=f"SIRIUS JSON rank={i}",
                meta={k: v for k, v in row.items() if not isinstance(v, (dict, list))},
            )
        )
    return hits


def build_sirius_command(
    *,
    sirius_bin: Path,
    ms_path: Path,
    project_dir: Path,
    summaries_dir: Path | None = None,
    profile: str = "orbitrap",
    max_mz: float | None = None,
    no_structure: bool = False,
    structure_db: str = "BIO",
    extra_args: list[str] | None = None,
) -> list[str]:
    """
    Build a SIRIUS 6-style CLI invocation.

    SIRIUS 6 subcommands: ``formulas`` → ``structures`` (CSI:FingerID) → ``summaries``.
    (Older docs said formula/structure/write-summaries; we use plural form.)
    """
    cmd: list[str] = [
        str(sirius_bin),
        "--input",
        str(ms_path),
        "--output",
        str(project_dir),
        "formulas",
        "-p",
        profile,
    ]
    if max_mz is not None:
        cmd += ["--maxmz", str(max_mz)]
    if not no_structure:
        # CSI:FingerID DB search — requires academic/commercial login
        cmd += ["structures", "-d", structure_db]
    # POSTPROCESSING summaries (TSV by default)
    if summaries_dir is not None:
        cmd += ["summaries", "-o", str(summaries_dir), "--format", "tsv"]
    if extra_args:
        cmd += list(extra_args)
    return cmd


def run_sirius(
    ms_path: str | Path,
    project_dir: str | Path,
    *,
    sirius_bin: str | Path | None = None,
    summaries_dir: str | Path | None = None,
    profile: str = "orbitrap",
    no_structure: bool = False,
    structure_db: str = "BIO",
    extra_args: list[str] | None = None,
    timeout_s: float = 600.0,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Execute SIRIUS CLI on one ``.ms`` file.

    Returns dict with returncode, cmd, stdout/stderr tails, and paths.
    Does not raise on non-zero exit — caller inspects returncode.
    """
    bin_path = find_sirius_binary(sirius_bin)
    ms_path = Path(ms_path)
    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    if summaries_dir is None:
        summaries_dir = project_dir / "summaries"
    summaries_dir = Path(summaries_dir)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    if bin_path is None:
        return {
            "ok": False,
            "error": "SIRIUS binary not found (set SIRIUS_BIN or install CLI on PATH)",
            "cmd": None,
            "returncode": None,
            "project_dir": str(project_dir),
            "summaries_dir": str(summaries_dir),
        }

    cmd = build_sirius_command(
        sirius_bin=bin_path,
        ms_path=ms_path,
        project_dir=project_dir,
        summaries_dir=summaries_dir,
        profile=profile,
        no_structure=no_structure,
        structure_db=structure_db,
        extra_args=extra_args,
    )
    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "cmd": cmd,
            "returncode": 0,
            "project_dir": str(project_dir),
            "summaries_dir": str(summaries_dir),
        }

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        return {
            "ok": False,
            "error": f"timeout after {timeout_s}s",
            "cmd": cmd,
            "returncode": None,
            "stdout": (e.stdout or "")[-4000:] if isinstance(e.stdout, str) else "",
            "stderr": (e.stderr or "")[-4000:] if isinstance(e.stderr, str) else "",
            "project_dir": str(project_dir),
            "summaries_dir": str(summaries_dir),
        }
    except FileNotFoundError:
        return {
            "ok": False,
            "error": f"failed to execute {bin_path}",
            "cmd": cmd,
            "returncode": None,
            "project_dir": str(project_dir),
            "summaries_dir": str(summaries_dir),
        }

    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    # SIRIUS 6 may exit 0 while printing a hard Login ERROR and doing no work.
    login_blocked = "Login ERROR" in out or "Please Login to use the SIRIUS" in out
    ok = proc.returncode == 0 and not login_blocked
    return {
        "ok": ok,
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "")[-4000:],
        "stderr": (proc.stderr or "")[-4000:],
        "error": "SIRIUS CLI not logged in — run: sirius login" if login_blocked else None,
        "project_dir": str(project_dir),
        "summaries_dir": str(summaries_dir),
    }


def identify_spectrum(
    *,
    compound_id: str,
    precursor_mz: float,
    peaks: list[tuple[float, float]],
    work_dir: str | Path,
    ion_mode: str | None = None,
    sirius_bin: str | Path | None = None,
    top_k: int = 8,
    profile: str = "orbitrap",
    no_structure: bool = False,
    structure_db: str = "BIO",
    timeout_s: float = 600.0,
    dry_run: bool = False,
    parse_existing_only: bool = False,
) -> tuple[list[SiriusHit], dict[str, Any]]:
    """
    End-to-end: write .ms → run SIRIUS → parse hits.

    If ``parse_existing_only``, skip CLI and parse ``work_dir`` outputs.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    ms_path = work_dir / f"{compound_id}.ms"
    project_dir = work_dir / f"{compound_id}.sirius"
    summaries_dir = work_dir / f"{compound_id}_summaries"

    meta: dict[str, Any] = {
        "compound_id": compound_id,
        "ms_path": str(ms_path),
        "project_dir": str(project_dir),
        "summaries_dir": str(summaries_dir),
        "ion_mode": ion_mode,
    }

    if not parse_existing_only:
        write_sirius_ms(
            ms_path,
            compound_id=compound_id,
            precursor_mz=precursor_mz,
            peaks=peaks,
            ion_mode=ion_mode,
        )
        run_meta = run_sirius(
            ms_path,
            project_dir,
            sirius_bin=sirius_bin,
            summaries_dir=summaries_dir,
            profile=profile,
            no_structure=no_structure,
            structure_db=structure_db,
            timeout_s=timeout_s,
            dry_run=dry_run,
        )
        meta["run"] = run_meta
        if dry_run:
            return [], meta
        if not run_meta.get("ok") and not project_dir.exists() and not summaries_dir.exists():
            return [], meta
    else:
        meta["run"] = {"parse_existing_only": True}

    # Prefer summaries folder, then project tree
    hits = parse_sirius_project(summaries_dir, top_k=top_k)
    if not hits:
        hits = parse_sirius_project(project_dir, top_k=top_k)
    meta["n_hits"] = len(hits)
    return hits, meta


def hits_to_prompt_block(hits: Iterable[SiriusHit], *, max_n: int = 8) -> list[str]:
    """Format SIRIUS/CSI hits for LLM prompts."""
    hits = list(hits)[:max_n]
    lines = [
        "",
        "=== SIRIUS / CSI:FingerID (independent spectral structure search) ===",
    ]
    if not hits:
        lines.append("(no SIRIUS results for this spectrum)")
        return lines
    lines.append(
        "  Use high-confidence CSI structures when mass-consistent; "
        "combine with network neighbors and NIST, do not ignore mass gates."
    )
    for h in hits:
        smi = f" SMILES={h.smiles}" if h.smiles else ""
        ik = f" IK={h.inchikey}" if h.inchikey else ""
        nm = f" name={h.name}" if h.name else ""
        sc = f" csi={h.csi_score:.3f}" if h.csi_score is not None else ""
        cf = f" conf={h.confidence:.3f}" if h.confidence is not None else ""
        lines.append(
            f"  #{h.rank}: formula={h.formula or '?'} adduct={h.adduct or '?'}"
            f"{sc}{cf}{smi}{ik}{nm}"
        )
    return lines
