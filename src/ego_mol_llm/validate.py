"""Parse model output and validate SMILES / formula vs precursor mass."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")

# Explicit field lines from ChemDFM-style free text
FIELD_PATTERNS: dict[str, re.Pattern[str]] = {
    "smiles": re.compile(
        r"(?im)^\s*(?:[-*•]\s*)?(?:canonical\s+)?smiles\s*[:：=]\s*[`'\"]?"
        r"([A-Za-z0-9@+\-=#$:/\\().%\[\]]+)"
    ),
    "name": re.compile(
        r"(?im)^\s*(?:[-*•]\s*)?(?:iupac_or_common_name|iupac(?:\s+name)?|common_name|name)\s*[:：=]\s*(.+?)\s*$"
    ),
    "formula": re.compile(
        r"(?im)^\s*(?:[-*•]\s*)?(?:molecular\s+)?formula\s*[:：=]\s*[`'\"]?"
        r"([A-Za-z0-9]+)"
    ),
    "adduct": re.compile(
        r"(?im)^\s*(?:[-*•]\s*)?adduct\s*[:：=]\s*[`'\"]?"
        r"(\[[^\]]+\][+-]?|[Mm](?:\+[A-Za-z0-9]+|\-[A-Za-z0-9]+)[+-]?)"
    ),
    "confidence": re.compile(
        r"(?im)^\s*(?:[-*•]\s*)?confidence\s*[:：=]\s*([0-9]*\.?[0-9]+)"
    ),
    "rationale": re.compile(
        r"(?im)^\s*(?:[-*•]\s*)?(?:rationale|reasoning|explanation)\s*[:：=]\s*(.+?)\s*$"
    ),
}

# Atomic masses for rough formula monoisotopic estimate
_ATOMIC = {
    "H": 1.007825,
    "C": 12.0,
    "N": 14.003074,
    "O": 15.994915,
    "F": 18.998403,
    "Na": 22.989769,
    "P": 30.973762,
    "S": 31.972071,
    "Cl": 34.968853,
    "K": 38.963707,
    "Br": 78.918338,
    "I": 126.904473,
}


@dataclass
class ParsedPrediction:
    smiles: str | None = None
    name: str | None = None
    formula: str | None = None
    adduct: str | None = None
    confidence: float | None = None
    rationale: str | None = None
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    raw_text: str = ""
    smiles_valid: bool | None = None
    canonical_smiles: str | None = None
    exact_mass: float | None = None
    mass_error_da: float | None = None
    mass_ok: bool | None = None
    matched_adduct: str | None = None
    parse_mode: str | None = None  # json | key_value | smiles_line | none
    parse_errors: list[str] = field(default_factory=list)
    source: str = "model"  # model | neighbor_rescue | hybrid


def _try_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        return Chem, Descriptors
    except Exception:
        return None, None


def _strip_quotes(s: str) -> str:
    return s.strip().strip("`").strip('"').strip("'")


def canonicalize_smiles(smiles: str) -> str | None:
    Chem, _ = _try_rdkit()
    smi = _strip_quotes(smiles or "")
    if not smi or smi.lower() in {"null", "none", "n/a", "na"}:
        return None
    if Chem is None:
        # Accept only if it looks like SMILES
        if re.fullmatch(r"[A-Za-z0-9@+\-=#$:/\\().%\[\]]+", smi):
            return smi
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def exact_mass_from_smiles(smiles: str) -> float | None:
    Chem, Descriptors = _try_rdkit()
    if Chem is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return float(Descriptors.ExactMolWt(mol))


def formula_to_mass(formula: str | None) -> float | None:
    if not formula:
        return None
    f = formula.strip()
    if not re.fullmatch(r"(?:[A-Z][a-z]?\d*)+", f):
        return None
    total = 0.0
    for el, num in re.findall(r"([A-Z][a-z]?)(\d*)", f):
        if el not in _ATOMIC:
            return None
        n = int(num) if num else 1
        total += _ATOMIC[el] * n
    return total


# Proton / common metal masses
H = 1.007825
NA = 22.989218
K = 38.963158
NH4 = 18.033823


def adduct_mass_offset(adduct: str | None) -> float | None:
    """
    Monomer adduct offsets only (neutral + offset → ion m/z).
    Multimer adducts are handled in theoretical_ion_mz / check_mass.
    """
    if not adduct:
        return None
    a = adduct.replace(" ", "").lower()
    table = {
        "[m-h]-": -H,
        "[m+h]+": H,
        "[m+na]+": NA,
        "[m+k]+": K,
        "[m+nh4]+": NH4,
        "[m-h2o-h]-": -19.01839,
        "[m+h-h2o]+": -17.00274,
        "[m+h-2h2o]+": -35.01339,
        "m-h": -H,
        "m+h": H,
        "[m]-": 0.0,
        "[m]+": 0.0,
    }
    return table.get(a)


# Even-electron monomer adducts (preferred for ESI)
COMMON_ADDUCT_OFFSETS: list[tuple[str, float]] = [
    ("[M-H]-", -H),
    ("[M+H]+", H),
    ("[M+Na]+", NA),
    ("[M+K]+", K),
    ("[M+NH4]+", NH4),
    ("[M-H2O-H]-", -19.01839),
    ("[M+H-H2O]+", -17.00274),  # phenol / alcohol water loss
    ("[M+H-2H2O]+", -35.01339),
]

# Radical cations — rare in ESI; only considered as last resort with tight tolerance
ODD_ELECTRON_ADDUCTS: list[tuple[str, float]] = [
    ("[M]+", 0.0),
    ("[M]-", 0.0),
]

# Multimer adducts: (name, n_copies, charge_offset)
# ion m/z = n * exact_mass + charge_offset
MULTIMER_ADDUCTS: list[tuple[str, int, float]] = [
    ("[2M-H]-", 2, -H),
    ("[2M+H]+", 2, H),
    ("[2M+Na]+", 2, NA),
    ("[2M+K]+", 2, K),
    ("[2M+NH4]+", 2, NH4),
    ("[2M+H-H2O]+", 2, -17.00274),
    ("[3M-H]-", 3, -H),
    ("[3M+H]+", 3, H),
    ("[3M+Na]+", 3, NA),
]


def monomer_mass_targets(
    precursor_mz: float,
    min_precursor_for_multimer: float = 250.0,
) -> list[tuple[str, float]]:
    """
    Implied monomer *ion* m/z values if precursor were a multimer.

    Only used when precursor is large enough that dimers are plausible.
    Does NOT include the precursor m/z itself (that would collapse half-mass
    scoring into ordinary near-isobar scoring).
    """
    if precursor_mz < min_precursor_for_multimer:
        return []
    targets: list[tuple[str, float]] = []
    for name, n, o in MULTIMER_ADDUCTS:
        mono_exact = (precursor_mz - o) / n
        if mono_exact < 80:
            continue
        targets.append((f"neutral via {name}", mono_exact))
        for ion_name, ion_o in COMMON_ADDUCT_OFFSETS:
            ion_mz = mono_exact + ion_o
            if ion_mz > 80:
                targets.append((f"neighbor ion via {name}/{ion_name}", ion_mz))
    return targets


def theoretical_ion_mz(
    exact_mass: float,
    include_multimer: bool = True,
    include_odd_electron: bool = False,
) -> list[tuple[str, float]]:
    """All theoretical precursor m/z values for a neutral mass."""
    out: list[tuple[str, float]] = []
    for name, o in COMMON_ADDUCT_OFFSETS:
        out.append((name, exact_mass + o))
    if include_odd_electron:
        for name, o in ODD_ELECTRON_ADDUCTS:
            out.append((name, exact_mass + o))
    if include_multimer:
        for name, n, o in MULTIMER_ADDUCTS:
            out.append((name, n * exact_mass + o))
    return out


def check_mass(
    smiles: str,
    precursor_mz: float | None,
    adduct: str | None = None,
    tol_da: float = 0.05,
    formula: str | None = None,
    allow_formula_fallback: bool = True,
    include_multimer: bool = True,
    include_odd_electron: bool = False,
    odd_electron_tol_da: float = 0.005,
) -> tuple[bool | None, float | None, float | None, str | None]:
    """
    Return (ok, exact_mass, error_da, matched_adduct).

    Matches even-electron monomer adducts (incl. [M+H-H2O]+) and multimers.
    Radical [M]+/[M]- only if include_odd_electron=True and within tight tol.
    """
    em = exact_mass_from_smiles(smiles)
    if em is None and allow_formula_fallback:
        em = formula_to_mass(formula)
    if em is None or precursor_mz is None:
        return None, em, None, None

    candidates = theoretical_ion_mz(
        em,
        include_multimer=include_multimer,
        include_odd_electron=False,
    )

    # Prefer user-stated adduct if known
    if adduct:
        a = adduct.replace(" ", "")
        al = a.lower()
        for name, n, o in MULTIMER_ADDUCTS:
            if name.lower().replace(" ", "") == al.lower():
                candidates.insert(0, (name, n * em + o))
        off = adduct_mass_offset(adduct)
        if off is not None:
            candidates.insert(0, (adduct, em + off))

    best_name, best_err = None, float("inf")
    for name, theo in candidates:
        err = abs(precursor_mz - theo)
        if err < best_err:
            best_err = err
            best_name = name

    # Optional radical ions only as last resort with tight tolerance
    if include_odd_electron or best_err > tol_da:
        for name, o in ODD_ELECTRON_ADDUCTS:
            theo = em + o
            err = abs(precursor_mz - theo)
            # only beat even-electron if clearly better AND within tight tol
            if err <= odd_electron_tol_da and err < best_err - 0.002:
                best_err = err
                best_name = name

    ok = best_err <= tol_da
    # Reject pure [M]+/[M]- matches unless extremely tight (ESI-unfriendly)
    if best_name in {"[M]+", "[M]-"} and best_err > odd_electron_tol_da:
        ok = False
    return ok, em, best_err, best_name if ok or best_err < 1.0 else best_name


def is_multimer_adduct(adduct: str | None) -> bool:
    if not adduct:
        return False
    a = adduct.lower().replace(" ", "")
    return a.startswith("[2m") or a.startswith("[3m") or a.startswith("2m") or a.startswith("3m")


def infer_multimer_adduct(
    precursor_mz: float | None,
    neighbor_mz: float | None,
    tol_da: float = 0.05,
) -> tuple[str | None, float | None]:
    """
    Infer multimer adduct by relating precursor m/z to a neighbor ion m/z.

    Example: neighbor 407.28 ≈ [M+H]+ and precursor 813.55 ≈ [2M+H]+
    → ( '[2M+H]+', error )
    """
    if precursor_mz is None or neighbor_mz is None:
        return None, None
    # Assume neighbor is a common monomer ion; back out neutral and test multimer ions
    best: tuple[str, float] | None = None
    for ion_name, ion_o in COMMON_ADDUCT_OFFSETS:
        mono_exact = float(neighbor_mz) - ion_o
        if mono_exact < 50:
            continue
        for m_name, n, m_o in MULTIMER_ADDUCTS:
            theo = n * mono_exact + m_o
            err = abs(float(precursor_mz) - theo)
            if best is None or err < best[1]:
                best = (m_name, err)
    if best and best[1] <= tol_da:
        return best[0], best[1]
    if best and best[1] <= 0.5:  # looser report
        return best[0], best[1]
    return None, best[1] if best else None


def extract_json(text: str) -> dict[str, Any] | None:
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    blob = fence.group(1) if fence else None
    if blob is None:
        matches = list(JSON_BLOCK_RE.finditer(text))
        if not matches:
            return None
        # Prefer the last JSON-looking block that has "smiles"
        blob = None
        for m in reversed(matches):
            if "smiles" in m.group(0).lower() or "formula" in m.group(0).lower():
                blob = m.group(0)
                break
        if blob is None:
            blob = matches[-1].group(0)
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        cleaned = re.sub(r",\s*}", "}", blob)
        cleaned = re.sub(r",\s*]", "]", cleaned)
        # single quotes -> double for simple cases
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None


def extract_key_value_fields(text: str) -> dict[str, Any]:
    """Parse ChemDFM-style free-text key: value lines."""
    out: dict[str, Any] = {}
    for key, pat in FIELD_PATTERNS.items():
        m = pat.search(text)
        if not m:
            continue
        val = _strip_quotes(m.group(1))
        if val.lower() in {"null", "none", "n/a", "na", "-"}:
            out[key] = None
            continue
        if key == "confidence":
            try:
                out[key] = float(val)
            except ValueError:
                continue
        else:
            out[key] = val
    return out


def extract_smiles_fallback(text: str) -> str | None:
    for pat in [
        r"(?i)(?:canonical\s+)?smiles\s*[:：=]\s*[`'\"]?([A-Za-z0-9@+\-=#$:/\\().%\[\]]+)",
        r"(?i)best\s+(?:structure|prediction)[^\n]*?([CNOcno][A-Za-z0-9@+\-=#$:/\\().%\[\]]{3,})",
    ]:
        m = re.search(pat, text)
        if m:
            cand = _strip_quotes(m.group(1))
            if canonicalize_smiles(cand):
                return cand
    return None


def _apply_fields(pred: ParsedPrediction, data: dict[str, Any]) -> None:
    if "smiles" in data and data["smiles"]:
        pred.smiles = str(data["smiles"]).strip() or None
    pred.name = data.get("iupac_or_common_name") or data.get("name") or pred.name
    if data.get("formula"):
        pred.formula = str(data["formula"]).strip()
    if data.get("adduct"):
        pred.adduct = str(data["adduct"]).strip()
    if data.get("confidence") is not None:
        try:
            pred.confidence = float(data["confidence"])
        except (TypeError, ValueError):
            pass
    if data.get("rationale"):
        pred.rationale = str(data["rationale"]).strip()
    alts = data.get("alternatives") or []
    if isinstance(alts, list):
        pred.alternatives = [a for a in alts if isinstance(a, dict)]


def validate_smiles_fields(
    pred: ParsedPrediction,
    precursor_mz: float | None,
    mass_tol_da: float,
) -> ParsedPrediction:
    """Canonicalize SMILES and check precursor mass consistency."""
    if not pred.smiles:
        return pred

    can = canonicalize_smiles(pred.smiles)
    if can is None:
        pred.smiles_valid = False
        pred.parse_errors.append(f"Invalid SMILES: {pred.smiles}")
        return pred

    pred.smiles_valid = True
    pred.canonical_smiles = can
    ok, em, err, matched = check_mass(
        can,
        precursor_mz,
        pred.adduct,
        tol_da=mass_tol_da,
        formula=pred.formula,
        allow_formula_fallback=True,
    )
    pred.exact_mass = em
    pred.mass_error_da = err
    pred.mass_ok = ok
    pred.matched_adduct = matched
    if matched and (not pred.adduct or pred.mass_ok):
        # Prefer adduct that actually fits
        if ok:
            pred.adduct = matched
    if ok is False:
        pred.parse_errors.append(
            f"Mass inconsistent: precursor m/z={precursor_mz}, "
            f"exact_mass={em}, best_error_Da={err}, tried_adduct={matched}"
        )
        # Downgrade confidence when mass fails
        if pred.confidence is None:
            pred.confidence = 0.15
        else:
            pred.confidence = min(pred.confidence, 0.25)
    return pred


def parse_model_output(
    text: str,
    precursor_mz: float | None = None,
    mass_tol_da: float = 0.05,
) -> ParsedPrediction:
    """
    Parse LLM output robustly:
    1) JSON block (preferred)
    2) key: value lines (ChemDFM free text)
    3) SMILES: line fallback
    Then validate SMILES and precursor mass.
    """
    pred = ParsedPrediction(raw_text=text or "")
    if not text or not text.strip():
        pred.parse_errors.append("Empty model output")
        pred.parse_mode = "none"
        return pred

    data = extract_json(text)
    if data is not None:
        pred.parse_mode = "json"
        _apply_fields(pred, data)
    else:
        kv = extract_key_value_fields(text)
        if kv:
            pred.parse_mode = "key_value"
            _apply_fields(pred, kv)
            # Map name key
            if "name" in kv and not pred.name:
                pred.name = kv["name"]
        else:
            pred.parse_mode = "none"
            pred.parse_errors.append("No JSON object found in model output")

    if not pred.smiles:
        smi = extract_smiles_fallback(text)
        if smi:
            pred.smiles = smi
            if pred.parse_mode in {None, "none"}:
                pred.parse_mode = "smiles_line"

    if not pred.rationale and pred.parse_mode in {"key_value", "smiles_line"}:
        # Keep a short head of free text as rationale
        head = " ".join(text.strip().split())
        pred.rationale = head[:400]

    return validate_smiles_fields(pred, precursor_mz, mass_tol_da)
