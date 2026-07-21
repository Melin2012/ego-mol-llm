"""Parse model output and validate SMILES / formula vs precursor mass."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


SMILES_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"("
    r"(?:Br|Cl|Si|Se|Na|Mg|Ca|Fe|Zn|As|B|C|N|O|P|S|F|I|H|c|n|o|p|s|"
    r"\[.*?\]|"
    r"[0-9@+\-=#$:/\\().%])+"
    r")"
)

JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


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
    parse_errors: list[str] = field(default_factory=list)


def _try_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        return Chem, Descriptors
    except Exception:
        return None, None


def canonicalize_smiles(smiles: str) -> str | None:
    Chem, _ = _try_rdkit()
    if Chem is None:
        return smiles.strip() or None
    mol = Chem.MolFromSmiles(smiles)
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


def adduct_mass_offset(adduct: str | None) -> float | None:
    if not adduct:
        return None
    a = adduct.replace(" ", "").lower()
    table = {
        "[m-h]-": -1.007825,
        "[m+h]+": 1.007825,
        "[m+na]+": 22.989218,
        "[m+k]+": 38.963158,
        "[m+nh4]+": 18.033823,
        "[m-h2o-h]-": -19.01839,
        "[m+h-h2o]+": -17.00274,
        "m-h": -1.007825,
        "m+h": 1.007825,
    }
    return table.get(a)


def check_mass(
    smiles: str,
    precursor_mz: float | None,
    adduct: str | None,
    tol_da: float = 0.05,
) -> tuple[bool | None, float | None, float | None]:
    """Return (ok, exact_mass, error_da)."""
    em = exact_mass_from_smiles(smiles)
    if em is None or precursor_mz is None:
        return None, em, None
    off = adduct_mass_offset(adduct)
    candidates = []
    if off is not None:
        candidates.append(em + off)
    else:
        # try common adducts
        for o in (-1.007825, 1.007825, 22.989218):
            candidates.append(em + o)
    errors = [abs(precursor_mz - c) for c in candidates]
    best = min(errors)
    return best <= tol_da, em, best


def extract_json(text: str) -> dict[str, Any] | None:
    # Prefer fenced json
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    blob = fence.group(1) if fence else None
    if blob is None:
        matches = list(JSON_BLOCK_RE.finditer(text))
        if not matches:
            return None
        blob = matches[-1].group(0)
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        # try to fix trailing commas
        cleaned = re.sub(r",\s*}", "}", blob)
        cleaned = re.sub(r",\s*]", "]", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None


def extract_smiles_fallback(text: str) -> str | None:
    # Look for explicit SMILES: lines
    for pat in [
        r"SMILES[:\s]+[`'\"]?([A-Za-z0-9@+\-=#$:/\\().%\[\]]+)",
        r"canonical SMILES[:\s]+[`'\"]?([A-Za-z0-9@+\-=#$:/\\().%\[\]]+)",
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip("`'\"")
    return None


def parse_model_output(
    text: str,
    precursor_mz: float | None = None,
    mass_tol_da: float = 0.05,
) -> ParsedPrediction:
    pred = ParsedPrediction(raw_text=text)
    data = extract_json(text)
    if data is None:
        pred.parse_errors.append("No JSON object found in model output")
        smi = extract_smiles_fallback(text)
        if smi:
            pred.smiles = smi
    else:
        pred.smiles = data.get("smiles") or None
        pred.name = data.get("iupac_or_common_name") or data.get("name")
        pred.formula = data.get("formula")
        pred.adduct = data.get("adduct")
        try:
            pred.confidence = float(data["confidence"]) if data.get("confidence") is not None else None
        except (TypeError, ValueError):
            pred.confidence = None
        pred.rationale = data.get("rationale")
        alts = data.get("alternatives") or []
        if isinstance(alts, list):
            pred.alternatives = [a for a in alts if isinstance(a, dict)]

    if pred.smiles:
        can = canonicalize_smiles(pred.smiles)
        if can is None:
            pred.smiles_valid = False
            pred.parse_errors.append(f"Invalid SMILES: {pred.smiles}")
        else:
            pred.smiles_valid = True
            pred.canonical_smiles = can
            ok, em, err = check_mass(can, precursor_mz, pred.adduct, tol_da=mass_tol_da)
            pred.exact_mass = em
            pred.mass_error_da = err
            pred.mass_ok = ok
    return pred
