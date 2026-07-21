"""Deterministic offline backend for tests and CI (no model download)."""

from __future__ import annotations

import json
import re

from ego_mol_llm.backends.base import GenerationConfig, LLMBackend
from ego_mol_llm.validate import check_mass


def _extract_precursor_mz(user: str) -> float | None:
    m = re.search(r"precursor m/z\s*=\s*([0-9.]+)", user)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


class DryRunBackend(LLMBackend):
    """
    Offline heuristic that mimics mass-first ego reasoning:
    prefer mass-consistent library SMILES listed in the prompt.
    Never copy near-isobar SMILES without a mass check (avoids pachymic/cinnamic junk).
    """

    name = "dry-run"

    def generate(self, messages: list[dict[str, str]], config: GenerationConfig | None = None) -> str:
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        mz = _extract_precursor_mz(user)

        # Prefer pre-filtered mass-consistent library block (monomer or multimer)
        mass_block = re.search(
            r"MASS-CONSISTENT LIBRARY SMILES.*?\n((?:0\d\..*\n)+)",
            user,
            re.S,
        )
        if mass_block:
            for line in mass_block.group(1).strip().splitlines():
                smi_m = re.search(r"SMILES=([A-Za-z0-9@+\-=#$:/\\().%\[\]]+)", line)
                if not smi_m:
                    continue
                smiles = smi_m.group(1)
                name_m = re.search(r"name=(.+)$", line)
                name = name_m.group(1).strip() if name_m else None
                adduct_m = re.search(r"adduct~(\[[^\]]+\][+-]?)", line)
                adduct = adduct_m.group(1) if adduct_m else "[M+H]+"
                # Optional verify
                if mz is not None:
                    ok, _, err, matched = check_mass(smiles, mz, adduct, tol_da=0.1)
                    if ok is False:
                        continue
                    if matched:
                        adduct = matched
                payload = {
                    "smiles": smiles,
                    "iupac_or_common_name": name,
                    "formula": None,
                    "adduct": adduct,
                    "confidence": 0.82 if "2M" in adduct or "3M" in adduct else 0.8,
                    "rationale": (
                        "Dry-run mass-first: selected top mass-consistent library SMILES "
                        f"(adduct {adduct})."
                    ),
                    "alternatives": [],
                }
                return (
                    "DRY-RUN (mass-consistent neighbor).\n\n```json\n"
                    + json.dumps(payload, indent=2)
                    + "\n```"
                )

        # Half-mass section only when prompt shows half-mass stars and precursor is large
        if mz is not None and mz >= 250:
            half = re.search(
                r"HALF-MASS / MULTIMER-MONOMER NEIGHBORS.*?\n((?:0\d\..*\n)+)",
                user,
                re.S,
            )
            if half:
                for line in half.group(1).splitlines():
                    if "SMILES=" not in line or "HALF-MASS" not in line and "half-mass" not in line:
                        # still allow any SMILES line in this section
                        pass
                    smi_m = re.search(r"SMILES=([A-Za-z0-9@+\-=#$:/\\().%\[\]]+)", line)
                    if not smi_m:
                        continue
                    smiles = smi_m.group(1)
                    ok, _, _, matched = check_mass(
                        smiles, mz, None, tol_da=0.1, include_multimer=True
                    )
                    if ok is False:
                        continue
                    adduct = matched or "[2M+H]+"
                    payload = {
                        "smiles": smiles,
                        "iupac_or_common_name": None,
                        "formula": None,
                        "adduct": adduct,
                        "confidence": 0.78,
                        "rationale": (
                            "Dry-run half-mass: multimer-monomer neighbor SMILES; "
                            f"precursor treated as {adduct}."
                        ),
                        "alternatives": [],
                    }
                    return (
                        "DRY-RUN (half-mass multimer).\n\n```json\n"
                        + json.dumps(payload, indent=2)
                        + "\n```"
                    )

        # ★ NEAR-ISOBAR only — require mass check when possible
        iso = re.search(
            r"NEAR-ISOBAR NEIGHBORS.*?\n((?:0\d\..*\n)+)",
            user,
            re.S,
        )
        if iso and mz is not None:
            for line in iso.group(1).splitlines():
                if "★ NEAR-ISOBAR" not in line and "near-isobar" not in line:
                    continue
                smi_m = re.search(r"SMILES=([A-Za-z0-9@+\-=#$:/\\().%\[\]]+)", line)
                if not smi_m:
                    continue
                smiles = smi_m.group(1)
                ok, _, err, matched = check_mass(smiles, mz, None, tol_da=0.05)
                if ok is not True:
                    continue  # do not copy unvalidated / failing SMILES
                payload = {
                    "smiles": smiles,
                    "iupac_or_common_name": None,
                    "formula": None,
                    "adduct": matched or "[M+H]+",
                    "confidence": 0.75,
                    "rationale": (
                        f"Dry-run: near-isobar SMILES passed mass check "
                        f"(err={err}, adduct={matched})."
                    ),
                    "alternatives": [],
                }
                return (
                    "DRY-RUN (validated near-isobar).\n\n```json\n"
                    + json.dumps(payload, indent=2)
                    + "\n```"
                )

        # Legacy MTCA-style keyword fallback
        conf = 0.35
        smiles = None
        name = None
        formula = None
        rationale = (
            "Dry-run: no mass-validated neighbor SMILES; abstaining "
            "(noisy isobars not copied)."
        )

        if re.search(r"5470-37-1|tetrahydroharmane|Tetrahydroharmane", user, re.I):
            smiles = "CC1NC(Cc2c1[nH]c1ccccc21)C(=O)O"
            name = "1-methyl-1,2,3,4-tetrahydro-beta-carboline-3-carboxylic acid"
            formula = "C13H14N2O2"
            conf = 0.82
            rationale = (
                "Dry-run heuristic: high-cosine neighbors include CAS 5470-37-1 / "
                "tetrahydroharmane-3-carboxylic acid at m/z ~229."
            )
        elif re.search(r"C13H14N2O2", user):
            smiles = "CC1NC(Cc2c1[nH]c1ccccc21)C(=O)O"
            name = "putative C13H14N2O2 tetrahydro-beta-carboline carboxylic acid"
            formula = "C13H14N2O2"
            conf = 0.55
            rationale = "Dry-run heuristic: formula C13H14N2O2 appears among neighbors."

        payload = {
            "smiles": smiles,
            "iupac_or_common_name": name,
            "formula": formula,
            "adduct": "[M-H]-" if smiles else None,
            "confidence": conf,
            "rationale": rationale,
            "alternatives": [],
        }
        return "DRY-RUN reasoning complete.\n\n```json\n" + json.dumps(payload, indent=2) + "\n```"
