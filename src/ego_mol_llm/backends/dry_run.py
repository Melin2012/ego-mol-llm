"""Deterministic offline backend for tests and CI (no model download)."""

from __future__ import annotations

import json
import re

from ego_mol_llm.backends.base import GenerationConfig, LLMBackend


class DryRunBackend(LLMBackend):
    """
    Offline heuristic that mimics mass-first ego reasoning:
    prefer mass-consistent library SMILES listed in the prompt.
    """

    name = "dry-run"

    def generate(self, messages: list[dict[str, str]], config: GenerationConfig | None = None) -> str:
        user = next((m["content"] for m in messages if m["role"] == "user"), "")

        # Prefer pre-filtered mass-consistent library block if present
        mass_block = re.search(
            r"MASS-CONSISTENT LIBRARY SMILES FROM NEIGHBORS.*?\n((?:0\d\..*\n)+)",
            user,
            re.S,
        )
        if mass_block:
            first = mass_block.group(1).strip().splitlines()[0]
            smi_m = re.search(r"SMILES=([A-Za-z0-9@+\-=#$:/\\().%\[\]]+)", first)
            name_m = re.search(r"name=(.+)$", first)
            if smi_m:
                smiles = smi_m.group(1)
                name = name_m.group(1).strip() if name_m else None
                payload = {
                    "smiles": smiles,
                    "iupac_or_common_name": name,
                    "formula": None,
                    "adduct": "[M+H]+",
                    "confidence": 0.8,
                    "rationale": (
                        "Dry-run mass-first: selected top mass-consistent near-isobar/"
                        "annotated neighbor SMILES from the ego network."
                    ),
                    "alternatives": [],
                }
                return "DRY-RUN (mass-consistent neighbor).\n\n```json\n" + json.dumps(
                    payload, indent=2
                ) + "\n```"

        # Near-isobar section with SMILES
        iso = re.search(
            r"NEAR-ISOBAR NEIGHBORS.*?\n((?:0\d\..*\n)+)",
            user,
            re.S,
        )
        if iso:
            for line in iso.group(1).splitlines():
                if "SMILES=" not in line:
                    continue
                smi_m = re.search(r"SMILES=([A-Za-z0-9@+\-=#$:/\\().%\[\]]+)", line)
                if smi_m:
                    payload = {
                        "smiles": smi_m.group(1),
                        "iupac_or_common_name": None,
                        "formula": None,
                        "adduct": "[M+H]+",
                        "confidence": 0.72,
                        "rationale": "Dry-run: used first near-isobar neighbor with SMILES.",
                        "alternatives": [],
                    }
                    return "DRY-RUN (near-isobar SMILES).\n\n```json\n" + json.dumps(
                        payload, indent=2
                    ) + "\n```"

        # Legacy MTCA-style keyword fallback
        conf = 0.35
        smiles = None
        name = None
        formula = None
        rationale = "Dry-run backend: no mass-consistent neighbor SMILES found in prompt."

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
            "adduct": "[M-H]-",
            "confidence": conf,
            "rationale": rationale,
            "alternatives": [],
        }
        return "DRY-RUN reasoning complete.\n\n```json\n" + json.dumps(payload, indent=2) + "\n```"
